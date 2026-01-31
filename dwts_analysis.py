# dwts_analysis.py
"""
DWTS Analysis toolkit
- Implements DataProcessor, VoteEstimator (wrapper), VotingSimulator,
  MethodEvaluator, TOPSISDecision, SensitivityAnalyzer, VisualizationEngine
- Pipeline class DWTSPipeline to orchestrate tasks per season
Author: ChatGPT-generated (adapt to your needs)
"""
import os
import json
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# try to import user's FinalFanVoteEstimator if present
try:
    from Problem1_Final import FinalFanVoteEstimator

    _HAS_EXTERNAL_ESTIMATOR = True
except Exception:
    FinalFanVoteEstimator = None
    _HAS_EXTERNAL_ESTIMATOR = False


# ---------- Utilities ----------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient; expects 1D non-negative array"""
    x = np.array(x, dtype=float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = x + 1e-12  # avoid zeros exactly
    x_sorted = np.sort(x)
    n = x.size
    cumx = np.cumsum(x_sorted)
    return (2.0 * np.sum((np.arange(1, n + 1) * x_sorted))) / (n * cumx[-1]) - (
        n + 1
    ) / n


def coeff_variation(x: np.ndarray) -> float:
    """Coefficient of variation (std/mean). Returns large value if mean ~ 0"""
    x = np.array(x, dtype=float)
    mu = x.mean()
    sigma = x.std(ddof=0)
    if abs(mu) < 1e-12:
        return np.inf
    return sigma / mu


# ---------- DataProcessor ----------
class DataProcessor:
    """
    Load CSV, extract per-season matrices of judge scores (X), elimination order, names.
    Provides robust handling of N/A, 0 after elimination, variable weeks.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.raw = pd.read_csv(csv_path)
        self.season_index = sorted(self.raw["season"].unique())

    def list_seasons(self):
        return self.season_index

    def prepare_season(self, season_num: int, debug: bool = False):
        """
        Returns:
          X: np.ndarray (n_contestants, T_weeks) averaged judge scores per week
          e: np.ndarray elimination_order_indices (in contestant index space)
          names: list of contestant names
          elimination_weeks: np.ndarray of weeks (0-based) when each was eliminated in e order
        """
        # Try to reuse enhanced_prepare_season_data if available in Problem1_Final (user's file)
        if _HAS_EXTERNAL_ESTIMATOR and hasattr(
            __import__("Problem1_Final"), "enhanced_prepare_season_data"
        ):
            mod = __import__("Problem1_Final")
            return mod.enhanced_prepare_season_data(season_num, self.raw, debug=debug)

        # Fallback implementation (robust)
        df = self.raw[self.raw["season"] == season_num].reset_index(drop=True)
        names = df["celebrity_name"].astype(str).tolist()
        N = len(names)
        # extract week numbers by scanning column names
        week_indices = set()
        for c in df.columns:
            if c.startswith("week") and "judge" in c:
                try:
                    w = int(c.split("_")[0].replace("week", ""))
                    week_indices.add(w)
                except:
                    continue
        if not week_indices:
            # try alternative detection
            max_week = 0
            for c in df.columns:
                if "week" in c and "judge" in c:
                    max_week = max(max_week, 1)
            if max_week == 0:
                raise ValueError(f"No week judge columns found for season {season_num}")
        max_week = max(week_indices)
        T = max_week
        X = np.zeros((N, T))
        for i in range(N):
            for w in range(1, T + 1):
                cols = [c for c in df.columns if f"week{w}_" in c and "judge" in c]
                vals = []
                for col in cols:
                    v = df.at[i, col]
                    if pd.isna(v) or v == "N/A":
                        continue
                    try:
                        fv = float(v)
                        vals.append(fv)
                    except:
                        continue
                if len(vals) > 0:
                    X[i, w - 1] = np.mean(vals)
                else:
                    # try to copy previous week if available
                    if w > 1:
                        X[i, w - 1] = X[i, w - 2]
        # determine elimination order from "results" column
        elim_info = []
        finalists = []
        for i in range(N):
            res = str(df.at[i, "results"])
            if any(s in res for s in ["1st", "Winner", "Champion"]):
                # champion considered survived
                continue
            # try parse "Eliminated Week k"
            import re

            m = re.search(r"Week\s*(\d+)", res)
            if m:
                wk = int(m.group(1)) - 1
            elif "Eliminated" in res:
                wk = T - 1
            else:
                wk = T - 1
            elim_info.append((i, wk, df.at[i, "results"]))
        # sort by week then by judge score at that week
        elim_info.sort(key=lambda x: (x[1], X[x[0], x[1]] if x[1] < X.shape[1] else 0))
        e = np.array([it[0] for it in elim_info], dtype=int)
        elimination_weeks = np.array([it[1] for it in elim_info], dtype=int)
        # trim X to last elimination week if desired
        if len(elimination_weeks) > 0:
            T_model = max(elimination_weeks) + 1
            if T_model < X.shape[1]:
                X = X[:, :T_model]
        return X, e, names, elimination_weeks


# ---------- VoteEstimator (wrapper) ----------
class VoteEstimator:
    """
    Wraps either user's FinalFanVoteEstimator if available, otherwise
    provides a built-in estimator (a simpler variant).
    Interface:
      estimator = VoteEstimator(method='rank'/'percent', config=dict(...))
      estimator.fit(X, e, elimination_weeks)
      V = estimator.estimated_votes   # normalized per week columns sum to 1
    """

    def __init__(self, method="rank", config: Optional[Dict] = None):
        self.method = method
        self.config = config or {}
        self.estimator = None
        self.estimated_votes = None

    def fit(self, X: np.ndarray, e: np.ndarray, elimination_weeks: np.ndarray):
        if _HAS_EXTERNAL_ESTIMATOR and FinalFanVoteEstimator is not None:
            # use external class (preferred)
            fe = FinalFanVoteEstimator(
                method=self.method,
                lambda_constraint=self.config.get("lambda_constraint", 10000.0),
                lambda_smooth=self.config.get("lambda_smooth", 0.1),
                lambda_regularization=self.config.get("lambda_regularization", 0.01),
                verbose=self.config.get("verbose", 0),
            )
            fe.fit(
                X,
                e,
                elimination_weeks=elimination_weeks.tolist(),
                max_iter=self.config.get("maxiter", 2000),
            )
            self.estimator = fe
            self.estimated_votes = fe.fan_votes_
            return self
        # fallback: simple parameterized model with constrained optimization
        N, T = X.shape

        def _compute_votes_from_params(theta):
            # params: alpha_i (N), beta (1), gamma_t (T)
            alpha = theta[:N]
            beta = theta[N]
            gamma = theta[N + 1 :]
            logV = 1.0 / (1.0 + np.exp(-alpha[:, None])) + beta * X + gamma[None, :]
            V = np.exp(np.clip(logV, -50, 50))
            # normalize per week
            col_sums = V.sum(axis=0)
            col_sums[col_sums == 0] = 1.0
            return V / col_sums

        # initial guess
        x0 = np.concatenate([np.zeros(N), np.array([1.0]), np.zeros(T)])
        bounds = [(-5, 5)] * N + [(0.0, 3.0)] + [(-3, 3)] * T

        # objective: penalize constraint violations + smoothness + regularization
        def objective(theta):
            V = _compute_votes_from_params(theta)
            # hard constraints similar to earlier: ensure eliminated contestant gets lowest combined metric
            penalty = 0.0
            for idx, week in enumerate(elimination_weeks):
                elim = e[idx]
                active = [
                    i for i in range(N) if not (i in list(e[:idx]))
                ]  # approximate
                if elim not in active:
                    continue
                aidx = active.index(elim)
                week_X = X[active, week]
                week_V = V[active, week]
                if self.method == "rank":
                    X_rank = np.argsort(np.argsort(-week_X))
                    V_rank = np.argsort(np.argsort(-week_V))
                    combined = X_rank + V_rank
                    others = np.delete(combined, aidx)
                    if np.any(others <= combined[aidx]):
                        penalty += np.sum(
                            np.maximum(0, (others - combined[aidx]) + 0.1)
                        )
                else:
                    X_p = week_X / (week_X.sum() + 1e-12)
                    V_p = week_V / (week_V.sum() + 1e-12)
                    combined = X_p + V_p
                    others = np.delete(combined, aidx)
                    if np.any(others <= combined[aidx]):
                        penalty += np.sum(
                            np.maximum(0, (combined[aidx] - others) + 1e-3)
                        )
            # smoothness
            smooth = np.sum(
                np.abs(
                    _compute_votes_from_params(theta)[:, 1:]
                    - _compute_votes_from_params(theta)[:, :-1]
                )
            )
            reg = np.sum(theta**2)
            return 1e4 * penalty + 0.1 * smooth + 0.01 * reg

        res = differential_evolution(objective, bounds=bounds, maxiter=50, seed=42)
        res_local = minimize(
            objective, res.x, method="SLSQP", bounds=bounds, options={"maxiter": 500}
        )
        theta_hat = res_local.x
        V_hat = _compute_votes_from_params(theta_hat)
        self.estimated_votes = V_hat
        self.estimator = {"params": theta_hat}
        return self


# ---------- VotingSimulator ----------
class VotingSimulator:
    """
    Simulate elimination process under rank / percent methods.
    Given judge score matrix X and fan votes matrix V (normalized per week),
    compute weekly combined scores and simulate eliminations. Supports variant:
    - judges_choose_bottom_two (True/False): when True, bottom two are chosen by combined,
      but judges vote to eliminate (choose higher judge score to survive)
    """

    def __init__(self, X: np.ndarray, V: np.ndarray, names: List[str]):
        self.X = X.copy()
        self.V = V.copy()
        self.names = names.copy()
        self.N, self.T = X.shape

    @staticmethod
    def _ranks(scores: np.ndarray):
        # average ranking for ties; higher score -> better rank 1..n
        order = np.argsort(-scores)  # descending
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        # adjust ties to average rank
        unique_vals, inv = np.unique(scores, return_inverse=True)
        for i, val in enumerate(unique_vals):
            idx = np.where(inv == i)[0]
            if len(idx) > 1:
                avg_rank = ranks[idx].mean()
                ranks[idx] = avg_rank
        return ranks

    def simulate(self, method="rank", judges_choose_bottom_two=False):
        """
        Returns:
          elimination_sequence: list of contestant indices in order eliminated (0-based)
          elim_weeks: corresponding weeks (0-based)
          history: list of dicts with keys per week (combined_scores, active_indices)
        """
        active = list(range(self.N))
        elimination_sequence = []
        elim_weeks = []
        history = []
        for week in range(self.T):
            if len(active) <= 1:
                break
            week_X = self.X[active, week]
            week_V = (
                self.V[active, week]
                if week < self.V.shape[1]
                else np.ones(len(active)) / len(active)
            )
            if method == "rank":
                X_rank = self._ranks(week_X)
                V_rank = self._ranks(week_V)
                combined = X_rank + V_rank  # smaller better (1+1 best)
                # elimination: max combined (worst)
                if judges_choose_bottom_two:
                    # bottom two by combined (largest combined)
                    sorted_idx = np.argsort(combined)  # ascending => best first
                    bottom_two_idx = (
                        sorted_idx[-2:] if len(sorted_idx) >= 2 else sorted_idx[-1:]
                    )
                    # judges choose: pick one with lower judge score to eliminate? typical: judges vote to save better judge score
                    judge_scores = week_X[bottom_two_idx]
                    # element to eliminate = one with smaller judge score
                    elim_local = bottom_two_idx[np.argmin(judge_scores)]
                else:
                    elim_local = int(np.argmax(combined))
            else:  # percent
                X_p = week_X / (week_X.sum() + 1e-12)
                V_p = week_V / (week_V.sum() + 1e-12)
                combined = (
                    X_p + V_p
                )  # larger better? In Problem appendix percent small => eliminated min combined percent
                # elimination: min combined
                if judges_choose_bottom_two:
                    sorted_idx = np.argsort(combined)  # ascending => worst first
                    bottom_two_idx = (
                        sorted_idx[:2] if len(sorted_idx) >= 2 else sorted_idx[:1]
                    )
                    # judges vote: eliminate one with lower judge percent among bottom two
                    judge_percent = X_p[bottom_two_idx]
                    elim_local = bottom_two_idx[np.argmin(judge_percent)]
                else:
                    elim_local = int(np.argmin(combined))
            elim_contestant = active[elim_local]
            elimination_sequence.append(elim_contestant)
            elim_weeks.append(week)
            history.append(
                {
                    "week": week,
                    "active": active.copy(),
                    "combined": combined.copy(),
                    "method": method,
                }
            )
            # remove eliminated
            active.pop(elim_local)
        return elimination_sequence, elim_weeks, history


class MethodEvaluator:
    """
    Compute the five criteria between two methods (rank vs percent) for a season.
    This refactor provides method-specific evaluation: each criterion depends on
    the method's combined scores / final placements derived from simulation.
    """

    def __init__(self, names: List[str]):
        self.names = names

    @staticmethod
    def kendall_tau_b(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
        try:
            tau, p = kendalltau(rank_a, rank_b, nan_policy="omit")
            if np.isnan(tau):
                return 0.0
            return float(tau)
        except Exception:
            return 0.0

    @staticmethod
    def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
        try:
            rho, p = spearmanr(a, b, nan_policy="omit")
            if np.isnan(rho):
                return 0.0
            return float(rho)
        except Exception:
            return 0.0

    @staticmethod
    def grey_relation_grade(
        reference: np.ndarray, comparators: List[np.ndarray], rho=0.5
    ) -> float:
        ref = np.array(reference, dtype=float)

        # normalize
        def norm(s):
            s = np.array(s, dtype=float)
            mn, mx = np.min(s), np.max(s)
            if mx - mn < 1e-12:
                return np.zeros_like(s)
            return (s - mn) / (mx - mn)

        ref_n = norm(ref)
        grades = []
        for comp in comparators:
            comp_n = norm(comp)
            diff = np.abs(ref_n - comp_n)
            diff_min = np.min(diff)
            diff_max = np.max(diff)
            coeffs = (diff_min + rho * diff_max) / (diff + rho * diff_max + 1e-12)
            grades.append(coeffs.mean())
        if not grades:
            return 0.0
        return float(np.mean(grades))

    @staticmethod
    def stability_inverse_cv(series_list: List[np.ndarray]) -> float:
        cvs = []
        for s in series_list:
            cv = coeff_variation(np.array(s))
            cvs.append(cv if np.isfinite(cv) else 1e6)
        mean_cv = np.mean(cvs) if cvs else 1e6
        return float(1.0 / (mean_cv + 1e-12))

    @staticmethod
    def fairness_1_minus_gini(series: np.ndarray) -> float:
        g = gini_coefficient(series)
        return float(max(0.0, 1.0 - g))

    # --- new helpers ---
    @staticmethod
    def _final_placement_from_elim_sequence(
        n_contestants: int, elimination_sequence: List[int]
    ) -> np.ndarray:
        """
        Build placement vector: 1 best ... N worst.
        elimination_sequence is list of eliminated contestants in order (first eliminated -> worst).
        """
        place = np.zeros(n_contestants, dtype=int) + (n_contestants + 1)
        for idx, contestant in enumerate(elimination_sequence):
            place[contestant] = n_contestants - idx
        rem = [i for i in range(n_contestants) if place[i] > n_contestants]
        for i, contestant in enumerate(reversed(rem), 1):
            place[contestant] = i
        return place

    def compute_method_combined_series(
        self, X: np.ndarray, V: np.ndarray, method: str
    ) -> np.ndarray:
        """
        For each week, compute method-specific combined score per contestant.
        Returns array shape (n_contestants, T_weeks) of combined (non-normalized) scores.
        method: "rank" or "percent"
        """
        N, T = X.shape
        combined = np.zeros((N, T), dtype=float)
        for w in range(T):
            judge_scores = X[:, w]
            fan_scores = V[:, w]
            if method == "rank":
                # ranks: lower rank value = better (1 is best). We convert to score where larger is better.
                jr = np.argsort(-judge_scores).argsort() + 1  # 1..N
                fr = np.argsort(-fan_scores).argsort() + 1
                # convert to score: higher better => invert rank
                judge_score = (N + 1) - jr
                fan_score = (N + 1) - fr
                combined[:, w] = judge_score + fan_score
            else:  # percent
                j_p = judge_scores / (np.sum(judge_scores) + 1e-12)
                f_p = fan_scores / (np.sum(fan_scores) + 1e-12)
                combined[:, w] = j_p + f_p
        return combined

    def evaluate_single_method(
        self, X: np.ndarray, V: np.ndarray, elimination_sequence: List[int], method: str
    ) -> Dict[str, float]:
        """
        Evaluate the five criteria for a single method.
        method: 'rank' or 'percent'
        """
        N, T = X.shape
        # final placement from elimination sequence
        placement_method = self._final_placement_from_elim_sequence(
            N, elimination_sequence
        )
        # judge placement (aggregate by judge ranks across weeks)
        judge_ranks_week = []
        for w in range(T):
            judge_scores = X[:, w]
            judge_rank = np.argsort(-judge_scores).argsort() + 1
            judge_ranks_week.append(judge_rank)
        judge_total = np.sum(np.vstack(judge_ranks_week), axis=1)
        judge_placement = np.argsort(judge_total).argsort() + 1

        # 1) Historical consistency: Kendall tau between judge_placement and method placement
        historical_consistency = self.kendall_tau_b(judge_placement, placement_method)

        # 2) Fan influence: compare fan-only final placement vs method placement
        # fan-only placement derived from V aggregated over weeks (sum or final week). Use sum across weeks as fan signal.
        fan_sum = np.sum(V, axis=1)
        fan_placement = np.argsort(fan_sum).argsort() + 1
        fan_influence = self.spearman_rho(fan_placement, placement_method)

        # 3) Controversy (GRA): compare judge percent time series vs method combined percent time series
        judge_percent_series = np.array(
            [X[:, w] / (np.sum(X[:, w]) + 1e-12) for w in range(T)]
        ).T  # N x T
        combined_series = self.compute_method_combined_series(X, V, method)  # N x T
        # normalize combined per week to percent
        combined_percent_series = np.array(
            [
                combined_series[:, w] / (np.sum(combined_series[:, w]) + 1e-12)
                for w in range(T)
            ]
        ).T
        gra_scores = []
        for i in range(N):
            gra = self.grey_relation_grade(
                judge_percent_series[i, :], [combined_percent_series[i, :]]
            )
            gra_scores.append(gra)
        controversy_gra = float(np.mean(gra_scores))

        # 4) Stability: inverse CV of each contestant's combined percent time series
        stability = self.stability_inverse_cv(
            [combined_percent_series[i, :] for i in range(N)]
        )

        # 5) Fairness: 1 - Gini of final combined percent distribution (last week)
        final_combined = (
            combined_percent_series[:, -1]
            if combined_percent_series.shape[1] >= 1
            else combined_percent_series[:, 0]
        )
        fairness = self.fairness_1_minus_gini(final_combined)

        return {
            "historical_consistency": float(historical_consistency),
            "fan_influence": float(fan_influence),
            "controversy_gra": float(controversy_gra),
            "stability": float(stability),
            "fairness": float(fairness),
            "placement_method": placement_method.tolist(),
            "judge_placement": judge_placement.tolist(),
            "fan_placement": fan_placement.tolist(),
        }

    def evaluate_methods(
        self,
        X: np.ndarray,
        V: np.ndarray,
        V_alt: np.ndarray,
        method_rank_sim: Tuple[List[int], List[int], List[dict]],
        method_percent_sim: Tuple[List[int], List[int], List[dict]],
    ) -> Tuple[Dict, Dict]:
        """
        Wrapper: return two dicts (eval_rank, eval_percent) computed via evaluate_single_method
        method_rank_sim/percent_sim are the simulate() outputs (seq, weeks, history)
        """
        rank_seq, _, _ = method_rank_sim
        percent_seq, _, _ = method_percent_sim
        eval_rank = self.evaluate_single_method(X, V, rank_seq, method="rank")
        eval_percent = self.evaluate_single_method(X, V, percent_seq, method="percent")
        return eval_rank, eval_percent


# ---------- TOPSISDecision ----------
class TOPSISDecision:
    """
    Implements TOPSIS with normalization, weighting, ideal solutions
    Supports benefit/cost criteria flags and returns closeness coefficients and ranking.
    """

    def __init__(self, criteria_names: List[str], benefit_criteria: List[bool]):
        self.criteria_names = criteria_names
        self.benefit_criteria = (
            benefit_criteria  # True if benefit (higher better), False if cost
        )
        assert len(criteria_names) == len(benefit_criteria)

    @staticmethod
    def entropy_weights(data: np.ndarray) -> np.ndarray:
        """
        data: m x n (alternatives x criteria), assumed non-negative
        return: n-length weight vector (entropy method)
        """
        X = np.array(data, dtype=float)
        m, n = X.shape
        # avoid zeros
        X = X + 1e-12
        P = X / X.sum(axis=0, keepdims=True)
        # compute entropy
        k = 1.0 / math.log(m + 1e-12)
        ent = -k * (P * np.log(P)).sum(axis=0)
        d = 1 - ent
        w = d / (np.sum(d) + 1e-12)
        return w

    @staticmethod
    def critic_weights(data: np.ndarray) -> np.ndarray:
        """
        CRITIC method: uses standard deviation and correlation
        """
        X = np.array(data, dtype=float)
        # standardize by column
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
        std = X_std.std(axis=0, ddof=0)
        # correlation matrix
        R = np.corrcoef(X_std.T)
        # information: sum(1 - r_ij)
        C = std * (np.sum(1 - R, axis=1))
        w = C / (np.sum(C) + 1e-12)
        return w

    @staticmethod
    def combine_weights(w1: np.ndarray, w2: np.ndarray, alpha=0.5) -> np.ndarray:
        w = alpha * w1 + (1 - alpha) * w2
        w = w / (np.sum(w) + 1e-12)
        return w

    def run(
        self,
        alternatives: List[str],
        criteria_matrix: np.ndarray,
        weight_method: str = "entropy+critic",
        alpha=0.5,
    ) -> Dict:
        """
        criteria_matrix: m x n (alternatives x criteria)
        weight_method: "entropy", "critic", "entropy+critic"
        """
        # --- [DEBUG B] 最稳健的版本：直接看输入 ---
        print("\n" + "="*50)
        print(f"=== DEBUG B: TOPSIS INPUT CHECK ===")
        A = np.array(criteria_matrix, dtype=float)
        print(f"Matrix A shape: {A.shape}")
        print("Matrix A content (Row 0 is Rank, Row 1 is Percent):")
        print(A)

        A = np.array(criteria_matrix, dtype=float)
        m, n = A.shape
        # normalize by column (vector norm)
        A_norm = A / (np.sqrt((A**2).sum(axis=0, keepdims=True)) + 1e-12)
        # compute weights
        if weight_method == "entropy":
            w = self.entropy_weights(A)
        elif weight_method == "critic":
            w = self.critic_weights(A)
        else:
            w_e = self.entropy_weights(A)
            w_c = self.critic_weights(A)
            w = self.combine_weights(w_e, w_c, alpha=alpha)
        # weighted normalized
        V = A_norm * w
        # ideal positive/negative
        ideal_pos = np.zeros(n)
        ideal_neg = np.zeros(n)
        for j in range(n):
            if self.benefit_criteria[j]:
                ideal_pos[j] = V[:, j].max()
                ideal_neg[j] = V[:, j].min()
            else:
                # cost criterion: ideal pos = min, ideal neg = max
                ideal_pos[j] = V[:, j].min()
                ideal_neg[j] = V[:, j].max()
        # distances
        d_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
        d_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))
        closeness = d_neg / (d_pos + d_neg + 1e-12)
        rank_idx = np.argsort(-closeness)  # higher closeness better
        results = {
            "alternatives": alternatives,
            "weights": w.tolist(),
            "closeness": closeness.tolist(),
            "rank_idx": rank_idx.tolist(),
            "d_pos": d_pos.tolist(),
            "d_neg": d_neg.tolist(),
        }
        return results


# ---------- SensitivityAnalyzer ----------
class SensitivityAnalyzer:
    """
    Monte Carlo sensitivity analysis over criteria weights.
    - samples Dirichlet-distributed weight vectors around base weights
    - runs TOPSIS many times and computes rank stability metrics (e.g., frequency of top choice)
    """

    def __init__(self, topsis: TOPSISDecision, base_weights: np.ndarray):
        self.topsis = topsis
        self.base_weights = np.array(base_weights, dtype=float)

    def monte_carlo(
        self,
        alternatives: List[str],
        criteria_matrix: np.ndarray,
        n_sim=1000,
        perturbation_scale=0.3,
        seed=42,
    ):
        rng = np.random.RandomState(seed)
        m, n = np.array(criteria_matrix).shape
        # sample around base weights using Dirichlet with concentration proportional to base
        alpha = self.base_weights + 1e-6
        alpha = alpha / alpha.sum()
        alpha = alpha * (1.0 / perturbation_scale) + 1e-3  # small floor
        ranks = np.zeros(
            (n_sim, len(alternatives)), dtype=int
        )  # store rank positions of alternatives by index
        top_counts = {}
        closeness_matrix = np.zeros((n_sim, len(alternatives)))
        for i in range(n_sim):
            w_sample = rng.dirichlet(alpha)
            # run TOPSIS with fixed normalization but replaced weights
            # To use custom weights, we modify TOPSIS: create a quick run
            A = np.array(criteria_matrix, dtype=float)
            A_norm = A / (np.sqrt((A**2).sum(axis=0, keepdims=True)) + 1e-12)
            V = A_norm * w_sample
            ideal_pos = np.zeros(A.shape[1])
            ideal_neg = np.zeros(A.shape[1])
            for j in range(A.shape[1]):
                if self.topsis.benefit_criteria[j]:
                    ideal_pos[j] = V[:, j].max()
                    ideal_neg[j] = V[:, j].min()
                else:
                    ideal_pos[j] = V[:, j].min()
                    ideal_neg[j] = V[:, j].max()
            d_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
            d_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))
            closeness = d_neg / (d_pos + d_neg + 1e-12)
            closeness_matrix[i, :] = closeness
            rank_idx = np.argsort(-closeness)
            ranks[i, :] = rank_idx
            top = alternatives[rank_idx[0]]
            top_counts[top] = top_counts.get(top, 0) + 1
        # stability metric: proportion of simulations where base best remains best
        base_result = self.topsis.run(
            alternatives, criteria_matrix, weight_method="entropy+critic"
        )
        base_best = base_result["alternatives"][np.argmax(base_result["closeness"])]
        stability_index = top_counts.get(base_best, 0) / float(n_sim)
        # compute robustness: how often top-1 remains any of top-3
        # return summary
        # --- DEBUG C: MC DISTRIBUTION ---
        print(f"\n[DEBUG] MC Results:")
        print(f"Top 1 Counts: {top_counts}")
        print(f"Closeness Matrix Sample (First 5): \n{closeness_matrix[:5]}")
        return {
            "stability_index": stability_index,
            "top_counts": top_counts,
            "closeness_matrix": closeness_matrix,
            "ranks": ranks,
            "base_best": base_best,
        }


# ---------- VisualizationEngine ----------
class VisualizationEngine:
    """
    Generate and save the collection of plots requested.
    All plotting functions save high-res PNGs into out_dir/figs/
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        ensure_dir(self.out_dir)
        self.fig_dir = os.path.join(self.out_dir, "figs")
        ensure_dir(self.fig_dir)
        sns.set(style="whitegrid")

    def save_fig(self, fig, name, dpi=200):
        path = os.path.join(self.fig_dir, f"{name}.png")
        print("=== SAVING FIG TO ===", os.path.abspath(path))
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return path

    def heatmap_votes(self, V: np.ndarray, names: List[str], season: int):
        fig, ax = plt.subplots(figsize=(12, max(4, V.shape[0] * 0.25)))
        sns.heatmap(
            V,
            annot=False,
            yticklabels=names,
            xticklabels=[f"W{k+1}" for k in range(V.shape[1])],
            ax=ax,
            cmap="viridis",
        )
        ax.set_title(f"Estimated Fan Votes Heatmap - Season {season}")
        return self.save_fig(fig, f"season_{season}_votes_heatmap")

    def trend_lines(self, V: np.ndarray, names: List[str], season: int, top_n=5):
        # plot trends for top_n contestants by initial week votes
        week1 = V[:, 0]
        idx = np.argsort(-week1)[:top_n]
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in idx:
            ax.plot(np.arange(1, V.shape[1] + 1), V[i, :], marker="o", label=names[i])
        ax.set_xlabel("Week")
        ax.set_ylabel("Estimated Fan Vote %")
        ax.set_title(f"Fan Vote Trends (Top {top_n}) - Season {season}")
        ax.legend(loc="best")
        return self.save_fig(fig, f"season_{season}_vote_trends_top{top_n}")

    def radar_compare(
        self,
        judge_vec: np.ndarray,
        fan_vec: np.ndarray,
        label_j="Judge",
        label_f="Fan",
        name="radar",
    ):
        # both 1D arrays same length
        N = len(judge_vec)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        # close the polygon
        judge = np.concatenate([judge_vec, [judge_vec[0]]])
        fan = np.concatenate([fan_vec, [fan_vec[0]]])
        angles = angles + [angles[0]]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, judge, label=label_j)
        ax.fill(angles, judge, alpha=0.1)
        ax.plot(angles, fan, label=label_f)
        ax.fill(angles, fan, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"V{i+1}" for i in range(N)])
        ax.set_title("Radar Comparison")
        ax.legend()
        return self.save_fig(fig, name)

    def criteria_bar_comparison(
        self,
        criteria_scores_rank: Dict[str, float],
        criteria_scores_percent: Dict[str, float],
        season: int,
    ):
        keys = list(criteria_scores_rank.keys())
        vals_rank = [criteria_scores_rank[k] for k in keys]
        vals_percent = [criteria_scores_percent[k] for k in keys]
        df = pd.DataFrame(
            {"criteria": keys, "rank": vals_rank, "percent": vals_percent}
        )
        df_long = df.melt(id_vars="criteria", var_name="method", value_name="score")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_long, x="criteria", y="score", hue="method", ax=ax)
        ax.set_title(f"Criteria Comparison by Method - Season {season}")
        return self.save_fig(fig, f"season_{season}_criteria_comparison")

    def scatter_matrix_criteria(self, criteria_df: pd.DataFrame, season: int):
        # criteria_df: alternatives x criteria
        pd.plotting.scatter_matrix(criteria_df, figsize=(10, 10))
        fig = plt.gcf()
        fig.suptitle(f"Scatter Matrix of Criteria - Season {season}")
        return self.save_fig(fig, f"season_{season}_criteria_scatter_matrix")

    def parallel_coordinates(
        self, criteria_df: pd.DataFrame, class_column: Optional[str], season: int
    ):
        df = criteria_df.copy()
        if class_column is None:
            df[class_column] = df.index.astype(str)
        fig, ax = plt.subplots(figsize=(12, 6))
        parallel_coordinates(
            df.reset_index().rename(columns={"index": "alt"}),
            class_column if class_column else df.columns[-1],
            ax=ax,
        )
        ax.set_title(f"Parallel Coordinates - Season {season}")
        return self.save_fig(fig, f"season_{season}_parallel_coords")

    def weight_bar(self, weights: np.ndarray, criteria_names: List[str], season: int):
        # Robust weight bar: ensure numeric array and correct length
        w = np.array(weights, dtype=float).flatten()
        # if sizes mismatch, pad/truncate or fallback to equal weights
        if w.size != len(criteria_names) or np.allclose(w.sum(), 0.0):
            # fallback to equal weights
            w = np.ones(len(criteria_names)) / float(len(criteria_names))
        # ensure non-negative
        w = np.clip(w, 0.0, None)
        # normalize for nice display
        if w.sum() > 0:
            w = w / (w.sum() + 1e-12)

        df = pd.DataFrame({"criteria": criteria_names, "weight": w})

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df, x="criteria", y="weight", ax=ax)
        ax.set_title(f"Weights Distribution - Season {season}")
        ax.set_ylim(0, max(0.05, w.max() * 1.2))  # small top margin
        return self.save_fig(fig, f"season_{season}_weights")

    def closeness_radar(self, closeness: np.ndarray, labels: List[str], season: int):
        return self.radar_compare(
            closeness,
            closeness,
            label_j="Closeness",
            label_f=" ",
            name=f"season_{season}_closeness_radar",
        )

    def plot_monte_carlo_results(
        self, sensitivity_result: Dict, alternatives: List[str], season: int
    ):
        top_counts = sensitivity_result.get("top_counts", {})
        # ensure all alternatives are present (maybe some have 0 counts)
        counts = [top_counts.get(a, 0) for a in alternatives]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=alternatives, y=counts, ax=ax)
        ax.set_title(f"Monte Carlo Top-1 Counts - Season {season}")
        ax.set_ylabel("Top-1 Count")
        ax.set_xlabel("Method")
        for i, v in enumerate(counts):
            ax.text(
                i,
                v + max(1, 0.01 * len(sensitivity_result.get("closeness_matrix", []))),
                str(v),
                ha="center",
            )
        return self.save_fig(fig, f"season_{season}_mc_top1_counts")

    def boxplot_closeness(
        self, closeness_matrix: np.ndarray, alternatives: List[str], season: int
    ):
        """
        Robust boxplot using pure matplotlib (avoids seaborn/matplotlib compatibility bugs).
        """
        data = [closeness_matrix[:, i] for i in range(closeness_matrix.shape[1])]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.boxplot(data, labels=alternatives, patch_artist=True, showfliers=True)

        ax.set_title(
            f"Distribution of TOPSIS Closeness Coefficients\n"
            f"under Monte Carlo Weight Perturbations (Season {season})"
        )
        ax.set_ylabel("TOPSIS Closeness Coefficient")
        ax.set_xlabel("Voting Method")

        ax.grid(axis="y", linestyle="--", alpha=0.6)

        return self.save_fig(fig, f"season_{season}_closeness_boxplot")


# ---------- Pipeline ----------
class DWTSPipeline:
    """
    High-level orchestration for per-season analysis and multi-season runs.
    """

    def __init__(
        self,
        csv_path: str,
        out_dir: str = "dwts_outputs",
        config: Optional[Dict] = None,
    ):
        self.dp = DataProcessor(csv_path)
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.config = config or {}
        self.vis = VisualizationEngine(out_dir)
        self.global_summary = []

    def run_for_season(self, season_num: int, save_intermediate=True):
        print(f"Processing season {season_num} ...")
        X, e, names, elimination_weeks = self.dp.prepare_season(season_num, debug=False)
        if X is None or X.size == 0:
            print("No data for this season.")
            return None
        # Estimate fan votes
        estimator = VoteEstimator(
            method=get_season_method(season_num),
            config=self.config.get("estimator", {}),
        )
        estimator.fit(X, e, elimination_weeks)
        V = estimator.estimated_votes
        # simulate both methods
        sim = VotingSimulator(X, V, names)
        seq_rank, weeks_rank, hist_rank = sim.simulate(
            method="rank", judges_choose_bottom_two=False
        )
        seq_percent, weeks_percent, hist_percent = sim.simulate(
            method="percent", judges_choose_bottom_two=False
        )
        # also simulate judges choosing bottom two (variant)
        seq_rank_judges, _, _ = sim.simulate(
            method="rank", judges_choose_bottom_two=True
        )
        seq_percent_judges, _, _ = sim.simulate(
            method="percent", judges_choose_bottom_two=True
        )

        evaluator = MethodEvaluator(names)
        eval_rank_dict, eval_percent_dict = evaluator.evaluate_methods(
            X,
            V,
            V,
            (seq_rank, weeks_rank, hist_rank),
            (seq_percent, weeks_percent, hist_percent),
        )

        # produce criteria arrays
        criteria_names = [
            "historical_consistency",
            "fan_influence",
            "controversy_gra",
            "stability",
            "fairness",
        ]
        criteria_rank = np.array(
            [eval_rank_dict[k] for k in criteria_names], dtype=float
        )
        criteria_percent = np.array(
            [eval_percent_dict[k] for k in criteria_names], dtype=float
        )
        # defensive
        criteria_rank = np.nan_to_num(criteria_rank, nan=0.0, posinf=0.0, neginf=0.0)
        criteria_percent = np.nan_to_num(
            criteria_percent, nan=0.0, posinf=0.0, neginf=0.0
        )

        criteria_mat = np.vstack([criteria_rank, criteria_percent])
        alternatives = ["rank", "percent"]
        # TOPSIS
        benefit_flags = [True] * len(criteria_names)
        topsis = TOPSISDecision(criteria_names, benefit_flags)
        topsis_res = topsis.run(
            alternatives, criteria_mat, weight_method="entropy+critic", alpha=0.5
        )
        # Sensitivity
        sa = SensitivityAnalyzer(topsis, np.array(topsis_res["weights"]))
        sens_res = sa.monte_carlo(
            alternatives, criteria_mat, n_sim=300, perturbation_scale=0.3
        )
        # Visualizations
        season_out = os.path.join(self.out_dir, f"season_{season_num}")
        ensure_dir(season_out)
        # save estimated votes
        np.save(os.path.join(season_out, "estimated_votes.npy"), V)
        # some plots
        self.vis.heatmap_votes(V, names, season_num)
        self.vis.trend_lines(V, names, season_num, top_n=6)

        # criteria bars — use method-specific evaluation dicts
        crit_rank_dict = {k: float(eval_rank_dict.get(k, 0.0)) for k in criteria_names}
        crit_percent_dict = {
            k: float(eval_percent_dict.get(k, 0.0)) for k in criteria_names
        }
        self.vis.criteria_bar_comparison(crit_rank_dict, crit_percent_dict, season_num)

        # weights bar
        self.vis.weight_bar(np.array(topsis_res["weights"]), criteria_names, season_num)

        # Monte Carlo plots
        self.vis.plot_monte_carlo_results(sens_res, alternatives, season_num)
        self.vis.boxplot_closeness(
            sens_res["closeness_matrix"], alternatives, season_num
        )

        # save CSVs
        df_crit = pd.DataFrame(
            [criteria_rank, criteria_percent],
            index=alternatives,
            columns=criteria_names,
        )
        df_crit.to_csv(os.path.join(season_out, "criteria_values.csv"))
        # --- DEBUG A: CRITERIA COMPARISON ---
        print(f"\n[DEBUG] Season {season_num} Criteria Comparison:")
        print(f"Rank Row: {criteria_rank}")
        print(f"Percent Row: {criteria_percent}")
        is_identical = np.allclose(criteria_rank, criteria_percent)
        print(f"Are methods mathematically identical in criteria? {is_identical}")
        if is_identical:
            print("WARNING: Criteria logic is likely ignoring the method difference!")
        with open(
            os.path.join(season_out, "topsis_weights.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"weights": topsis_res["weights"], "criteria": criteria_names},
                f,
                indent=2,
            )

        df_topsis = pd.DataFrame(
            {
                "alternative": topsis_res["alternatives"],
                "closeness": topsis_res["closeness"],
                "rank_order": np.argsort(-np.array(topsis_res["closeness"])) + 1,
            }
        )
        df_topsis.to_csv(os.path.join(season_out, "topsis_results.csv"), index=False)

        # generate a simple HTML report
        html_lines = []
        html_lines.append(f"<h1>DWTS Season {season_num} Analysis</h1>")
        html_lines.append("<h2>Summary</h2>")
        html_lines.append(
            f"<p>Estimated votes saved. TOPSIS recommended: <b>{topsis_res['alternatives'][int(np.argmax(topsis_res['closeness']))]}</b></p>"
        )
        # embed figures from fig dir
        fig_paths = sorted(
            [
                os.path.join(self.vis.fig_dir, f)
                for f in os.listdir(self.vis.fig_dir)
                if f"season_{season_num}_" in f
            ]
        )
        for p in fig_paths:
            html_lines.append(
                f"<div><img src='../{os.path.relpath(p, season_out)}' style='max-width:800px'></div>"
            )
        report_path = os.path.join(season_out, f"report_season_{season_num}.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))

        print(f"Season {season_num} outputs saved to {season_out}")

        # summary: include both eval dicts
        summary = {
            "season": season_num,
            "topsis": topsis_res,
            "eval": {"rank": eval_rank_dict, "percent": eval_percent_dict},
            "sensitivity": {
                "stability_index": sens_res["stability_index"],
                "base_best": sens_res["base_best"],
            },
        }
        self.global_summary.append(summary)
        return summary

    def run_all_seasons(self, limit: Optional[int] = None):
        seasons = self.dp.list_seasons()
        if limit:
            seasons = seasons[:limit]
        all_summaries = []
        for s in seasons:
            try:
                res = self.run_for_season(s)
                if res:
                    all_summaries.append(res)
            except Exception as e:
                print(f"Error processing season {s}: {e}")
        # save global summary
        with open(
            os.path.join(self.out_dir, "global_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.global_summary, f, indent=2)
        return all_summaries


# ---------- Helper: Determine method by season (mirror of your existing logic) ----------
def get_season_method(season_num: int) -> str:
    if season_num in [1, 2]:
        return "rank"
    elif 3 <= season_num <= 27:
        return "percent"
    elif 28 <= season_num <= 34:
        return "rank"
    else:
        return "rank"


# ---------- If run as script: small demo ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DWTS analysis pipeline demo")
    parser.add_argument("--csv", type=str, default="2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--out", type=str, default="dwts_outputs")
    parser.add_argument("--season", type=int, default=2)
    args = parser.parse_args()
    pipeline = DWTSPipeline(args.csv, out_dir=args.out)
    pipeline.run_for_season(args.season)
    # pipeline.run_all_seasons()
