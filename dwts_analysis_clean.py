"""
DWTS Analysis toolkit
- Implements DataProcessor, VoteEstimator (wrapper), VotingSimulator,
  MethodEvaluator, TOPSISDecision, SensitivityAnalyzer, VisualizationEngine
- Pipeline class DWTSPipeline to orchestrate tasks per season
"""

import os
import json
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

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
# ---------- TOPSISDecision ----------
class TOPSISDecision:
    """
    重构后的TOPSIS：比较不同投票方法的性能指标
    输入不再是原始特征，而是从Monte Carlo分析中提取的方法级性能指标
    """

    def __init__(self, benefit_criteria: List[bool] = None):
        """
        benefit_criteria: 每个指标是否为效益型（True: 越大越好，False: 越小越好）
        """
        self.benefit_criteria = benefit_criteria

    @staticmethod
    def compute_method_performance_metrics(
        closeness_matrix: np.ndarray, method_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        从Monte Carlo模拟的closeness矩阵中提取方法级性能指标
        closeness_matrix: shape (n_sim, n_methods) - Monte Carlo模拟中每个方法的closeness系数
        method_names: 方法名称列表

        返回: 字典 {method_name: {metric_name: metric_value}}
        """
        n_sim, n_methods = closeness_matrix.shape

        if n_methods != len(method_names):
            raise ValueError(
                f"closeness_matrix列数({n_methods})与方法名称数({len(method_names)})不匹配"
            )

        method_metrics = {}

        for method_idx, method_name in enumerate(method_names):
            method_closeness = closeness_matrix[:, method_idx]

            # 1. 平均closeness - 效益型
            mean_closeness = np.mean(method_closeness)

            # 2. closeness的标准差 - 成本型（越小越稳定）
            std_closeness = np.std(method_closeness)

            # 3. 成为Top-1的频率 - 效益型
            # 计算每个模拟中哪个方法closeness最高
            top_1_counts = 0
            for sim_idx in range(n_sim):
                if method_closeness[sim_idx] == np.max(closeness_matrix[sim_idx, :]):
                    top_1_counts += 1
            top_1_freq = top_1_counts / n_sim

            # 4. 最差情形closeness（CVaR 5%） - 效益型
            # CVaR: Conditional Value at Risk
            sorted_closeness = np.sort(method_closeness)
            alpha = 0.05  # 5%分位数
            var_idx = int(alpha * n_sim)
            cvar = (
                np.mean(sorted_closeness[:var_idx])
                if var_idx > 0
                else np.min(method_closeness)
            )

            # 5. 与其他方法的分歧度 - 效益型
            # 计算该方法与其他方法closeness的平均绝对差异
            divergence = 0.0
            for other_idx in range(n_methods):
                if other_idx != method_idx:
                    other_closeness = closeness_matrix[:, other_idx]
                    divergence += np.mean(np.abs(method_closeness - other_closeness))
            divergence /= (n_methods - 1) if n_methods > 1 else 1.0

            method_metrics[method_name] = {
                "mean_closeness": float(mean_closeness),
                "std_closeness": float(std_closeness),
                "top1_frequency": float(top_1_freq),
                "worst_case_cvar": float(cvar),
                "divergence_score": float(divergence),
            }

        return method_metrics

    def run_method_comparison(
        self,
        closeness_matrix: np.ndarray,
        method_names: List[str],
        weight_method: str = "entropy+critic",
        custom_weights: Optional[List[float]] = None,
    ) -> Dict:
        """
        主方法：比较不同投票方法的性能

        参数:
            closeness_matrix: Monte Carlo模拟得到的closeness矩阵，shape (n_sim, n_methods)
            method_names: 方法名称列表
            weight_method: 权重计算方法 ("entropy", "critic", "entropy+critic", "equal")
            custom_weights: 自定义权重（如果提供，则忽略weight_method）

        返回:
            包含比较结果的字典
        """
        # 1. 计算每个方法的性能指标
        method_metrics = self.compute_method_performance_metrics(
            closeness_matrix, method_names
        )

        # 2. 构建性能矩阵 (n_methods × n_metrics)
        metric_names = [
            "mean_closeness",
            "std_closeness",
            "top1_frequency",
            "worst_case_cvar",
            "divergence_score",
        ]

        performance_matrix = []
        for method in method_names:
            row = [method_metrics[method][metric] for metric in metric_names]
            performance_matrix.append(row)

        performance_matrix = np.array(performance_matrix)
        n_methods, n_metrics = performance_matrix.shape

        # 3. 确定指标类型
        # mean_closeness: 效益型, std_closeness: 成本型, top1_frequency: 效益型,
        # worst_case_cvar: 效益型, divergence_score: 效益型
        if self.benefit_criteria is None:
            self.benefit_criteria = [True, False, True, True, True]  # 默认设置

        # 4. 标准化性能矩阵
        # 使用向量归一化
        norm_matrix = performance_matrix / (
            np.sqrt((performance_matrix**2).sum(axis=0, keepdims=True)) + 1e-12
        )

        # 5. 计算权重
        if custom_weights is not None:
            weights = np.array(custom_weights, dtype=float)
            if len(weights) != n_metrics:
                raise ValueError(
                    f"自定义权重数量({len(weights)})与指标数量({n_metrics})不匹配"
                )
        else:
            if weight_method == "entropy":
                weights = self.entropy_weights(performance_matrix)
            elif weight_method == "critic":
                weights = self.critic_weights(performance_matrix)
            elif weight_method == "entropy+critic":
                w_e = self.entropy_weights(performance_matrix)
                w_c = self.critic_weights(performance_matrix)
                weights = self.combine_weights(w_e, w_c, alpha=0.5)
            elif weight_method == "equal":
                weights = np.ones(n_metrics) / n_metrics
            else:
                raise ValueError(f"不支持的权重计算方法: {weight_method}")

        # 6. 加权标准化矩阵
        weighted_matrix = norm_matrix * weights

        # 7. 确定理想解和负理想解
        ideal_best = np.zeros(n_metrics)
        ideal_worst = np.zeros(n_metrics)

        for j in range(n_metrics):
            if self.benefit_criteria[j]:
                ideal_best[j] = weighted_matrix[:, j].max()
                ideal_worst[j] = weighted_matrix[:, j].min()
            else:
                ideal_best[j] = weighted_matrix[:, j].min()
                ideal_worst[j] = weighted_matrix[:, j].max()

        # 8. 计算距离
        dist_to_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
        dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

        # 9. 计算相对接近度
        closeness = dist_to_worst / (dist_to_best + dist_to_worst + 1e-12)

        # 10. 排序
        rank_idx = np.argsort(-closeness)  # 降序排列

        # 11. 返回结果
        return {
            "method_names": method_names,
            "metric_names": metric_names,
            "performance_matrix": performance_matrix.tolist(),
            "weights": weights.tolist(),
            "closeness_coefficients": closeness.tolist(),
            "rank_order": rank_idx.tolist(),
            "recommended_method": method_names[rank_idx[0]],
            "method_metrics": method_metrics,
            "dist_to_best": dist_to_best.tolist(),
            "dist_to_worst": dist_to_worst.tolist(),
        }

    # 保留原有的权重计算方法
    @staticmethod
    def entropy_weights(data: np.ndarray) -> np.ndarray:
        """计算熵权法权重"""
        X = np.array(data, dtype=float)
        m, n = X.shape
        X = X + 1e-12
        P = X / X.sum(axis=0, keepdims=True)
        k = 1.0 / math.log(m + 1e-12)
        ent = -k * (P * np.log(P)).sum(axis=0)
        d = 1 - ent
        w = d / (np.sum(d) + 1e-12)
        return w

    @staticmethod
    def critic_weights(data: np.ndarray) -> np.ndarray:
        """计算CRITIC法权重"""
        X = np.array(data, dtype=float)
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
        std = X_std.std(axis=0, ddof=0)
        R = np.corrcoef(X_std.T)
        C = std * (np.sum(1 - R, axis=1))
        w = C / (np.sum(C) + 1e-12)
        return w

    @staticmethod
    def combine_weights(w1: np.ndarray, w2: np.ndarray, alpha=0.5) -> np.ndarray:
        """组合权重"""
        w = alpha * w1 + (1 - alpha) * w2
        w = w / (np.sum(w) + 1e-12)
        return w


# ---------- SensitivityAnalyzer ----------
# ---------- SensitivityAnalyzer ----------
class SensitivityAnalyzer:
    """
    Monte Carlo灵敏度分析 - 返回closeness矩阵供TOPSIS使用
    """

    def __init__(self, alternatives: List[str], benefit_criteria: List[bool]):
        self.alternatives = alternatives
        self.benefit_criteria = benefit_criteria

    def monte_carlo_sensitivity(
        self,
        criteria_matrix: np.ndarray,
        n_sim: int = 1000,
        perturbation_scale: float = 0.3,
        seed: int = 42,
    ) -> Dict:
        rng = np.random.RandomState(seed)
        n_methods, n_criteria = criteria_matrix.shape

        # 存储每次模拟的closeness系数
        closeness_matrix = np.zeros((n_sim, n_methods))
        rank_matrix = np.zeros((n_sim, n_methods), dtype=int)
        top_counts = {method: 0 for method in self.alternatives}

        # 检查并处理criteria_matrix中的异常值
        A = np.array(criteria_matrix, dtype=float)

        # 处理NaN和无限值
        A = np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)

        # 确保所有值非负（对于效益型指标）
        for i in range(A.shape[1]):
            col = A[:, i]
            if self.benefit_criteria[i] and np.min(col) <= 0:
                # 对于效益型指标，确保所有值为正
                A[:, i] = col - np.min(col) + 0.01

        for sim_idx in range(n_sim):
            # 生成随机权重向量
            alpha = np.ones(n_criteria) * (1.0 / perturbation_scale)
            alpha = np.maximum(alpha, 1e-6)
            w_sample = rng.dirichlet(alpha)

            # 执行TOPSIS步骤
            # 使用更稳健的向量归一化方法
            norm_sq = (A**2).sum(axis=0, keepdims=True)
            # 避免除以0
            norm_sq[norm_sq == 0] = 1.0
            A_norm = A / np.sqrt(norm_sq)

            # 加权标准化矩阵
            V = A_norm * w_sample

            # 计算理想解
            ideal_best = np.zeros(n_criteria)
            ideal_worst = np.zeros(n_criteria)

            for j in range(n_criteria):
                col_values = V[:, j]
                if len(col_values) == 0:
                    continue

                if self.benefit_criteria[j]:
                    ideal_best[j] = np.max(col_values)
                    ideal_worst[j] = np.min(col_values)
                else:
                    ideal_best[j] = np.min(col_values)
                    ideal_worst[j] = np.max(col_values)

            # 计算距离，避免数值问题
            d_best = np.zeros(n_methods)
            d_worst = np.zeros(n_methods)

            for i in range(n_methods):
                d_best[i] = np.sqrt(np.sum((V[i, :] - ideal_best) ** 2))
                d_worst[i] = np.sqrt(np.sum((V[i, :] - ideal_worst) ** 2))

            # 计算closeness系数，避免除0
            denominator = d_best + d_worst
            denominator[denominator == 0] = 1e-12
            closeness = d_worst / denominator

            closeness_matrix[sim_idx, :] = closeness

            # 记录排名
            rank_order = np.argsort(-closeness)  # 降序
            rank_matrix[sim_idx, :] = rank_order

            # 记录Top-1
            if len(rank_order) > 0:
                top_method = self.alternatives[rank_order[0]]
                top_counts[top_method] += 1

        # 检查closeness_matrix中是否有NaN
        if np.any(np.isnan(closeness_matrix)):
            print("Warning: NaN found in closeness_matrix, replacing with 0.5")
            closeness_matrix = np.nan_to_num(closeness_matrix, nan=0.5)

        # 计算稳定性指标
        stability = {method: count / n_sim for method, count in top_counts.items()}

        return {
            "closeness_matrix": closeness_matrix,
            "rank_matrix": rank_matrix,
            "top_counts": top_counts,
            "stability": stability,
            "alternatives": self.alternatives,
            "n_sim": n_sim,
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

    def plot_method_comparison_summary(self, comparison_result: Dict, season_num: int):
        """绘制方法比较汇总图"""
        method_names = comparison_result["method_names"]
        metric_names = comparison_result["metric_names"]
        weights = comparison_result["weights"]
        closeness = comparison_result["closeness_coefficients"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 权重分布
        axes[0, 0].bar(metric_names, weights, color="skyblue")
        axes[0, 0].set_title("Criteria Weights Distribution")
        axes[0, 0].set_ylabel("Weight")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Closeness系数比较
        bars = axes[0, 1].bar(
            method_names, closeness, color=["skyblue", "lightgreen", "salmon"]
        )
        axes[0, 1].set_title("TOPSIS Closeness Coefficients")
        axes[0, 1].set_ylabel("Closeness Coefficient")

        # 添加数值标签
        for i, v in enumerate(closeness):
            axes[0, 1].text(i, v + 0.01, f"{v:.3f}", ha="center")

        # 3. 性能矩阵热图
        perf_matrix = np.array(comparison_result["performance_matrix"])
        sns.heatmap(
            perf_matrix,
            annot=True,
            fmt=".3f",
            xticklabels=metric_names,
            yticklabels=method_names,
            ax=axes[1, 0],
            cmap="YlOrRd",
        )
        axes[1, 0].set_title("Performance Matrix")

        # 4. 距离图
        dist_best = comparison_result["dist_to_best"]
        dist_worst = comparison_result["dist_to_worst"]

        x = np.arange(len(method_names))
        width = 0.35
        axes[1, 1].bar(
            x - width / 2, dist_best, width, label="Distance to Best", color="skyblue"
        )
        axes[1, 1].bar(
            x + width / 2, dist_worst, width, label="Distance to Worst", color="salmon"
        )
        axes[1, 1].set_xlabel("Method")
        axes[1, 1].set_ylabel("Distance")
        axes[1, 1].set_title("Distances to Ideal Solutions")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(method_names)
        axes[1, 1].legend()

        fig.suptitle(f"Voting Method Comparison - Season {season_num}", fontsize=16)
        plt.tight_layout()

        return self.save_fig(fig, f"season_{season_num}_method_comparison_summary")


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

        # 1. 数据准备和粉丝投票估计
        X, e, names, elimination_weeks = self.dp.prepare_season(season_num, debug=False)
        if X is None or X.size == 0:
            print("No data for this season.")
            return None

        # 2. 估计粉丝投票
        estimator = VoteEstimator(
            method=get_season_method(season_num),
            config=self.config.get("estimator", {}),
        )
        estimator.fit(X, e, elimination_weeks)
        V = estimator.estimated_votes

        # 3. 模拟不同投票方法
        sim = VotingSimulator(X, V, names)

        # 模拟三种方法
        methods_to_simulate = [
            ("rank", False, "rank"),
            ("percent", False, "percent"),
            ("rank", True, "rank_judges_choice"),
        ]

        method_results = {}
        method_names = []

        for method, judges_choice, display_name in methods_to_simulate:
            seq, weeks, hist = sim.simulate(
                method=method, judges_choose_bottom_two=judges_choice
            )
            method_results[display_name] = {
                "sequence": seq,
                "weeks": weeks,
                "history": hist,
            }
            method_names.append(display_name)

        # 4. 构建改进的指标矩阵
        basic_metrics = []
        N = len(names)

        # 使用MethodEvaluator计算更合理的指标
        evaluator = MethodEvaluator(names)

        for method_name in method_names:
            seq = method_results[method_name]["sequence"]

            if method_name == "rank_judges_choice":
                method_type = "rank"
            else:
                method_type = method_name

            # 使用MethodEvaluator计算指标
            eval_result = evaluator.evaluate_single_method(X, V, seq, method_type)

            # 提取和转换指标
            # 1. 历史一致性 (historical_consistency) - 效益型 [0, 1]
            hist_consistency = max(0.0, min(1.0, eval_result["historical_consistency"]))

            # 2. 粉丝影响力 (fan_influence) - 效益型 [0, 1]
            fan_influence = max(0.0, min(1.0, eval_result["fan_influence"]))

            # 3. 稳定性 (stability) - 需要转换，原始值可能很大
            stability_raw = eval_result["stability"]
            # 将稳定性转换为[0,1]范围
            stability_norm = (
                min(1.0, stability_raw / 10.0) if stability_raw > 0 else 0.01
            )

            # 4. 公平性 (fairness) - 效益型 [0, 1]
            fairness = max(0.01, min(1.0, eval_result["fairness"]))

            # 5. 与评委的一致性 (添加新指标)
            # 计算最终排名与评委排名的相关性
            placement = MethodEvaluator._final_placement_from_elim_sequence(N, seq)
            judge_total = X.sum(axis=1)
            judge_placement = np.argsort(-judge_total).argsort() + 1

            try:
                from scipy.stats import spearmanr

                correlation, _ = spearmanr(placement, judge_placement)
                if np.isnan(correlation):
                    correlation = 0.0
                correlation_score = (correlation + 1) / 2  # 转换为[0,1]
            except:
                correlation_score = 0.5

            # 收集指标 - 所有指标都应该是正数且在合理范围内
            basic_metrics.append(
                [
                    max(0.01, hist_consistency),  # 指标1: 历史一致性
                    max(0.01, fan_influence),  # 指标2: 粉丝影响力
                    max(0.01, stability_norm),  # 指标3: 稳定性
                    max(0.01, fairness),  # 指标4: 公平性
                    max(0.01, correlation_score),  # 指标5: 与评委一致性
                ]
            )

        basic_metrics = np.array(basic_metrics)

        # 检查并处理异常值
        if np.any(np.isnan(basic_metrics)):
            print(f"Warning: NaN values in basic_metrics for season {season_num}")
            basic_metrics = np.nan_to_num(basic_metrics, nan=0.5)

        # 确保所有值为正
        basic_metrics = np.maximum(basic_metrics, 0.01)

        print(f"Season {season_num} basic metrics:")
        for i, method in enumerate(method_names):
            print(f"  {method}: {basic_metrics[i]}")

        # 5. 执行Monte Carlo灵敏度分析
        # 所有指标都是效益型（越大越好）
        benefit_flags = [True, True, True, True, True]

        sa = SensitivityAnalyzer(method_names, benefit_flags)
        mc_results = sa.monte_carlo_sensitivity(
            basic_metrics, n_sim=500, perturbation_scale=0.3
        )

        # 检查closeness_matrix
        closeness_matrix = mc_results["closeness_matrix"]
        if np.any(np.isnan(closeness_matrix)) or np.any(closeness_matrix <= 0):
            print(
                f"Warning: Invalid values in closeness_matrix for season {season_num}"
            )
            closeness_matrix = np.clip(
                np.nan_to_num(closeness_matrix, nan=0.5), 0.01, 0.99
            )
            mc_results["closeness_matrix"] = closeness_matrix

        # 6. 使用新的TOPSIS进行方法性能比较
        topsis = TOPSISDecision(benefit_criteria=None)

        comparison_result = topsis.run_method_comparison(
            closeness_matrix=closeness_matrix,
            method_names=method_names,
            weight_method="entropy+critic",
        )

        # 7. 可视化
        season_out = os.path.join(self.out_dir, f"season_{season_num}")
        ensure_dir(season_out)

        # 保存结果
        with open(os.path.join(season_out, "method_comparison.json"), "w") as f:
            json.dump(comparison_result, f, indent=2, default=float)

        # 保存basic_metrics用于调试
        np.save(os.path.join(season_out, "basic_metrics.npy"), basic_metrics)

        # 保存closeness_matrix用于调试
        np.save(os.path.join(season_out, "closeness_matrix.npy"), closeness_matrix)

        # 生成可视化图表
        self._plot_method_performance_radar(comparison_result, season_num)
        self._plot_monte_carlo_stability(mc_results, season_num)

        # 添加TOPSIS结果可视化
        self.vis.plot_method_comparison_summary(comparison_result, season_num)

        # 添加详细的调试输出
        print(f"\nSeason {season_num} Analysis Results:")
        print(f"Method names: {method_names}")
        print(f"Basic metrics shape: {basic_metrics.shape}")
        print(f"Closeness matrix shape: {closeness_matrix.shape}")
        print(f"Closeness coefficients: {comparison_result['closeness_coefficients']}")
        print(f"Recommended method: {comparison_result['recommended_method']}")

        # 显示每个方法的详细性能指标
        print("\nMethod Performance Details:")
        for method in method_names:
            if method in comparison_result["method_metrics"]:
                metrics = comparison_result["method_metrics"][method]
                print(f"\n{method}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")

        # 8. 返回结果
        summary = {
            "season": season_num,
            "method_comparison": comparison_result,
            "monte_carlo_results": {
                "stability": mc_results["stability"],
                "top_counts": mc_results["top_counts"],
            },
            "recommended_method": comparison_result["recommended_method"],
            "method_names": method_names,
            "basic_metrics": basic_metrics.tolist(),
        }

        self.global_summary.append(summary)
        return summary

    def _plot_method_performance_radar(self, comparison_result, season_num):
        """绘制方法性能雷达图"""
        method_names = comparison_result["method_names"]
        metric_names = comparison_result["metric_names"]
        performance_matrix = np.array(comparison_result["performance_matrix"])

        # 归一化性能矩阵用于雷达图
        norm_matrix = (performance_matrix - performance_matrix.min(axis=0)) / (
            performance_matrix.max(axis=0) - performance_matrix.min(axis=0) + 1e-12
        )

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection="polar"))

        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        for i, method in enumerate(method_names):
            values = norm_matrix[i, :].tolist()
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, "o-", linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(f"Method Performance Radar - Season {season_num}", size=15)
        ax.legend(loc="upper right")

        self.vis.save_fig(fig, f"season_{season_num}_method_performance_radar")

    def _plot_monte_carlo_stability(self, mc_results, season_num):
        """绘制Monte Carlo稳定性分析图"""
        closeness_matrix = mc_results["closeness_matrix"]
        method_names = mc_results["alternatives"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Box plot of closeness coefficients
        data = [closeness_matrix[:, i] for i in range(closeness_matrix.shape[1])]
        axes[0].boxplot(data, labels=method_names, patch_artist=True)
        axes[0].set_title(f"Closeness Coefficient Distribution\nSeason {season_num}")
        axes[0].set_ylabel("TOPSIS Closeness Coefficient")
        axes[0].grid(True, alpha=0.3)

        # 2. Top-1 frequency bar chart
        top_counts = mc_results["top_counts"]
        methods = list(top_counts.keys())
        counts = [top_counts[m] for m in methods]

        axes[1].bar(methods, counts, color=["skyblue", "lightgreen", "salmon"])
        axes[1].set_title(
            f"Top-1 Frequency in Monte Carlo Simulations\nSeason {season_num}"
        )
        axes[1].set_ylabel("Frequency")
        axes[1].set_xlabel("Voting Method")

        # 添加数值标签
        for i, v in enumerate(counts):
            axes[1].text(i, v + 5, str(v), ha="center")

        plt.tight_layout()
        self.vis.save_fig(fig, f"season_{season_num}_monte_carlo_stability")

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
