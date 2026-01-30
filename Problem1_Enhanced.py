import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
from scipy.optimize import Bounds
import warnings

warnings.filterwarnings("ignore")


class EnhancedFanVoteEstimator:
    def __init__(
        self,
        method="rank",
        lambda_constraint=1000.0,
        lambda_smooth=1.0,
        lambda_regularization=0.1,
        epsilon=1e-6,
        use_features=False,
        feature_names=None,
        verbose=1,
    ):
        self.method = method
        self.lambda_constraint = lambda_constraint
        self.lambda_smooth = lambda_smooth
        self.lambda_regularization = lambda_regularization
        self.epsilon = epsilon
        self.use_features = use_features
        self.feature_names = feature_names
        self.verbose = verbose

        # 将在fit中初始化
        self.N = None  # 选手数
        self.T = None  # 周数
        self.params = None
        self.X = None  # 评委分数矩阵
        self.e = None  # 淘汰顺序
        self.elimination_weeks = None
        self.F = None  # 特征矩阵

        # 优化历史
        self.objective_history = []
        self.violation_history = []
        self.best_params = None
        self.best_objective = float("inf")

        # 缓存活跃选手列表
        self.active_cache = {}

    def fit(self, X, e, elimination_weeks=None, features=None, max_iter=5000, tol=1e-8):
        """
        改进的拟合方法
        """
        self.X = X.copy()
        self.e = e.copy()

        if elimination_weeks is None:
            self.elimination_weeks = list(range(len(e)))
        else:
            self.elimination_weeks = elimination_weeks.copy()

        self.N, self.T = X.shape

        # 改进的评委分数预处理
        self._preprocess_judge_scores()

        # 存储特征矩阵
        if features is not None and self.use_features:
            self.F = features.copy()
            D = features.shape[1]
        else:
            self.F = None
            D = 0

        # 构建活跃选手缓存
        self._build_active_cache()

        # 参数初始化（改进策略）
        if self.use_features and self.F is not None:
            total_params = D + 1 + self.T  # w, β, γ
        else:
            total_params = self.N + 1 + self.T  # α, β, γ

        initial_params = self._smart_initialization(
            total_params, D if self.use_features else None
        )

        if self.verbose >= 1:
            print(f"Total parameters: {total_params}")
            print(
                f"Initial parameter range: [{initial_params.min():.4f}, {initial_params.max():.4f}]"
            )

        # 设置边界约束
        bounds = self._create_enhanced_bounds(
            total_params, D if self.use_features else None
        )

        # 使用更强大的优化方法
        if self.verbose >= 1:
            print("Starting optimization...")

        result = minimize(
            fun=self._objective_function_with_breakdown,
            x0=initial_params,
            method="SLSQP",  # 改为SLSQP，更适合约束优化
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "ftol": tol,
                "eps": 1e-8,
                "disp": self.verbose >= 2,
            },
            callback=self._enhanced_callback,
        )

        self.params = result.x
        self.fan_votes_ = self._compute_fan_votes(self.params, normalize=True)

        if self.verbose >= 1:
            print(f"\nOptimization completed.")
            print(f"Success: {result.success}")
            print(f"Final objective: {result.fun:.6f}")
            print(f"Iterations: {result.nit}")
            print(f"Message: {result.message}")

            # 输出目标函数分解
            self._print_objective_breakdown(self.params)

        return self

    def _preprocess_judge_scores(self):
        """
        改进的评委分数预处理
        """
        # 1. 处理缺失值：用前一周分数填充
        for i in range(self.N):
            for t in range(1, self.T):
                if self.X[i, t] == 0 and self.X[i, t - 1] > 0:
                    self.X[i, t] = self.X[i, t - 1]

        # 2. 温和标准化：只对非零值进行，保留原始尺度信息
        for t in range(self.T):
            week_scores = self.X[:, t]
            non_zero = week_scores > 0
            if np.sum(non_zero) > 1:
                mean_val = week_scores[non_zero].mean()
                std_val = week_scores[non_zero].std()
                if std_val > 1e-10:
                    # 只标准化非零值，零值保持不变
                    week_scores[non_zero] = (
                        (week_scores[non_zero] - mean_val) / std_val * 0.5
                    )
                self.X[:, t] = week_scores

    def _build_active_cache(self):
        """构建活跃选手缓存"""
        for week in range(self.T):
            self.active_cache[week] = self._active_contestants(week)

    def _smart_initialization(self, total_params, D=None):
        """
        智能参数初始化
        """
        params = np.zeros(total_params)

        if self.use_features and D is not None:
            # 使用特征
            params[:D] = np.random.randn(D) * 0.05  # 小权重
            params[D] = 0.3 + np.random.randn() * 0.05  # β ~ 0.3
            params[D + 1 :] = np.random.randn(self.T) * 0.1
        else:
            # 不使用特征
            # α: 基于初始评委分数初始化
            initial_scores = self.X[:, 0]
            score_mean = initial_scores.mean() if np.any(initial_scores > 0) else 0

            for i in range(self.N):
                if initial_scores[i] > 0:
                    # 初始分数高的选手可能有更高的α
                    params[i] = (
                        np.random.randn() * 0.2 + (initial_scores[i] - score_mean) * 0.1
                    )
                else:
                    params[i] = np.random.randn() * 0.2

            # β: 评委影响系数
            params[self.N] = 0.3 + np.random.randn() * 0.05

            # γ: 时间效应
            params[self.N + 1 :] = np.random.randn(self.T) * 0.1

        return params

    def _create_enhanced_bounds(self, total_params, D=None):
        """
        创建增强的边界约束
        """
        bounds = []

        if self.use_features and D is not None:
            # w: [-1, 1]
            for _ in range(D):
                bounds.append((-1.0, 1.0))
            # β: [0.01, 2.0]
            bounds.append((0.01, 2.0))
            # γ: [-2, 2]
            for _ in range(self.T):
                bounds.append((-2.0, 2.0))
        else:
            # α: [-3, 3]
            for _ in range(self.N):
                bounds.append((-3.0, 3.0))
            # β: [0.01, 2.0]
            bounds.append((0.01, 2.0))
            # γ: [-2, 2]
            for _ in range(self.T):
                bounds.append((-2.0, 2.0))

        return bounds

    def _active_contestants(self, week):
        """
        获取第week周的活跃选手
        """
        if week == 0:
            return list(range(self.N))

        if week in self.active_cache:
            return self.active_cache[week]

        # 计算活跃选手
        eliminated = []
        for i, elim_week in enumerate(self.elimination_weeks):
            if elim_week < week:
                eliminated.append(self.e[i])

        active = [i for i in range(self.N) if i not in eliminated]
        return active

    def _compute_fan_votes(self, params, normalize=False, debug=False):
        """
        计算粉丝投票，添加非线性变换提高灵活性
        """
        if self.use_features and self.F is not None:
            D = self.F.shape[1]
            w = params[:D]
            beta = params[D]
            gamma = params[D + 1 :]

            # α = w^T * F
            alpha = self.F @ w
        else:
            alpha = params[: self.N]
            beta = params[self.N]
            gamma = params[self.N + 1 :]

        # 计算 log(V) = tanh(α_i) + β·X + γ_t
        # 使用tanh增加非线性，防止极端值
        log_V = np.tanh(alpha[:, np.newaxis]) + beta * self.X + gamma[np.newaxis, :]

        # 数值稳定性
        log_V = np.clip(log_V, -20, 20)
        V = np.exp(log_V)

        # 防止零值
        V = np.maximum(V, 1e-8)

        if normalize:
            # 按周归一化，使每周总投票数为1
            for t in range(self.T):
                week_sum = V[:, t].sum()
                if week_sum > 1e-10:
                    V[:, t] = V[:, t] / week_sum

        if debug and self.verbose >= 3:
            print(f"Alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
            print(f"Beta: {beta:.4f}")
            print(f"Gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]")
            print(f"V range: [{V.min():.6e}, {V.max():.6e}]")

        return V

    def _constraint_violation(self, week, V, detailed=False):
        """
        改进的约束违反计算，添加松弛变量
        """
        if week not in self.elimination_weeks:
            return 0.0

        elim_idx = self.elimination_weeks.index(week)
        e_t = self.e[elim_idx]

        active = self._active_contestants(week)
        if e_t not in active:
            return 0.0

        e_idx_in_active = active.index(e_t)
        week_X = self.X[active, week]
        week_V = V[active, week]

        # 检查数据有效性
        if week_V.sum() < 1e-10:
            return 0.0

        if self.method == "rank":
            # 计算排名
            X_rank = self._compute_rank(-week_X)
            V_rank = self._compute_rank(-week_V)
            combined = X_rank + V_rank

            # 淘汰者应有最大综合排名
            combined_others = np.delete(combined, e_idx_in_active)
            if len(combined_others) > 0:
                max_others = np.max(combined_others)
            else:
                max_others = combined[e_idx_in_active]

            # 带松弛的违反计算
            violation = max(0.0, max_others - combined[e_idx_in_active])

            # 添加小惩罚，即使满足约束也鼓励更大的差距
            margin = (
                combined[e_idx_in_active] - max_others
                if len(combined_others) > 0
                else 0
            )
            if margin < 1.0:  # 如果差距太小，添加小惩罚
                violation += max(0.0, 1.0 - margin) * 0.1

        else:  # 'percent'
            # 计算百分比
            X_percent = (
                week_X / week_X.sum()
                if week_X.sum() > 0
                else np.ones_like(week_X) / len(week_X)
            )
            V_percent = week_V / week_V.sum()
            combined = X_percent + V_percent

            # 淘汰者应有最小综合百分比
            combined_others = np.delete(combined, e_idx_in_active)
            if len(combined_others) > 0:
                min_others = np.min(combined_others)
            else:
                min_others = combined[e_idx_in_active]

            violation = max(0.0, combined[e_idx_in_active] - min_others)

            # 添加小惩罚
            margin = (
                min_others - combined[e_idx_in_active]
                if len(combined_others) > 0
                else 0
            )
            if margin < 0.01:  # 如果差距太小
                violation += max(0.0, 0.01 - margin) * 10

        if detailed and violation > 0 and self.verbose >= 2:
            print(f"  Week {week+1}: violation = {violation:.6f}")
            print(
                f"    Eliminated: {e_t}, Combined rank: {combined[e_idx_in_active]:.2f}"
            )

        return violation

    def _constraint_penalty_exponential(self, violations):
        """指数惩罚：对任何违反都很敏感"""
        penalty = 0.0
        for v in violations:
            if v > self.epsilon:
                # 使用指数函数：即使小的违反也会产生显著惩罚
                penalty += np.exp(10.0 * v) - 1.0
        return penalty

    def _compute_rank(self, scores):
        """计算排名"""
        order = np.argsort(scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))

        # 处理并列
        unique_scores, inverse = np.unique(scores, return_inverse=True)
        for i in range(len(unique_scores)):
            indices = np.where(inverse == i)[0]
            if len(indices) > 1:
                avg_rank = ranks[indices].mean()
                ranks[indices] = avg_rank

        return ranks + 1

    def _objective_function_with_breakdown(self, params):
        """
        返回目标函数值，并分解各部分贡献
        """
        V = self._compute_fan_votes(params, normalize=False)

        # 1. 约束惩罚项（平方和）
        constraint_penalty = 0.0
        for week in self.elimination_weeks:
            violation = self._constraint_violation(week, V)
            constraint_penalty += violation**2

        # 2. 平滑项（相邻周变化惩罚）
        smooth_penalty = 0.0
        for t in range(1, self.T):
            smooth_penalty += np.sum((V[:, t] - V[:, t - 1]) ** 2)

        # 3. 参数正则化
        param_penalty = np.sum(params**2)

        # 4. 投票值正则化（防止极端值）
        vote_regularization = np.sum((V - 1.0 / self.N) ** 2) * 0.01

        # 总目标
        total = (
            self.lambda_constraint * constraint_penalty
            + self.lambda_smooth * smooth_penalty
            + self.lambda_regularization * param_penalty
            + vote_regularization
        )

        # 存储分解（用于调试）
        self._last_objective_breakdown = {
            "constraint": constraint_penalty,
            "smooth": smooth_penalty,
            "param": param_penalty,
            "vote": vote_regularization,
            "total": total,
        }

        return total

    def _print_objective_breakdown(self, params):
        """打印目标函数分解"""
        if hasattr(self, "_last_objective_breakdown"):
            breakdown = self._last_objective_breakdown
            print("\nObjective function breakdown:")
            print(f"  Constraint penalty: {breakdown['constraint']:.4f}")
            print(f"  Smooth penalty:     {breakdown['smooth']:.4f}")
            print(f"  Parameter penalty:  {breakdown['param']:.4f}")
            print(f"  Vote regularization:{breakdown['vote']:.4f}")
            print(f"  Total:              {breakdown['total']:.4f}")

    def _enhanced_callback(self, xk):
        """增强的回调函数"""
        current_obj = self._objective_function_with_breakdown(xk)
        self.objective_history.append(current_obj)

        # 计算当前违反
        V = self._compute_fan_votes(xk, normalize=False)
        total_violation = 0.0
        for week in self.elimination_weeks:
            total_violation += self._constraint_violation(week, V)
        self.violation_history.append(total_violation)

        # 记录最佳解
        if current_obj < self.best_objective:
            self.best_objective = current_obj
            self.best_params = xk.copy()

        # 定期打印进度
        if self.verbose >= 2 and len(self.objective_history) % 50 == 0:
            print(
                f"Iter {len(self.objective_history)}: "
                f"Obj={current_obj:.2f}, "
                f"Viol={total_violation:.4f}"
            )

        return False

    def predict(self, X=None):
        """预测粉丝投票"""
        if X is not None:
            # 更新X并重新计算
            self.X = X.copy()
            self._preprocess_judge_scores()

        if self.params is None:
            raise ValueError("Model not fitted yet.")

        return self._compute_fan_votes(self.params, normalize=True)

    def evaluate_constraints(self, V=None, print_report=True):
        """
        评估所有约束，返回详细报告
        """
        if V is None:
            V = self.fan_votes_

        if print_report:
            print("\n" + "=" * 60)
            print("CONSTRAINT EVALUATION REPORT")
            print("=" * 60)

        violations = []
        details = []

        for week_idx, week in enumerate(self.elimination_weeks):
            violation = self._constraint_violation(week, V, detailed=print_report)
            violations.append(violation)

            elim_idx = self.elimination_weeks.index(week)
            e_t = self.e[elim_idx]

            details.append(
                {
                    "week": week + 1,
                    "eliminated": e_t,
                    "violation": violation,
                    "satisfied": violation <= self.epsilon,
                }
            )

        # 统计
        satisfied = sum(1 for v in violations if v <= self.epsilon)
        total = len(violations)
        satisfaction_rate = satisfied / total * 100 if total > 0 else 0

        if print_report:
            print(f"\nSummary:")
            print(f"  Total constraints: {total}")
            print(f"  Satisfied: {satisfied}")
            print(f"  Violated: {total - satisfied}")
            print(f"  Satisfaction rate: {satisfaction_rate:.1f}%")
            print(f"  Average violation: {np.mean(violations):.6f}")
            print(f"  Max violation: {np.max(violations):.6f}")

            # 显示违反最严重的几周
            if total - satisfied > 0:
                print(f"\nTop 5 worst violations:")
                worst_indices = np.argsort(violations)[-5:][::-1]
                for idx in worst_indices:
                    detail = details[idx]
                    status = "✓" if detail["satisfied"] else "✗"
                    print(
                        f"  Week {detail['week']}: {status} violation = {detail['violation']:.6f} "
                        f"(eliminated: {detail['eliminated']})"
                    )

        return {
            "violations": violations,
            "satisfaction_rate": satisfaction_rate,
            "avg_violation": np.mean(violations),
            "max_violation": np.max(violations),
        }


# 改进的数据准备函数
def enhanced_prepare_season_data(season_num, raw_data, debug=False):
    """
    增强的数据准备函数
    """
    season_data = raw_data[raw_data["season"] == season_num].copy()
    season_data = season_data.reset_index(drop=True)

    contestant_names = season_data["celebrity_name"].tolist()
    N = len(contestant_names)

    # 确定最大周数
    max_week = 0
    for col in season_data.columns:
        if "week" in col and "judge" in col:
            try:
                week_num = int(col.split("_")[0].replace("week", ""))
                max_week = max(max_week, week_num)
            except:
                continue

    # 初始化评委分数矩阵
    X = np.zeros((N, max_week))

    # 填充评委分数
    for i in range(N):
        for week in range(1, max_week + 1):
            week_cols = [col for col in season_data.columns if f"week{week}_" in col]
            week_total = 0
            judge_count = 0

            for col in week_cols:
                score = season_data.at[i, col]
                if pd.isna(score) or score == "N/A" or score == 0:
                    continue
                try:
                    score_val = float(score)
                    week_total += score_val
                    judge_count += 1
                except:
                    continue

            if judge_count > 0:
                X[i, week - 1] = week_total / judge_count
            elif week > 1:
                # 用前一周分数填充
                X[i, week - 1] = X[i, week - 2]

    # 解析淘汰信息（改进版本）
    elimination_info = []

    for i in range(N):
        result = str(season_data.at[i, "results"])

        # 处理各种结果格式
        if "1st" in result or "Winner" in result or "Champion" in result:
            # 冠军：最后一周不被淘汰
            elim_week = max_week - 1
        elif "2nd" in result or "3rd" in result:
            # 亚军/季军：决赛周被淘汰
            elim_week = max_week - 1
        elif "Withdrew" in result or "withdrew" in result:
            # 退赛：找到第一个0分周
            elim_week = max_week - 1
            for week in range(1, max_week):
                if X[i, week] == 0 and X[i, week - 1] > 0:
                    elim_week = week - 1
                    break
        elif "Eliminated" in result:
            # 提取淘汰周数
            import re

            week_match = re.search(r"(\d+)", result)
            if week_match:
                elim_week = int(week_match.group(1)) - 1
            else:
                elim_week = max_week - 1
        else:
            # 默认：最后一周
            elim_week = max_week - 1

        # 确保周数有效
        elim_week = max(0, min(elim_week, max_week - 1))

        elimination_info.append(
            {
                "index": i,
                "elim_week": elim_week,
                "name": contestant_names[i],
                "result": result,
            }
        )

    # 排序：按淘汰周次，同周按评委分数
    elimination_info.sort(
        key=lambda x: (x["elim_week"], -X[x["index"], x["elim_week"]])
    )

    # 构建淘汰顺序
    e = []
    elimination_weeks = []

    for info in elimination_info:
        # 跳过冠军
        if "1st" in info["result"] or "Winner" in info["result"]:
            continue

        e.append(info["index"])
        elimination_weeks.append(info["elim_week"])

    e = np.array(e)
    elimination_weeks = np.array(elimination_weeks)

    # 确定实际处理的周数
    if len(elimination_weeks) > 0:
        T = max(elimination_weeks) + 1
    else:
        T = max_week

    X = X[:, :T]

    if debug:
        print(f"\nSeason {season_num}: {N} contestants, {T} weeks")
        print(f"Elimination weeks (1-based): {elimination_weeks + 1}")
        print(f"Elimination order: {[contestant_names[idx] for idx in e]}")

    return X, e, contestant_names, elimination_weeks


# 改进的分析函数
def analyze_season_enhanced(
    season_num,
    method="rank",
    lambda_constraint=100.0,  # 降低约束权重，避免主导
    lambda_smooth=0.5,
    lambda_regularization=0.01,
    debug=False,
    summary_only=False,
):
    """
    增强的赛季分析函数
    """
    if not summary_only:
        print(f"\n{'='*60}")
        print(f"ANALYZING SEASON {season_num}")
        print("=" * 60)

    try:
        # 准备数据
        X, e, names, elimination_weeks = enhanced_prepare_season_data(
            season_num, raw_data, debug=debug
        )

        if len(e) == 0:
            if not summary_only:
                print("No elimination data. Skipping...")
            return None

        if not summary_only:
            print(f"Data: {X.shape[0]} contestants, {X.shape[1]} weeks")
            print(f"Eliminations: {len(e)}")

        # 创建估计器
        estimator = EnhancedFanVoteEstimator(
            method=method,
            lambda_constraint=lambda_constraint,
            lambda_smooth=lambda_smooth,
            lambda_regularization=lambda_regularization,
            verbose=0 if summary_only else (2 if debug else 1),
        )

        # 拟合模型
        estimator.fit(
            X, e, elimination_weeks=elimination_weeks.tolist(), max_iter=1000, tol=1e-6
        )

        # 评估约束
        constraint_report = estimator.evaluate_constraints(
            print_report=not summary_only
        )

        # 获取估计的投票
        V = estimator.fan_votes_

        # 显示结果
        if not summary_only:
            print(f"\n{'='*40}")
            print("ESTIMATION RESULTS")
            print("=" * 40)

            # 第一周结果
            print(f"\nWeek 1 (estimated fan support):")
            week1_votes = V[:, 0]
            sorted_idx = np.argsort(-week1_votes)[:5]
            for rank, idx in enumerate(sorted_idx, 1):
                percentage = week1_votes[idx] * 100
                print(f"  {rank}. {names[idx]:20s}: {percentage:5.1f}%")

            # 最后一周结果
            if X.shape[1] > 1:
                last_week = X.shape[1] - 1
                print(f"\nWeek {last_week+1} (estimated fan support):")
                last_week_votes = V[:, last_week]
                sorted_idx = np.argsort(-last_week_votes)[:5]
                for rank, idx in enumerate(sorted_idx, 1):
                    percentage = last_week_votes[idx] * 100
                    print(f"  {rank}. {names[idx]:20s}: {percentage:5.1f}%")

            # 检查参数合理性
            params = estimator.params
            if not estimator.use_features:
                beta = params[X.shape[0]]
                print(f"\nModel parameters:")
                print(f"  Beta (judge influence): {beta:.4f}")
                print(
                    f"  Alpha range: [{params[:X.shape[0]].min():.3f}, {params[:X.shape[0]].max():.3f}]"
                )

        return {
            "season": season_num,
            "X": X,
            "e": e,
            "names": names,
            "V": V,
            "elimination_weeks": elimination_weeks,
            "estimator": estimator,
            **constraint_report,
        }

    except Exception as e:
        print(f"Error analyzing season {season_num}: {e}")
        import traceback

        traceback.print_exc()
        return None


# 主测试函数
def test_improved_model():
    """
    测试改进的模型
    """
    print("=" * 80)
    print("TESTING IMPROVED FAN VOTE ESTIMATION MODEL (SUMMARY)")
    print("=" * 80)

    # 加载数据
    global raw_data
    raw_data = pd.read_csv("2026_MCM_Problem_C_Data.csv")

    # 测试关键赛季
    test_seasons = [1, 2, 25, 26, 12]  # 包含问题赛季和成功赛季

    results = []

    for season in test_seasons:
        result = analyze_season_enhanced(
            season_num=season,
            method="rank",
            lambda_constraint=50.0,  # 进一步降低
            lambda_smooth=0.2,
            lambda_regularization=0.05,
            debug=False,
            summary_only=True,
        )

        if result:
            results.append(result)

    # 汇总结果
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY OF IMPROVED MODEL")
        print("=" * 80)

        print(
            f"{'Season':<8} {'Satisfaction':<15} {'Avg Violation':<15} {'Max Violation':<15}"
        )
        print("-" * 53)

        for res in results:
            season = res["season"]
            satisfaction = res["satisfaction_rate"]
            avg_viol = res["avg_violation"]
            max_viol = res["max_violation"]

            print(
                f"{season:<8} {satisfaction:<15.1f}% {avg_viol:<15.6f} {max_viol:<15.6f}"
            )

        # 计算平均值
        avg_satisfaction = np.mean([r["satisfaction_rate"] for r in results])
        avg_avg_viol = np.mean([r["avg_violation"] for r in results])
        avg_max_viol = np.mean([r["max_violation"] for r in results])

        print("-" * 53)
        print(
            f"{'AVERAGE':<8} {avg_satisfaction:<15.1f}% {avg_avg_viol:<15.6f} {avg_max_viol:<15.6f}"
        )

        print("\nAvg violation by season:")
        for res in results:
            print(f"  Season {res['season']}: {res['avg_violation']:.6f}")


# 运行测试
if __name__ == "__main__":
    test_improved_model()
