import numpy as np
import pandas as pd
from scipy.optimize import minimize


class FanVoteEstimator:
    def __init__(
        self, method="rank", lambda_constraint=1000.0, lambda_smooth=1.0, epsilon=1e-6
    ):
        self.method = method
        self.lambda_constraint = lambda_constraint
        self.lambda_smooth = lambda_smooth
        self.epsilon = epsilon

        # 将在fit中初始化
        self.N = None  # 选手数
        self.T = None  # 周数
        self.params = None  # 参数向量
        self.X = None  # 评委分数
        self.e = None  # 淘汰顺序

    def _compute_fan_votes(self, params):
        """从参数计算粉丝投票矩阵"""
        N, T = self.N, self.T
        alpha = params[:N]
        beta = params[N]
        gamma = params[N + 1 :]

        # V[i,t] = exp(alpha[i] + beta * X[i,t] + gamma[t])
        log_V = alpha[:, np.newaxis] + beta * self.X + gamma[np.newaxis, :]
        V = np.exp(log_V)
        return V

    def _active_contestants(self, week):
        """返回第week周仍在比赛的选手索引列表"""
        if week == 0:  # 第1周（索引0）
            return list(range(self.N))
        else:
            eliminated_before = self.e[:week]
            return [i for i in range(self.N) if i not in eliminated_before]

    def _constraint_violation(self, week, V):
        """计算第week周的约束违反量"""
        active = self._active_contestants(week)
        if len(active) <= 1:
            return 0.0

        week_X = self.X[active, week]
        week_V = V[active, week]
        e_t = self.e[week]
        e_idx = active.index(e_t)  # 淘汰者在活跃列表中的位置

        if self.method == "rank":
            # 计算排名
            X_rank = self._compute_rank(-week_X)  # 降序排名
            V_rank = self._compute_rank(-week_V)
            combined = X_rank + V_rank

            # 淘汰者应有最大综合排名
            # 计算其他选手中最大的综合排名
            combined_others = np.delete(combined, e_idx)
            max_others = (
                np.max(combined_others) if len(combined_others) > 0 else combined[e_idx]
            )

            violation = max_others - combined[e_idx] + self.epsilon

        else:  # 'percent'
            # 计算百分比
            X_percent = week_X / week_X.sum()
            V_percent = week_V / week_V.sum()
            combined = X_percent + V_percent

            # 淘汰者应有最小综合百分比
            combined_others = np.delete(combined, e_idx)
            min_others = (
                np.min(combined_others) if len(combined_others) > 0 else combined[e_idx]
            )

            violation = combined[e_idx] - min_others + self.epsilon

        return violation

    def _compute_rank(self, scores):
        """计算排名，处理并列（分数相同时获得平均排名）"""
        # 方法1: 使用argsort两次
        order = np.argsort(scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))

        # 处理并列：相同分数获得相同排名（平均排名）
        unique_scores, inverse, counts = np.unique(
            scores, return_inverse=True, return_counts=True
        )

        for i in range(len(unique_scores)):
            if counts[i] > 1:
                indices = np.where(inverse == i)[0]
                avg_rank = ranks[indices].mean()
                ranks[indices] = avg_rank

        # 转换为从1开始的排名
        return ranks + 1

    def objective_function(self, params):
        """目标函数"""
        V = self._compute_fan_votes(params)

        # 1. 约束惩罚项
        constraint_penalty = 0.0
        for t in range(self.T):
            violation = self._constraint_violation(t, V)
            if violation > 0:
                constraint_penalty += violation**2

        # 2. 平滑项
        smooth_penalty = np.sum((V[:, 1:] - V[:, :-1]) ** 2)

        # 3. 总目标
        total = (
            self.lambda_constraint * constraint_penalty
            + self.lambda_smooth * smooth_penalty
        )

        return total

    def fit(self, X, e, max_iter=1000, tol=1e-6):
        """拟合模型"""
        self.X = X
        self.e = e
        self.N, self.T = X.shape

        # 初始化参数
        np.random.seed(42)
        initial_params = np.zeros(self.N + 1 + self.T)
        initial_params[: self.N] = np.random.randn(self.N) * 0.1  # alpha
        initial_params[self.N] = 0.0  # beta
        initial_params[self.N + 1 :] = np.random.randn(self.T) * 0.1  # gamma

        # 使用SciPy优化
        result = minimize(
            self.objective_function,
            initial_params,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol, "disp": True},
        )

        self.params = result.x
        self.fan_votes_ = self._compute_fan_votes(self.params)

        return self

    def predict(self, X=None):
        """返回估计的粉丝投票"""
        if X is None:
            X = self.X
        # 注意：这里假设使用训练时的参数，不重新拟合
        return self.fan_votes_


def prepare_season_data(season_num, raw_data):
    """
    为指定赛季准备数据
    返回: X(评委分数矩阵), e(淘汰顺序), contestant_names(选手名称)
    """
    # 筛选指定赛季的数据
    season_data = raw_data[raw_data["season"] == season_num].copy()

    # 重置索引
    season_data = season_data.reset_index(drop=True)

    # 提取选手名称
    contestant_names = season_data["celebrity_name"].tolist()
    N = len(contestant_names)  # 选手数量

    # 提取所有评委分数列 (week1_judge1_score 到 week11_judge4_score)
    score_columns = [col for col in season_data.columns if "judge" in col]

    # 确定最大周数
    weeks = []
    for col in score_columns:
        week_num = int(col.split("_")[0].replace("week", ""))
        if week_num not in weeks:
            weeks.append(week_num)
    T = max(weeks) if weeks else 0

    # 初始化评委分数矩阵
    X = np.zeros((N, T))

    # 填充评委分数矩阵
    for i in range(N):
        for week in range(1, T + 1):
            # 获取本周所有评委分数列
            week_cols = [col for col in score_columns if f"week{week}_" in col]
            week_scores = []

            for col in week_cols:
                score = season_data.at[i, col]
                # 处理N/A和0值
                if pd.isna(score):
                    score = 0
                elif score == "N/A":
                    score = 0
                else:
                    try:
                        score = float(score)
                    except:
                        score = 0
                week_scores.append(score)

            # 计算本周总分 (所有评委分数之和)
            X[i, week - 1] = sum(week_scores)

    # 解析淘汰顺序
    # 首先提取每个选手的淘汰信息
    elimination_info = []
    for i in range(N):
        result = str(season_data.at[i, "results"])
        name = season_data.at[i, "celebrity_name"]

        if "Eliminated Week" in result:
            # 解析淘汰周次
            week_str = result.replace("Eliminated Week", "").strip()
            try:
                elim_week = int(week_str)
            except:
                # 处理特殊格式
                elim_week = T  # 默认设为最后一周
        elif "Withdrew" in result:
            # 退赛情况，找到第一个全0的周次
            for week in range(T):
                if X[i, week] == 0:
                    elim_week = week + 1
                    break
            else:
                elim_week = T
        else:
            # 进入决赛的选手，设为大值
            if (
                "1st" in result
                or "2nd" in result
                or "3rd" in result
                or "4th" in result
                or "5th" in result
            ):
                elim_week = T + 1  # 设为比最大周数大的值
            else:
                elim_week = T

        elimination_info.append(
            {"name": name, "elim_week": elim_week, "index": i, "result": result}
        )

    # 按淘汰周次排序
    elimination_info.sort(key=lambda x: x["elim_week"])

    # 构建淘汰顺序数组e
    # e[t] 表示在第t周被淘汰的选手索引 (t从0开始)
    e = []

    # 确定实际的淘汰周数 (排除进入决赛的选手)
    max_elim_week = max(
        [info["elim_week"] for info in elimination_info if info["elim_week"] <= T]
    )
    actual_T = max_elim_week

    # 构建每周的淘汰者列表
    week_eliminations = {}
    for info in elimination_info:
        week = info["elim_week"]
        if week <= T and week > 0:
            if week not in week_eliminations:
                week_eliminations[week] = []
            week_eliminations[week].append(info["index"])

    # 对于每周有多个淘汰者的情况，按评委分数排序（分数低的先淘汰）
    for week in range(1, actual_T + 1):
        if week in week_eliminations:
            elims = week_eliminations[week]
            # 按本周评委分数排序（升序）
            elims.sort(key=lambda idx: X[idx, week - 1])
            e.extend(elims)

    e = np.array(e[:actual_T])  # 确保淘汰顺序长度不超过实际淘汰周数

    # 调整X矩阵，只保留实际有淘汰的周数
    X = X[:, :actual_T]

    print(f"Season {season_num}: {N} contestants, {actual_T} elimination weeks")
    print(f"Elimination order indices: {e}")
    print(f"Elimination order names: {[contestant_names[idx] for idx in e]}")

    return X, e, contestant_names


def analyze_season(season_num, method="rank"):
    """Analyze a single season."""
    print(f"\n{'='*60}")
    print(f"Analyzing Season {season_num} using {method} method")
    print("=" * 60)

    # 准备数据
    X, e, names = prepare_season_data(season_num, raw_data)

    if len(e) == 0 or X.shape[1] == 0:
        print(f"Season {season_num} has no valid elimination data. Skipping...")
        return None

    # 创建估计器
    estimator = FanVoteEstimator(
        method=method, lambda_constraint=1000.0, lambda_smooth=1.0
    )

    # 拟合模型
    try:
        estimator.fit(X, e)

        # 获取估计的粉丝投票
        V_estimated = estimator.fan_votes_

        # 检查约束满足情况
        print("\nConstraint violations:")
        violations = []
        for t in range(X.shape[1]):
            violation = estimator._constraint_violation(t, V_estimated)
            violations.append(violation)
            print(f"Week {t+1}: constraint violation = {violation:.6f}")

        avg_violation = np.mean(violations)
        max_violation = np.max(violations)
        print(f"Average violation: {avg_violation:.6f}")
        print(f"Maximum violation: {max_violation:.6f}")

        # 返回结果
        result = {
            "season": season_num,
            "X": X,
            "e": e,
            "names": names,
            "V": V_estimated,
            "violations": violations,
            "avg_violation": avg_violation,
            "max_violation": max_violation,
        }

        return result

    except Exception as error:
        print("*********ERROR**********************")
        print(f"Error analyzing season {season_num}: {error}")
        print("*********ERROR**********************")
        return None


### main execution function
if __name__ == "__main__":
    # 加载数据
    data_path = "2026_MCM_Problem_C_Data.csv"
    raw_data = pd.read_csv(data_path)

    # 显示数据基本信息
    print("Data shape:", raw_data.shape)
    print("\nSeasons available:", sorted(raw_data["season"].unique()))
    print("\nColumns:", list(raw_data.columns))

    # 分析一个示例赛季（例如赛季1）
    season_to_analyze = 1
    result = analyze_season(season_to_analyze, method="rank")

    if result:
        # 显示估计的粉丝投票
        print(f"\nEstimated fan votes for Season {season_to_analyze}:")
        for i, name in enumerate(result["names"]):
            votes_str = ", ".join([f"{v:.2f}" for v in result["V"][i]])
            print(f"{name}: {votes_str}")

        # 分析评委分数与粉丝投票的关系
        print("\nCorrelation between judge scores and estimated fan votes:")
        for i, name in enumerate(result["names"]):
            correlation = np.corrcoef(result["X"][i], result["V"][i])[0, 1]
            print(f"{name}: correlation = {correlation:.4f}")

    # 可选：分析多个赛季
    print("\n\nAnalyzing multiple seasons...")
    all_season_results = []

    # 只分析前几个赛季以节省时间
    seasons_to_analyze = [1, 2, 3, 4, 5]

    for season in seasons_to_analyze:
        result = analyze_season(season, method="rank")
        if result:
            all_season_results.append(result)

    # 汇总结果
    if all_season_results:
        print("\n\nSummary across seasons:")
        print("Season | Contestants | Weeks | Avg Violation | Max Violation")
        print("-" * 60)

        for res in all_season_results:
            print(
                f"{res['season']:6d} | {len(res['names']):11d} | {res['X'].shape[1]:5d} | {res['avg_violation']:13.6f} | {res['max_violation']:12.6f}"
            )
