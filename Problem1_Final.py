import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import warnings

warnings.filterwarnings("ignore")


class FinalFanVoteEstimator:
    def __init__(
        self,
        method="rank",
        lambda_constraint=10000.0,  # 大幅增加约束权重
        lambda_smooth=0.1,
        lambda_regularization=0.01,
        epsilon=1e-6,
        verbose=1,
    ):
        self.method = method
        self.lambda_constraint = lambda_constraint
        self.lambda_smooth = lambda_smooth
        self.lambda_regularization = lambda_regularization
        self.epsilon = epsilon
        self.verbose = verbose

        # 将在fit中初始化
        self.N = None
        self.T = None
        self.params = None
        self.X = None
        self.e = None
        self.elimination_weeks = None
        self.active_cache = {}

    def fit(self, X, e, elimination_weeks=None, max_iter=2000):
        """两阶段拟合：全局搜索 + 局部优化"""
        self.X = X.copy()
        self.e = e.copy()

        if elimination_weeks is None:
            self.elimination_weeks = list(range(len(e)))
        else:
            self.elimination_weeks = elimination_weeks.copy()

        self.N, self.T = X.shape

        # 预处理
        self._preprocess_scores()
        self._build_active_cache()

        # 参数数量
        total_params = self.N + 1 + self.T  # α, β, γ

        # 阶段1：全局搜索
        if self.verbose >= 1:
            print("Phase 1: Global search...")

        bounds = [(-5.0, 5.0)] * self.N + [(0.01, 3.0)] + [(-3.0, 3.0)] * self.T

        # 使用差分进化进行全局搜索
        result_de = differential_evolution(
            func=self._hard_constraint_objective,
            bounds=bounds,
            maxiter=100,
            popsize=min(50, 10 * total_params),
            disp=self.verbose >= 2,
            seed=42,
        )

        # 阶段2：局部优化（从全局最优开始）
        if self.verbose >= 1:
            print("Phase 2: Local refinement...")

        result_local = minimize(
            fun=self._hard_constraint_objective,
            x0=result_de.x,
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "ftol": 1e-10,
                "eps": 1e-8,
                "disp": self.verbose >= 2,
            },
        )

        self.params = result_local.x
        self.fan_votes_ = self._compute_fan_votes(self.params)

        if self.verbose >= 1:
            print(f"Optimization completed.")
            print(f"Success: {result_local.success}")
            print(f"Final objective: {result_local.fun:.6f}")

        return self

    def _hard_constraint_objective(self, params):
        """硬约束目标函数：对任何违反零容忍"""
        V = self._compute_fan_votes(params, normalize=False)

        # 1. 硬约束惩罚：使用指数惩罚
        constraint_penalty = 0.0
        constraint_violations = 0

        for week in self.elimination_weeks:
            violation = self._compute_constraint_violation(week, V)
            if violation > self.epsilon:
                # 指数惩罚：对任何违反都非常敏感
                constraint_penalty += np.exp(5.0 * violation) - 1.0
                constraint_violations += 1

        # 2. 平滑项
        smooth_penalty = 0.0
        for t in range(1, self.T):
            smooth_penalty += np.sum(np.abs(V[:, t] - V[:, t - 1]))

        # 3. 参数正则化
        param_penalty = np.sum(params**2)

        # 总目标：约束惩罚权重最大
        total = (
            self.lambda_constraint * constraint_penalty
            + self.lambda_smooth * smooth_penalty
            + self.lambda_regularization * param_penalty
        )

        # 如果违反太多，增加惩罚
        if constraint_violations > 0:
            total *= 1.0 + 0.1 * constraint_violations

        return total

    def _compute_constraint_violation(self, week, V):
        """计算约束违反，现在要求严格满足"""
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

        if week_V.sum() < 1e-10:
            return 0.0

        if self.method == "rank":
            X_rank = self._compute_rank(-week_X)
            V_rank = self._compute_rank(-week_V)
            combined = X_rank + V_rank

            combined_others = np.delete(combined, e_idx_in_active)
            if len(combined_others) > 0:
                max_others = np.max(combined_others)
            else:
                max_others = combined[e_idx_in_active]

            # 严格检查：淘汰者必须严格最大
            violation = max(0.0, max_others - combined[e_idx_in_active] + 0.01)

            # 增加安全边际要求
            if len(combined_others) > 0:
                margin = combined[e_idx_in_active] - max_others
                if margin < 0.5:  # 要求至少0.5的安全边际
                    violation += (0.5 - margin) * 10

        else:  # 'percent'
            X_percent = (
                week_X / week_X.sum()
                if week_X.sum() > 0
                else np.ones_like(week_X) / len(week_X)
            )
            V_percent = week_V / week_V.sum()
            combined = X_percent + V_percent

            combined_others = np.delete(combined, e_idx_in_active)
            if len(combined_others) > 0:
                min_others = np.min(combined_others)
            else:
                min_others = combined[e_idx_in_active]

            violation = max(0.0, combined[e_idx_in_active] - min_others + 0.001)

            # 增加安全边际
            if len(combined_others) > 0:
                margin = min_others - combined[e_idx_in_active]
                if margin < 0.01:  # 要求至少1%的安全边际
                    violation += (0.01 - margin) * 100

        return max(0.0, violation)

    def _preprocess_scores(self):
        """简化预处理"""
        # 填充缺失值
        for i in range(self.N):
            for t in range(1, self.T):
                if self.X[i, t] == 0 and self.X[i, t - 1] > 0:
                    self.X[i, t] = self.X[i, t - 1]

        # 简单标准化：减去均值，除以标准差
        for t in range(self.T):
            week_scores = self.X[:, t]
            if np.std(week_scores) > 1e-10:
                self.X[:, t] = (week_scores - np.mean(week_scores)) / np.std(
                    week_scores
                )

    def _build_active_cache(self):
        for week in range(self.T):
            if week == 0:
                self.active_cache[week] = list(range(self.N))
            else:
                eliminated = []
                for i, elim_week in enumerate(self.elimination_weeks):
                    if elim_week < week:
                        eliminated.append(self.e[i])
                self.active_cache[week] = [
                    i for i in range(self.N) if i not in eliminated
                ]

    def _active_contestants(self, week):
        return self.active_cache.get(week, [])

    def _compute_fan_votes(self, params, normalize=True):
        """计算粉丝投票"""
        alpha = params[: self.N]
        beta = params[self.N]
        gamma = params[self.N + 1 :]

        # 使用sigmoid函数约束范围
        log_V = (
            1.0 / (1.0 + np.exp(-alpha[:, np.newaxis]))
            + beta * self.X
            + gamma[np.newaxis, :]
        )
        V = np.exp(np.clip(log_V, -20, 20))

        if normalize:
            for t in range(self.T):
                col_sum = V[:, t].sum()
                if col_sum > 1e-10:
                    V[:, t] = V[:, t] / col_sum

        return V

    def _compute_rank(self, scores):
        """计算排名"""
        order = np.argsort(scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))

        unique_scores, inverse = np.unique(scores, return_inverse=True)
        for i in range(len(unique_scores)):
            indices = np.where(inverse == i)[0]
            if len(indices) > 1:
                avg_rank = ranks[indices].mean()
                ranks[indices] = avg_rank

        return ranks + 1

    def evaluate(self):
        """评估模型性能"""
        V = self.fan_votes_

        violations = []
        for week in self.elimination_weeks:
            violation = self._compute_constraint_violation(week, V)
            violations.append(violation)

        satisfied = sum(1 for v in violations if v <= self.epsilon)
        total = len(violations)

        return {
            "satisfaction_rate": satisfied / total * 100 if total > 0 else 100,
            "avg_violation": np.mean(violations),
            "max_violation": np.max(violations) if violations else 0,
            "violations": violations,
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


def analyze_with_final_model(season_num):
    """使用最终模型分析赛季"""
    print(f"\n{'='*60}")
    print(f"FINAL MODEL - SEASON {season_num}")
    print("=" * 60)

    # 准备数据（使用之前的数据准备函数）

    X, e, names, elimination_weeks = enhanced_prepare_season_data(season_num, raw_data)

    if len(e) == 0:
        print("No elimination data.")
        return None

    print(f"Contestants: {X.shape[0]}, Weeks: {X.shape[1]}")
    print(f"Eliminations: {len(e)}")

    # 创建和拟合模型
    estimator = FinalFanVoteEstimator(
        method="rank",
        lambda_constraint=10000.0,  # 高权重
        lambda_smooth=0.1,
        lambda_regularization=0.01,
        verbose=1,
    )

    estimator.fit(X, e, elimination_weeks=elimination_weeks.tolist())

    # 评估
    results = estimator.evaluate()

    # 显示投票估计
    V = estimator.fan_votes_

    print(f"\nResults:")
    print(f"  Satisfaction rate: {results['satisfaction_rate']:.1f}%")
    print(f"  Average violation: {results['avg_violation']:.6f}")
    print(f"  Maximum violation: {results['max_violation']:.6f}")

    # 修复：只显示有淘汰的周的违反情况
    print("\nViolation by elimination week:")
    for i, week in enumerate(elimination_weeks):
        violation = results["violations"][i]
        actual_week = week + 1  # 转换为1-based
        print(f"  Week {actual_week}: {violation:.6f}")

    print(f"\nTop 5 estimated fan support (Week 1):")
    week1 = V[:, 0]
    top_idx = np.argsort(-week1)[:5]
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i}. {names[idx]:20s}: {week1[idx]*100:5.1f}%")

    return {"season": season_num, "results": results, "estimator": estimator}


# 快速测试函数
def quick_test():
    """快速测试关键赛季"""
    global raw_data
    raw_data = pd.read_csv("2026_MCM_Problem_C_Data.csv")

    test_seasons = [1, 2, 25, 26, 12]
    all_results = []

    for season in test_seasons:
        print(f"\n{'='*80}")
        print(f"TESTING SEASON {season}")
        print("=" * 80)

        result = analyze_with_final_model(season)
        if result:
            all_results.append(result)

    # 汇总
    if all_results:
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)

        print(f"{'Season':<8} {'Satisfaction':<12} {'Avg Viol':<12} {'Max Viol':<12}")
        print("-" * 44)

        for res in all_results:
            season = res["season"]
            stats = res["results"]
            print(
                f"{season:<8} {stats['satisfaction_rate']:<12.1f}% "
                f"{stats['avg_violation']:<12.6f} {stats['max_violation']:<12.6f}"
            )

        # 计算平均
        avg_satisfaction = np.mean(
            [r["results"]["satisfaction_rate"] for r in all_results]
        )
        avg_avg_viol = np.mean([r["results"]["avg_violation"] for r in all_results])
        avg_max_viol = np.mean([r["results"]["max_violation"] for r in all_results])

        print("-" * 44)
        print(
            f"{'AVERAGE':<8} {avg_satisfaction:<12.1f}% "
            f"{avg_avg_viol:<12.6f} {avg_max_viol:<12.6f}"
        )


def analyze_all_seasons():
    """分析所有34个赛季"""
    global raw_data
    raw_data = pd.read_csv("2026_MCM_Problem_C_Data.csv")

    # 获取所有赛季
    all_seasons = sorted(raw_data["season"].unique())
    print(f"Found {len(all_seasons)} seasons: {all_seasons}")

    all_results = []

    for season in all_seasons:
        print(f"\n{'='*80}")
        print(f"ANALYZING SEASON {season} ({len(all_results)+1}/{len(all_seasons)})")
        print("=" * 80)

        result = analyze_with_final_model(season)
        if result:
            all_results.append(result)

    # 汇总所有结果
    if all_results:
        print(f"\n{'='*80}")
        print("FINAL SUMMARY - ALL 34 SEASONS")
        print("=" * 80)

        print(f"{'Season':<8} {'Satisfaction':<12} {'Avg Viol':<12} {'Max Viol':<12}")
        print("-" * 44)

        for res in all_results:
            season = res["season"]
            stats = res["results"]
            print(
                f"{season:<8} {stats['satisfaction_rate']:<12.1f}% "
                f"{stats['avg_violation']:<12.6f} {stats['max_violation']:<12.6f}"
            )

        # 计算总体统计
        avg_satisfaction = np.mean(
            [r["results"]["satisfaction_rate"] for r in all_results]
        )
        avg_avg_viol = np.mean([r["results"]["avg_violation"] for r in all_results])
        avg_max_viol = np.mean([r["results"]["max_violation"] for r in all_results])

        # 计算中位数
        median_satisfaction = np.median(
            [r["results"]["satisfaction_rate"] for r in all_results]
        )

        print("-" * 44)
        print(
            f"{'AVERAGE':<8} {avg_satisfaction:<12.1f}% "
            f"{avg_avg_viol:<12.6f} {avg_max_viol:<12.6f}"
        )
        print(f"{'MEDIAN':<8} {median_satisfaction:<12.1f}%")

        # 保存结果到文件
        save_results_to_csv(all_results)

    return all_results


def save_results_to_csv(results, filename="fan_vote_results.csv"):
    """保存结果到CSV文件"""
    import csv

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Season", "Satisfaction_Rate", "Avg_Violation", "Max_Violation"]
        )

        for res in results:
            season = res["season"]
            stats = res["results"]
            writer.writerow(
                [
                    season,
                    stats["satisfaction_rate"],
                    stats["avg_violation"],
                    stats["max_violation"],
                ]
            )

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    analyze_all_seasons()
