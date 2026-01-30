import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import itertools
from typing import List, Dict, Tuple, Optional
from Problem1_Final import enhanced_prepare_season_data, FinalFanVoteEstimator


class VotingMethodEvaluator:
    """专业投票方法评估器，整合TOPSIS等高级方法"""

    def __init__(self):
        self.criteria_names = [
            "历史一致性",
            "粉丝影响力",
            "争议缓解度",
            "稳定性",
            "公平性",
        ]
        self.criteria_types = ["benefit", "benefit", "benefit", "benefit", "benefit"]

    def kendall_tau_b(self, actual: List[int], simulated: List[int]) -> float:
        """计算Kendall's Tau-b相关系数"""
        # 1. 长度检查
        if len(actual) < 2 or len(simulated) < 2:
            return 0.0

        # 2. 方差检查：如果全是同一个值，相关性定义为 0
        if np.std(actual) == 0 or np.std(simulated) == 0:
            return 0.0

        try:
            tau, _ = stats.kendalltau(actual, simulated)
            if np.isnan(tau):
                return 0.0  # 强制捕获 NaN
            return tau
        except:
            return 0.0

    def _convert_to_ranks(self, arr: List[int]) -> List[float]:
        """转换为排名，处理并列"""
        unique_values = sorted(set(arr))
        ranks = []

        for val in arr:
            # 找到所有等于该值的索引
            indices = [i for i, x in enumerate(arr) if x == val]
            avg_rank = np.mean([i + 1 for i in indices])
            ranks.append(avg_rank)

        return ranks

    def _count_ties(self, ranks: List[float]) -> float:
        """计算相同排名的调整项"""
        from collections import Counter

        counts = Counter(ranks)

        n1 = 0
        for count in counts.values():
            if count > 1:
                n1 += count * (count - 1) / 2

        return n1

    def spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算斯皮尔曼相关系数"""
        if len(x) < 2:
            return 0.0
        correlation, _ = stats.spearmanr(x, y)
        return correlation if not np.isnan(correlation) else 0.0

    def grey_relational_analysis(self, sequences: List[np.ndarray]) -> np.ndarray:
        """灰色关联分析"""
        n_seq = len(sequences)
        n_points = len(sequences[0])

        # 参考序列（理想序列）
        reference = np.max(sequences, axis=0)

        # 规范化
        normalized = []
        for seq in sequences:
            normalized_seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq) + 1e-10)
            normalized.append(normalized_seq)

        reference_norm = (reference - np.min(reference)) / (
            np.max(reference) - np.min(reference) + 1e-10
        )

        # 计算灰色关联系数
        zeta = 0.5  # 分辨系数
        coefficients = np.zeros(n_seq)

        for i in range(n_seq):
            abs_diff = np.abs(reference_norm - normalized[i])
            min_min = np.min(abs_diff)
            max_max = np.max(abs_diff)

            grey_coeff = (min_min + zeta * max_max) / (abs_diff + zeta * max_max)
            coefficients[i] = np.mean(grey_coeff)

        return coefficients

    def gini_coefficient(self, values: np.ndarray) -> float:
        """计算基尼系数"""
        if len(values) == 0:
            return 0.0

        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)

        gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
        return gini

    def calculate_criteria(self, method_results: Dict) -> Dict[str, float]:
        """计算所有准则值"""
        criteria_values = {}

        # 1. 历史一致性 (Kendall's Tau-b)
        criteria_values["历史一致性"] = self.kendall_tau_b(
            method_results["actual_elimination"], method_results["simulated_order"]
        )

        # 2. 粉丝影响力 (斯皮尔曼相关系数均值)
        fan_influences = []
        for week_data in method_results.get("weekly_details", []):
            if "fan_ranks" in week_data and "combined_ranks" in week_data:
                fan_inf = abs(
                    self.spearman_correlation(
                        week_data["fan_ranks"], week_data["combined_ranks"]
                    )
                )
                fan_influences.append(fan_inf)

        criteria_values["粉丝影响力"] = (
            np.mean(fan_influences) if fan_influences else 0.5
        )

        # 3. 争议缓解度 (灰色关联分析)
        controversial_seqs = []
        # 这里需要实际数据填充
        criteria_values["争议缓解度"] = 0.7  # 示例值

        # 4. 稳定性 (变异系数的倒数)
        weekly_scores = []
        for week_data in method_results.get("weekly_details", []):
            if "combined_scores" in week_data:
                weekly_scores.append(np.std(week_data["combined_scores"]))

        if weekly_scores:
            cv = np.std(weekly_scores) / np.mean(weekly_scores)
            criteria_values["稳定性"] = 1 / (cv + 1e-10)
        else:
            criteria_values["稳定性"] = 1.0

        # 5. 公平性 (1 - 基尼系数)
        judge_fan_gaps = []
        for week_data in method_results.get("weekly_details", []):
            if "judge_ranks" in week_data and "fan_ranks" in week_data:
                gap = np.abs(
                    np.array(week_data["judge_ranks"])
                    - np.array(week_data["fan_ranks"])
                )
                judge_fan_gaps.extend(gap)

        if judge_fan_gaps:
            gini = self.gini_coefficient(np.array(judge_fan_gaps))
            criteria_values["公平性"] = 1 - gini
        else:
            criteria_values["公平性"] = 0.8

        return criteria_values


class TOPSISDecisionModel:
    """TOPSIS决策模型"""

    def __init__(self):
        self.decision_matrix = None
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.weights = None
        self.ideal_solution = None
        self.negative_ideal = None

    def entropy_weight(self, decision_matrix: np.ndarray) -> np.ndarray:
        """熵权法确定权重"""
        m, n = decision_matrix.shape

        # 规范化
        p = decision_matrix / np.sum(decision_matrix, axis=0, keepdims=True)

        # 计算信息熵
        epsilon = 1e-10
        e = -np.sum(p * np.log(p + epsilon), axis=0) / np.log(m)

        # 计算差异系数
        d = 1 - e

        # 计算权重
        weights = d / np.sum(d)

        return weights

    def critic_weight(self, decision_matrix: np.ndarray) -> np.ndarray:
        """CRITIC法确定权重"""
        m, n = decision_matrix.shape

        # 标准差
        std_dev = np.std(decision_matrix, axis=0)

        # 相关系数矩阵
        corr_matrix = np.corrcoef(decision_matrix.T)

        # 冲突量
        conflict = np.sum(1 - corr_matrix, axis=1)

        # 信息量
        information = std_dev * conflict

        # 权重
        weights = information / np.sum(information)

        return weights

    def combined_weight(self, decision_matrix: np.ndarray) -> np.ndarray:
        """组合赋权：熵权法 + CRITIC法"""
        w_entropy = self.entropy_weight(decision_matrix)
        w_critic = self.critic_weight(decision_matrix)

        # 使用简单平均组合
        combined = (w_entropy + w_critic) / 2
        return combined

    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """向量规范化"""
        norms = np.sqrt(np.sum(matrix**2, axis=0))
        return matrix / norms

    def apply_weights(self, matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """应用权重"""
        return matrix * weights

    def determine_ideal_solutions(
        self, matrix: np.ndarray, criteria_types: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """确定理想解和负理想解"""
        n_criteria = len(criteria_types)

        ideal = np.zeros(n_criteria)
        negative = np.zeros(n_criteria)

        for j in range(n_criteria):
            if criteria_types[j] == "benefit":
                ideal[j] = np.max(matrix[:, j])
                negative[j] = np.min(matrix[:, j])
            else:  # cost
                ideal[j] = np.min(matrix[:, j])
                negative[j] = np.max(matrix[:, j])

        return ideal, negative

    def calculate_distances(
        self, matrix: np.ndarray, ideal: np.ndarray, negative: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算到理想解和负理想解的距离"""
        n_alternatives = matrix.shape[0]

        d_plus = np.zeros(n_alternatives)
        d_minus = np.zeros(n_alternatives)

        for i in range(n_alternatives):
            d_plus[i] = np.sqrt(np.sum((matrix[i, :] - ideal) ** 2))
            d_minus[i] = np.sqrt(np.sum((matrix[i, :] - negative) ** 2))

        return d_plus, d_minus

    def calculate_closeness(
        self, d_plus: np.ndarray, d_minus: np.ndarray
    ) -> np.ndarray:
        """计算相对贴近度"""
        return d_minus / (d_plus + d_minus)

    def topsis_evaluate(
        self,
        decision_matrix: np.ndarray,
        criteria_types: List[str],
        weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """执行完整的TOPSIS评估"""
        # 规范化
        normalized = self.normalize_matrix(decision_matrix)

        # 确定权重
        if weights is None:
            weights = self.combined_weight(decision_matrix)

        # 加权
        weighted = self.apply_weights(normalized, weights)

        # 确定理想解
        ideal, negative = self.determine_ideal_solutions(weighted, criteria_types)

        # 计算距离
        d_plus, d_minus = self.calculate_distances(weighted, ideal, negative)

        # 计算贴近度
        closeness = self.calculate_closeness(d_plus, d_minus)

        # 排名
        ranking = np.argsort(-closeness)  # 降序

        return {
            "weights": weights,
            "normalized_matrix": normalized,
            "weighted_matrix": weighted,
            "ideal_solution": ideal,
            "negative_ideal": negative,
            "d_plus": d_plus,
            "d_minus": d_minus,
            "closeness": closeness,
            "ranking": ranking,
            "best_alternative": ranking[0],
        }


class SensitivityAnalyzer:
    """敏感性分析"""

    def __init__(self):
        pass

    def monte_carlo_sensitivity(
        self,
        decision_matrix: np.ndarray,
        criteria_types: List[str],
        n_simulations: int = 1000,
        noise_level: float = 0.2,
    ) -> Dict:
        """蒙特卡洛敏感性分析"""
        n_alternatives, n_criteria = decision_matrix.shape

        results = []
        best_counts = np.zeros(n_alternatives)

        for _ in range(n_simulations):
            # 添加随机噪声到权重
            base_weights = np.ones(n_criteria) / n_criteria
            noise = np.random.uniform(-noise_level, noise_level, n_criteria)
            perturbed_weights = base_weights + noise
            perturbed_weights = np.clip(perturbed_weights, 0.01, 0.99)
            perturbed_weights = perturbed_weights / np.sum(perturbed_weights)

            # 执行TOPSIS
            topsis = TOPSISDecisionModel()
            result = topsis.topsis_evaluate(
                decision_matrix, criteria_types, perturbed_weights
            )

            results.append(result)
            best_counts[result["best_alternative"]] += 1

        # 计算稳定性指数
        stability_index = np.max(best_counts) / n_simulations

        return {
            "best_counts": best_counts,
            "stability_index": stability_index,
            "probability_distribution": best_counts / n_simulations,
            "all_results": results,
        }

    def criterion_sensitivity(
        self,
        decision_matrix: np.ndarray,
        criteria_types: List[str],
        base_weights: np.ndarray,
        variation_range: float = 0.3,
    ) -> Dict:
        """准则敏感性分析"""
        n_criteria = len(criteria_types)
        sensitivity_scores = np.zeros(n_criteria)

        for j in range(n_criteria):
            # 对当前准则权重进行扰动
            perturbed_results = []

            for delta in np.linspace(-variation_range, variation_range, 11):
                if delta == 0:
                    continue

                weights = base_weights.copy()
                weights[j] += delta

                # 重新归一化
                weights = np.clip(weights, 0.01, 0.99)
                weights = weights / np.sum(weights)

                # 执行TOPSIS
                topsis = TOPSISDecisionModel()
                result = topsis.topsis_evaluate(
                    decision_matrix, criteria_types, weights
                )

                perturbed_results.append(
                    {
                        "delta": delta,
                        "best_alternative": result["best_alternative"],
                        "closeness": result["closeness"].copy(),
                    }
                )

            # 计算该准则的敏感性
            original_ranking = None
            changes = 0

            for res in perturbed_results:
                if original_ranking is None:
                    original_ranking = res["best_alternative"]
                elif res["best_alternative"] != original_ranking:
                    changes += 1

            sensitivity_scores[j] = changes / len(perturbed_results)

        return {
            "sensitivity_scores": sensitivity_scores,
            "most_sensitive_criterion": np.argmax(sensitivity_scores),
        }


class FuzzyTOPSIS:
    """模糊TOPSIS扩展"""

    def __init__(self):
        pass

    def triangular_fuzzy_number(
        self, lower: float, middle: float, upper: float
    ) -> Tuple[float, float, float]:
        """创建三角模糊数"""
        return (lower, middle, upper)

    def fuzzy_normalize(self, fuzzy_matrix: np.ndarray) -> np.ndarray:
        """模糊数规范化"""
        # 对于三角模糊数 (l, m, u)
        max_upper = np.max(fuzzy_matrix[:, :, 2], axis=0)

        normalized = np.zeros_like(fuzzy_matrix)
        n_alternatives, n_criteria, _ = fuzzy_matrix.shape

        for i in range(n_alternatives):
            for j in range(n_criteria):
                l, m, u = fuzzy_matrix[i, j, :]
                normalized[i, j, :] = (
                    l / max_upper[j],
                    m / max_upper[j],
                    u / max_upper[j],
                )

        return normalized

    def fuzzy_distance(
        self, a: Tuple[float, float, float], b: Tuple[float, float, float]
    ) -> float:
        """计算两个模糊数之间的距离"""
        l1, m1, u1 = a
        l2, m2, u2 = b

        distance = np.sqrt(((l1 - l2) ** 2 + (m1 - m2) ** 2 + (u1 - u2) ** 2) / 3)
        return distance


class ComprehensiveAnalysisFramework:
    """综合分析框架"""

    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data
        self.evaluator = VotingMethodEvaluator()
        self.topsis_model = TOPSISDecisionModel()
        self.sensitivity_analyzer = SensitivityAnalyzer()

        # 存储结果
        self.season_results = {}
        self.method_performance = {}

    def analyze_season(self, season_num: int) -> Dict:
        """分析单个赛季"""
        # 获取赛季数据

        X, actual_e, names, elimination_weeks = enhanced_prepare_season_data(
            season_num, self.raw_data, debug=False
        )

        # 使用两种方法模拟
        method_results = {}

        for method in ["rank", "percent"]:
            # 估计粉丝投票
            estimator = FinalFanVoteEstimator(method=method, verbose=0)
            estimator.fit(X, actual_e, elimination_weeks=elimination_weeks.tolist())
            fan_votes = estimator.fan_votes_

            # 模拟淘汰
            simulated_e, weekly_details = self._simulate_elimination(
                X, fan_votes, elimination_weeks, method, names
            )

            # 收集每周详情
            weekly_data = []
            for week in elimination_weeks:
                # 获取本周活跃选手
                active_indices = estimator._active_contestants(week)

                if len(active_indices) > 1:
                    week_X = X[active_indices, week]
                    week_V = fan_votes[active_indices, week]

                    # 计算排名
                    judge_ranks = self.evaluator._convert_to_ranks(-week_X)
                    fan_ranks = self.evaluator._convert_to_ranks(-week_V)

                    # 组合得分
                    if method == "rank":
                        combined_ranks = np.array(judge_ranks) + np.array(fan_ranks)
                        combined_scores = combined_ranks
                    else:
                        judge_percent = week_X / np.sum(week_X)
                        fan_percent = week_V / np.sum(week_V)
                        combined_percent = judge_percent + fan_percent
                        combined_scores = combined_percent

                    weekly_data.append(
                        {
                            "judge_ranks": judge_ranks,
                            "fan_ranks": fan_ranks,
                            "combined_scores": combined_scores,
                        }
                    )

            method_results[method] = {
                "actual_elimination": actual_e,
                "simulated_order": simulated_e,
                "weekly_details": weekly_data,
                "method": method,
            }

        self.season_results[season_num] = method_results
        return method_results

    def _simulate_elimination(self, X, fan_votes, elimination_weeks, method, names):
        """模拟淘汰过程"""
        # 简化的模拟函数
        N, T = X.shape
        simulated = []
        active = list(range(N))
        weekly_details = []

        for week in elimination_weeks:
            if len(active) <= 1:
                break

            week_active = active.copy()
            week_X = X[week_active, week]
            week_V = fan_votes[week_active, week]

            if method == "rank":
                # 排名法
                judge_ranks = self.evaluator._convert_to_ranks(-week_X)
                fan_ranks = self.evaluator._convert_to_ranks(-week_V)
                combined = np.array(judge_ranks) + np.array(fan_ranks)
                elim_idx_in_active = np.argmax(combined)
            else:
                # 百分比法
                judge_percent = week_X / np.sum(week_X)
                fan_percent = week_V / np.sum(week_V)
                combined = judge_percent + fan_percent
                elim_idx_in_active = np.argmin(combined)

            elim_idx = week_active[elim_idx_in_active]
            simulated.append(elim_idx)
            active.remove(elim_idx)

        return np.array(simulated), weekly_details

    def calculate_all_criteria(self) -> pd.DataFrame:
        """计算所有准则值"""
        criteria_data = []

        for method in ["rank", "percent"]:
            method_criteria = []

            for season_num, results in self.season_results.items():
                if method in results:
                    criteria = self.evaluator.calculate_criteria(results[method])
                    method_criteria.append(criteria)

            # 计算平均值
            if method_criteria:
                avg_criteria = {}
                for key in method_criteria[0].keys():
                    values = [c[key] for c in method_criteria]
                    avg_criteria[key] = np.mean(values)

                self.method_performance[method] = avg_criteria
                criteria_data.append(avg_criteria)

        # 转换为DataFrame
        df = pd.DataFrame(criteria_data, index=["rank", "percent"])
        return df

    def perform_topsis_analysis(self) -> Dict:
        """执行TOPSIS分析"""
        # 获取准则数据
        criteria_df = self.calculate_all_criteria()

        # 构建决策矩阵
        decision_matrix = criteria_df.values
        print("=" * 10 + "decision matrix" + "=" * 10)
        print(decision_matrix)
        print("=" * 10 + "decision matrix" + "=" * 10)

        # TOPSIS评估
        criteria_types = self.evaluator.criteria_types
        topsis_result = self.topsis_model.topsis_evaluate(
            decision_matrix, criteria_types
        )

        # 敏感性分析
        sensitivity_result = self.sensitivity_analyzer.monte_carlo_sensitivity(
            decision_matrix, criteria_types
        )

        # 准则敏感性分析
        criterion_sensitivity = self.sensitivity_analyzer.criterion_sensitivity(
            decision_matrix, criteria_types, topsis_result["weights"]
        )

        return {
            "topsis_result": topsis_result,
            "sensitivity_analysis": sensitivity_result,
            "criterion_sensitivity": criterion_sensitivity,
            "criteria_data": criteria_df,
        }

    def generate_recommendation_report(self) -> Dict:
        """生成推荐报告"""
        # 执行完整分析
        analysis_results = self.perform_topsis_analysis()
        topsis_result = analysis_results["topsis_result"]
        sensitivity = analysis_results["sensitivity_analysis"]

        # 确定推荐方法
        best_idx = topsis_result["best_alternative"]
        method_names = ["排名法 (Rank)", "百分比法 (Percent)"]
        recommended_method = method_names[best_idx]

        # 计算置信度
        confidence = sensitivity["stability_index"]

        # 生成详细理由
        criteria_df = analysis_results["criteria_data"]
        weights = topsis_result["weights"]

        reasons = []
        for i, criterion in enumerate(self.evaluator.criteria_names):
            rank_score = (
                criteria_df.iloc[0, i] if best_idx == 0 else criteria_df.iloc[1, i]
            )
            percent_score = (
                criteria_df.iloc[1, i] if best_idx == 1 else criteria_df.iloc[0, i]
            )
            diff = (
                rank_score - percent_score
                if best_idx == 0
                else percent_score - rank_score
            )

            if abs(diff) > 0.1:  # 显著差异
                reasons.append(
                    f"{criterion}: {recommended_method}得分高{diff:.2f} (权重:{weights[i]:.2f})"
                )

        # 判断是否推荐评委选择机制
        judge_choice_recommended = sensitivity["stability_index"] < 0.8

        report = {
            "推荐方法": recommended_method,
            "置信度": f"{confidence:.1%}",
            "相对贴近度": {
                "排名法": f"{topsis_result['closeness'][0]:.3f}",
                "百分比法": f"{topsis_result['closeness'][1]:.3f}",
            },
            "权重分布": dict(
                zip(
                    self.evaluator.criteria_names,
                    [f"{w:.3f}" for w in topsis_result["weights"]],
                )
            ),
            "推荐理由": reasons,
            "评委选择机制": "推荐使用" if judge_choice_recommended else "不推荐使用",
            "敏感性分析": {
                "稳定性指数": f"{sensitivity['stability_index']:.3f}",
                "最敏感准则": self.evaluator.criteria_names[
                    analysis_results["criterion_sensitivity"][
                        "most_sensitive_criterion"
                    ]
                ],
            },
        }

        return report


# 主函数
def main():
    """主分析流程"""
    print("=" * 80)
    print("专业投票方法评估系统 - 基于TOPSIS的多准则决策分析")
    print("=" * 80)

    # 加载数据
    raw_data = pd.read_csv("2026_MCM_Problem_C_Data.csv")

    # 创建分析框架
    framework = ComprehensiveAnalysisFramework(raw_data)

    # 分析关键赛季
    key_seasons = [1, 2, 4, 11, 27, 28, 32]
    print(f"\n分析 {len(key_seasons)} 个关键赛季...")

    for season in key_seasons:
        print(f"  赛季 {season}...")
        framework.analyze_season(season)

    # 执行TOPSIS分析
    print("\n执行TOPSIS多准则决策分析...")
    results = framework.perform_topsis_analysis()

    # 生成推荐报告
    print("\n生成最终推荐报告...")
    report = framework.generate_recommendation_report()

    # 输出报告
    print("\n" + "=" * 80)
    print("最终推荐结果")
    print("=" * 80)

    print(f"\n推荐方法: {report['推荐方法']}")
    print(f"置信度: {report['置信度']}")

    print(f"\n相对贴近度:")
    for method, score in report["相对贴近度"].items():
        print(f"  {method}: {score}")

    print(f"\n准则权重分布:")
    for criterion, weight in report["权重分布"].items():
        print(f"  {criterion}: {weight}")

    print(f"\n推荐理由:")
    for reason in report["推荐理由"]:
        print(f"  • {reason}")

    print(f"\n评委选择机制: {report['评委选择机制']}")

    print(f"\n敏感性分析:")
    for key, value in report["敏感性分析"].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

    return framework, report


if __name__ == "__main__":
    framework, report = main()
