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

        for i in range(len(self.elimination_weeks)):
            violation = self._compute_constraint_violation(i, V)
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

    def _compute_constraint_violation(self, elim_idx, V):
        """计算约束违反，现在要求严格满足"""
        week = self.elimination_weeks[elim_idx]
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

        # 根据方法选择标准化方式
        if self.method == "rank":
            # 简单标准化：减去均值，除以标准差
            for t in range(self.T):
                week_scores = self.X[:, t]
                if np.std(week_scores) > 1e-10:
                    self.X[:, t] = (week_scores - np.mean(week_scores)) / np.std(
                        week_scores
                    )
        else:  # 'percent'
            # 归一化到[0,1]区间，保持非负性，避免Z-score导致求和为0的问题
            for t in range(self.T):
                week_scores = self.X[:, t]
                # 减去最小值，确保非负（如果有负值的话），然后除以范围
                # 或者简单地除以最大值（如果数据本身是非负的）
                # 考虑到评委打分通常非负，且我们希望保持"分数"的含义
                # 使用最大值归一化比较安全
                max_val = np.max(week_scores)
                if max_val > 1e-10:
                    self.X[:, t] = week_scores / max_val
                elif np.max(np.abs(week_scores)) < 1e-10:
                    # 全为0的情况
                    pass

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
        for i in range(len(self.elimination_weeks)):
            violation = self._compute_constraint_violation(i, V)
            violations.append(violation)

        satisfied = sum(1 for v in violations if v <= self.epsilon)
        total = len(violations)

        return {
            "satisfaction_rate": satisfied / total * 100 if total > 0 else 100,
            "avg_violation": np.mean(violations),
            "max_violation": np.max(violations) if violations else 0,
            "violations": violations,
        }

 # ============================================================
    # 请将这两个方法添加到 FinalFanVoteEstimator 类中
    # 不要删除原来的 fit, _compute_fan_votes 等方法
    # ============================================================

    def analyze_consistency(self, contestant_names):
        """[新增] 生成详细的一致性分析报告"""
        V = self.fan_votes_
        report_data = []

        print("\n=== Consistency Analysis Detail ===")
        correct_count = 0

        for idx, week in enumerate(self.elimination_weeks):
            actual_eliminated_idx = self.e[idx]
            actual_name = contestant_names[actual_eliminated_idx]

            # 获取本周数据
            active_indices = self._active_contestants(week)
            week_X = self.X[active_indices, week]
            week_V = V[active_indices, week]

            # 重新计算排名/分数以验证淘汰逻辑
            if self.method == "rank":
                # 注意：你的原逻辑是 -score 进行 rank，分数越高 rank值越小(1st)
                # 综合分 = JudgeRank + FanRank。数值越大表现越差。
                X_rank = self._compute_rank(-week_X)
                V_rank = self._compute_rank(-week_V)
                combined_score = X_rank + V_rank

                # 找出综合分最大(最差)的人
                max_score = np.max(combined_score)
                # 找出所有并列倒数第一的人
                candidates = [i for i, s in enumerate(combined_score) if s == max_score]

                # 平局规则（队友是对的，你的模型倾向于无平局，但分析时我们要兼容平局逻辑）
                if len(candidates) == 1:
                    pred_loser_local_idx = candidates[0]
                else:
                    # 平局时，比较粉丝排名，粉丝Rank数值大(票少)的淘汰
                    worst_fan_rank = -1
                    pred_loser_local_idx = candidates[0]
                    for c_idx in candidates:
                        if V_rank[c_idx] > worst_fan_rank:
                            worst_fan_rank = V_rank[c_idx]
                            pred_loser_local_idx = c_idx
            else:
                # Percent 模式：分数最低的淘汰
                X_pct = week_X / (week_X.sum() + 1e-10)
                V_pct = week_V / (week_V.sum() + 1e-10)
                combined_score = X_pct + V_pct
                pred_loser_local_idx = np.argmin(combined_score)

            pred_loser_global_idx = active_indices[pred_loser_local_idx]
            pred_name = contestant_names[pred_loser_global_idx]

            is_correct = (pred_loser_global_idx == actual_eliminated_idx)
            if is_correct: correct_count += 1

            status = "✅ Correct" if is_correct else f"❌ Predicted: {pred_name}"

            report_data.append({
                "Week": week + 1,
                "Actual Eliminated": actual_name,
                "Model Prediction": pred_name,
                "Result": "Correct" if is_correct else "Incorrect"
            })
            print(f"Week {week + 1}: {actual_name:<20} | {status}")

        accuracy = correct_count / len(self.elimination_weeks) * 100
        print(f"Overall Consistency Accuracy: {accuracy:.1f}%")
        return pd.DataFrame(report_data)


    def analyze_certainty(self, contestant_names):
        """[新增] 计算确定性：可行域探测法"""
        print("\n=== Certainty Analysis (Feasible Range Probe) ===")
        V = self.fan_votes_
        certainty_data = []

        # 只分析有淘汰发生的周
        for idx, week in enumerate(self.elimination_weeks):
            actual_eliminated_idx = self.e[idx]
            active_indices = self._active_contestants(week)

            # 当前模型预测出的“最佳”投票比例
            original_votes = V[active_indices, week]
            # 归一化
            if original_votes.sum() > 0:
                original_votes = original_votes / original_votes.sum()

            for local_i, global_i in enumerate(active_indices):
                name = contestant_names[global_i]
                current_vote = original_votes[local_i]

                # 定义辅助检查函数：如果某人票数变了，淘汰结果会变吗？
                def check_validity(target_vote):
                    # 构造新的投票分布：保持其他人相对比例不变
                    sim_votes = original_votes.copy()
                    others_sum = np.sum(sim_votes) - sim_votes[local_i]
                    remaining = 1.0 - target_vote

                    if others_sum > 1e-9:
                        scale = remaining / others_sum
                        sim_votes *= scale
                        sim_votes[local_i] = target_vote
                    else:
                        sim_votes[:] = 0
                        sim_votes[local_i] = 1.0  # 极端情况

                    # 重新计算排名
                    week_X = self.X[active_indices, week]

                    if self.method == "rank":
                        X_rank = self._compute_rank(-week_X)
                        V_rank = self._compute_rank(-sim_votes)  # 票高Rank小
                        combined = X_rank + V_rank
                        # 找最差（最大）
                        max_val = np.max(combined)
                        losers = [i for i, v in enumerate(combined) if v == max_val]

                        # 平局逻辑
                        final_loser = losers[0]
                        if len(losers) > 1:
                            worst_fr = -1
                            for l_idx in losers:
                                if V_rank[l_idx] > worst_fr:
                                    worst_fr = V_rank[l_idx]
                                    final_loser = l_idx
                    else:
                        # Percent
                        X_p = week_X / (week_X.sum() + 1e-9)
                        comb = X_p + sim_votes
                        final_loser = np.argmin(comb)

                    return active_indices[final_loser] == actual_eliminated_idx

                # 探测下限 (Min)
                min_vote = current_vote
                # 向下探测 20步
                for v in np.linspace(current_vote, 0, 21):
                    if check_validity(v):
                        min_vote = v
                    else:
                        break

                # 探测上限 (Max)
                max_vote = current_vote
                # 向上探测 20步
                for v in np.linspace(current_vote, 1.0, 21):
                    if check_validity(v):
                        max_vote = v
                    else:
                        break

                rng = max_vote - min_vote
                certainty = 1.0 - rng  # 简单定义：范围越小确定性越高

                certainty_data.append({
                    "Week": week + 1,
                    "Contestant": name,
                    "Min Vote": min_vote,
                    "Max Vote": max_vote,
                    "Range": rng,
                    "Certainty": certainty
                })

        df = pd.DataFrame(certainty_data)
        # 打印摘要：平均确定性
        print(f"Average Certainty Score: {df['Certainty'].mean():.4f}")
        return df

def get_season_method(season_num):
    """
    根据赛季号确定使用的计算方法
    """
    if season_num in [1, 2]:
        return "rank"
    elif 3 <= season_num <= 27:
        return "percent"
    elif 28 <= season_num <= 34:
        return "rank"
    else:
        # 默认使用rank
        return "rank"


# def enhanced_prepare_season_data(season_num, raw_data, debug=False):
#     """
#     增强的数据准备函数 - 基于评委N/A情况判断停播
#     """
#     season_data = raw_data[raw_data["season"] == season_num].copy()
#     season_data = season_data.reset_index(drop=True)
#
#     contestant_names = season_data["celebrity_name"].tolist()
#     N = len(contestant_names)
#
#     # 首先，找出数据中涉及的所有周数
#     all_weeks = set()
#     for col in season_data.columns:
#         if "week" in col and "judge" in col:
#             try:
#                 week_str = col.split("_")[0].replace("week", "")
#                 week_num = int(week_str)
#                 all_weeks.add(week_num)
#             except:
#                 continue
#
#     if not all_weeks:
#         if debug:
#             print(f"No week columns found for season {season_num}")
#         return None, None, None, None
#
#     max_theoretical_week = max(all_weeks)
#
#     if debug:
#         print(f"Theoretical max week from columns: {max_theoretical_week}")
#         print(f"All weeks found: {sorted(all_weeks)}")
#
#     # 检测停播周：找到第一个所有评委分数都是N/A的周
#     actual_max_week = max_theoretical_week  # 默认到最大周数
#
#     for week in sorted(all_weeks):
#         # 获取本周的所有评委列
#         judge_cols = [
#             col
#             for col in season_data.columns
#             if f"week{week}_" in col and "judge" in col
#         ]
#
#         if not judge_cols:
#             continue
#
#         # 检查是否有非N/A的评委分数
#         week_has_valid_data = False
#
#         # 随机检查几个选手（不需要检查所有选手）
#         sample_indices = list(range(min(5, N)))  # 检查前5个选手
#
#         for i in sample_indices:
#             for col in judge_cols:
#                 score = season_data.at[i, col]
#                 if pd.isna(score) or score == "N/A" or score == 0:
#                     continue
#                 try:
#                     score_val = float(score)
#                     if score_val > 0:
#                         week_has_valid_data = True
#                         break
#                 except:
#                     continue
#             if week_has_valid_data:
#                 break
#
#         # 如果本周所有评委分数都是N/A，则赛季到此结束
#         if not week_has_valid_data:
#             # 再详细检查确认
#             all_na_confirmed = True
#             for i in range(N):
#                 for col in judge_cols:
#                     score = season_data.at[i, col]
#                     if not pd.isna(score) and score != "N/A" and score != 0:
#                         try:
#                             if float(score) > 0:
#                                 all_na_confirmed = False
#                                 break
#                         except:
#                             all_na_confirmed = False
#                             break
#                 if not all_na_confirmed:
#                     break
#
#             if all_na_confirmed:
#                 actual_max_week = week - 1  # 赛季实际结束于上一周
#                 if debug:
#                     print(f"  Week {week}: All judge scores are NA")
#                     print(f"  Season ends at week {actual_max_week}")
#                 break
#
#     # 确保actual_max_week至少为1
#     actual_max_week = max(1, actual_max_week)
#
#     if debug:
#         print(f"Actual max week (after NA detection): {actual_max_week}")
#
#     # 初始化评委分数矩阵（只到实际最大周数）
#     X = np.zeros((N, actual_max_week))
#
#     # 填充评委分数
#     for i in range(N):
#         for week in range(1, actual_max_week + 1):
#             week_cols = [col for col in season_data.columns if f"week{week}_" in col]
#             week_total = 0
#             judge_count = 0
#
#             for col in week_cols:
#                 score = season_data.at[i, col]
#                 if pd.isna(score) or score == "N/A" or score == 0:
#                     continue
#                 try:
#                     score_val = float(score)
#                     week_total += score_val
#                     judge_count += 1
#                 except:
#                     continue
#
#             if judge_count > 0:
#                 X[i, week - 1] = week_total / judge_count
#             elif week > 1 and X[i, week - 2] > 0:
#                 # 用前一周分数填充
#                 X[i, week - 1] = X[i, week - 2]
#
#     # 解析淘汰信息
#     elimination_info = []
#     finalists = []
#
#     # 找出决赛选手
#     for i in range(N):
#         result = str(season_data.at[i, "results"])
#
#         if "1st" in result or "Winner" in result or "Champion" in result:
#             finalists.append(
#                 {"index": i, "name": contestant_names[i], "place": 1, "result": result}
#             )
#         elif "2nd" in result:
#             finalists.append(
#                 {"index": i, "name": contestant_names[i], "place": 2, "result": result}
#             )
#         elif "3rd" in result:
#             finalists.append(
#                 {"index": i, "name": contestant_names[i], "place": 3, "result": result}
#             )
#         elif "4th" in result or "5th" in result:
#             finalists.append(
#                 {"index": i, "name": contestant_names[i], "place": 4, "result": result}
#             )
#
#     # 确定决赛周：通常是最后一个实际比赛周
#     final_week = actual_max_week - 1
#
#     # 尝试从结果中推断决赛周
#     for f in finalists:
#         result = f["result"]
#         if "Week" in result:
#             import re
#
#             week_match = re.search(r"Week\s*(\d+)", result)
#             if week_match:
#                 week_candidate = int(week_match.group(1))
#                 if week_candidate <= actual_max_week:
#                     final_week = week_candidate - 1
#                     if debug:
#                         print(f"  Final week from {f['name']}: Week {final_week + 1}")
#
#         # === 修正后的淘汰周分配逻辑：结合你的“最后得分周”思路 ===
#     for i in range(N):
#         result = str(season_data.at[i, "results"])
#         is_finalist = any(f["index"] == i for f in finalists)
#         is_champion = False
#
#         # --- 核心改进：根据评委分数矩阵 X 找到最后活跃的周 ---
#         # 你的逻辑：最后一次得分不为 0 的周就是他的退出周
#         last_active_week = -1
#         for week_idx in range(actual_max_week):
#             if X[i, week_idx] > 0:
#                 last_active_week = week_idx
#
#         # 获取冠军标识
#         if is_finalist:
#             for f in finalists:
#                 if f["index"] == i and f["place"] == 1:
#                     is_champion = True
#                     break
#
#         # 优先级逻辑判定：
#         if is_champion:
#             # 冠军默认待到最后一周（虽然之后会被 e 列表跳过，但为了逻辑严密需保留）
#             elim_week = final_week
#         else:
#             # 非冠军选手，无论他是第几名，他“离开”的那一周就是他最后有得分的那一周
#             # 这样 Joey (3rd) 在 S1 W5 有分，W6 没分，last_active_week 就会准确指向 W5
#             elim_week = last_active_week
#
#         # 容错处理：如果因为数据缺失导致没找到活跃周，再尝试从文本提取（保留队友的正则作为兜底）
#         if elim_week == -1:
#             import re
#             week_match = re.search(r"Week\s*(\d+)", result)
#             if week_match:
#                 elim_week = int(week_match.group(1)) - 1
#             else:
#                 elim_week = final_week
#
#         # 最终确保周数在有效范围内
#         elim_week = max(0, min(elim_week, actual_max_week - 1))
#
#         elimination_info.append({
#             "index": i,
#             "elim_week": elim_week,  # 这里现在是基于真实得分数据了！
#             "name": contestant_names[i],
#             "result": result,
#             "is_champion": is_champion,
#             "is_finalist": is_finalist,
#         })
#
#     # 排序：按淘汰周次，同周按评委分数
#     elimination_info.sort(key=lambda x: (x["elim_week"], X[x["index"], x["elim_week"]]))
#
#     # 构建淘汰顺序 - 冠军不加入淘汰列表
#     e = []
#     elimination_weeks = []
#
#     for info in elimination_info:
#         # 冠军不加入淘汰列表
#         if info.get("is_champion", False):
#             if debug:
#                 print(f"  Skipping champion: {info['name']}")
#             continue
#
#         e.append(info["index"])
#         elimination_weeks.append(info["elim_week"])
#
#     e = np.array(e)
#     elimination_weeks = np.array(elimination_weeks)
#
#     # 确定模型需要处理的周数
#     if len(elimination_weeks) > 0:
#         T = max(elimination_weeks) + 1
#     else:
#         T = actual_max_week
#
#     # 如果T小于实际周数，截断X
#     if T < actual_max_week:
#         X = X[:, :T]
#
#     if debug:
#         print(f"\nFinal Processing Summary for Season {season_num}:")
#         print(f"  Contestants: {N}")
#         print(f"  Actual max week: {actual_max_week}")
#         print(f"  Model weeks: {T}")
#         print(f"  Final week: {final_week + 1}")
#         print(f"  Eliminations: {len(e)}")
#
#         if len(e) > 0:
#             print(f"  Elimination weeks (1-based): {elimination_weeks + 1}")
#             print(f"  Elimination order: {[contestant_names[idx] for idx in e]}")
#
#         # 显示淘汰详情
#         print(f"\nElimination details:")
#         for info in elimination_info:
#             if info.get("is_champion", False):
#                 status = "Champion"
#             elif info.get("is_finalist", False):
#                 status = "Finalist"
#             else:
#                 status = "Eliminated"
#
#             if status != "Champion":  # 冠军已跳过
#                 print(f"  {info['name']:20s}: {status} in Week {info['elim_week'] + 1}")
#
#     return X, e, contestant_names, elimination_weeks
#
def enhanced_prepare_season_data(season_num, raw_data, debug=False):
    """
    修正版数据准备：
    1. 修复了向前填充导致无法识别淘汰时间的问题。
    2. 修复了循环缩进错误。
    3. 采用数值证据优先判定淘汰周。
    """
    season_data = raw_data[raw_data["season"] == season_num].copy()
    season_data = season_data.reset_index(drop=True)
    contestant_names = season_data["celebrity_name"].tolist()
    N = len(contestant_names)

    all_weeks = set()
    for col in season_data.columns:
        if "week" in col and "judge" in col:
            try:
                week_num = int(col.split("_")[0].replace("week", ""))
                all_weeks.add(week_num)
            except:
                continue

    if not all_weeks: return None, None, None, None
    max_theoretical_week = max(all_weeks)

    # 1. 构建评委分数矩阵 X（不进行错误的向前填充）
    X = np.zeros((N, max_theoretical_week))
    actual_max_week = 0

    for i in range(N):
        for w in range(1, max_theoretical_week + 1):
            week_cols = [col for col in season_data.columns if f"week{w}_" in col and "judge" in col]
            scores = []
            for col in week_cols:
                val = season_data.at[i, col]
                if pd.notna(val) and val != "N/A":
                    try:
                        v = float(val)
                        if v > 0: scores.append(v)
                    except:
                        pass

            if scores:
                X[i, w - 1] = np.mean(scores)
                actual_max_week = max(actual_max_week, w)

    # 截断矩阵到实际有比赛的周
    X = X[:, :actual_max_week]

    # 2. 识别决赛选手
    finalists = []
    for i in range(N):
        res_str = str(season_data.at[i, "results"])
        if "1st" in res_str or "Winner" in res_str:
            finalists.append({"idx": i, "rank": 1})
        elif "2nd" in res_str:
            finalists.append({"idx": i, "rank": 2})
        elif "3rd" in res_str:
            finalists.append({"idx": i, "rank": 3})

    # 3. 分配淘汰周 (修正缩进，直接看原始得分)
    elimination_info = []
    final_week_idx = actual_max_week - 1

    for i in range(N):
        res_str = str(season_data.at[i, "results"])
        is_champion = any(f["idx"] == i and f["rank"] == 1 for f in finalists)

        # 寻找真实最后得分周
        last_score_week = -1
        for w_idx in range(actual_max_week):
            if X[i, w_idx] > 0:
                last_score_week = w_idx

        if is_champion:
            elim_week = final_week_idx
        else:
            # 选手离开的周就是他最后一次有分的周
            elim_week = last_score_week

        # 兜底：如果完全没分，读文本
        if elim_week == -1:
            import re
            m = re.search(r"Week\s*(\d+)", res_str)
            elim_week = int(m.group(1)) - 1 if m else final_week_idx

        elimination_info.append({
            "index": i,
            "elim_week": max(0, min(elim_week, final_week_idx)),
            "name": contestant_names[i],
            "is_champion": is_champion
        })

    # 4. 构建淘汰序列 e 和淘汰周
    elimination_info.sort(key=lambda x: (x["elim_week"], X[x["index"], x["elim_week"]]))

    e = []
    elim_weeks = []
    for info in elimination_info:
        if info["is_champion"]: continue
        e.append(info["index"])
        elim_weeks.append(info["elim_week"])

    if debug:
        print(f"Season {season_num} processed: {N} contestants, {actual_max_week} weeks.")
        print(f"Elimination order: {[contestant_names[i] for i in e]}")

    return X, np.array(e), contestant_names, np.array(elim_weeks)
def analyze_with_final_model(season_num):
    """
    全能分析函数：
    1. 根据赛季自动选择 'rank' 或 'percent' 方法。
    2. 运行优化模型，输出你原有的评估结果。
    3. 添加详细的一致性和确定性分析，以回答题目要求。
    """
    print(f"\n{'=' * 60}")
    print(f"FINAL MODEL - SEASON {season_num}")
    print("=" * 60)

    # 1. [队友新增逻辑] 根据赛季号自动获取计算方法
    method = get_season_method(season_num)
    print(f"Using '{method.upper()}' method for season {season_num}")

    # 2. 准备数据 (假设你环境里已经有 enhanced_prepare_season_data 和 raw_data)
    try:
        X, e, names, elimination_weeks = enhanced_prepare_season_data(
            season_num, raw_data, debug=False  # debug设为False保持输出清爽
        )
    except NameError:
        print("Error: `raw_data` or `enhanced_prepare_season_data` not found.")
        return None

    if len(e) == 0:
        print("No elimination data found for this season.")
        return None

    print(f"Contestants: {X.shape[0]}, Weeks: {X.shape[1]}")
    print(f"Eliminations: {len(e)}")

    # 3. 拟合模型 (保持你原有的参数，确保结果不变)
    estimator = FinalFanVoteEstimator(
        method=method,  # 使用队友的逻辑
        lambda_constraint=10000.0,
        verbose=1  # 保持原来的 verbose 输出优化过程
    )
    estimator.fit(X, e, elimination_weeks=elimination_weeks.tolist())

    # 4. [保留原有输出] 基础评估
    results = estimator.evaluate()
    V = estimator.fan_votes_

    print(f"\n[Original Optimization Results]")
    print(f"  Satisfaction rate: {results['satisfaction_rate']:.1f}%")
    print(f"  Average violation: {results['avg_violation']:.6f}")

    # 打印每周违反情况 (你原来的输出)
    print("\nViolation by elimination week:")
    for i, week in enumerate(elimination_weeks):
        violation = results["violations"][i]
        actual_week = week + 1
        print(f"  Week {actual_week}: {violation:.6f}")

    # 打印 Top 5 (你原来的输出)
    if V.shape[1] > 0:  # 确保至少有一周的数据
        print(f"\nTop 5 estimated fan support (Week 1):")
        week1 = V[:, 0]
        # 归一化用于展示
        week1_norm = week1 / (week1.sum() + 1e-9)
        top_idx = np.argsort(-week1_norm)[:5]
        for i, idx in enumerate(top_idx, 1):
            print(f"  {i}. {names[idx]:20s}: {week1_norm[idx] * 100:5.1f}%")

    # 5. [新增输出] 详细一致性分析 (回答问题1)
    # 请确保你已经将 analyze_consistency 方法添加到了 FinalFanVoteEstimator 类中
    try:
        df_consistency = estimator.analyze_consistency(names)
    except AttributeError:
        print("\n[ERROR] `analyze_consistency` method not found in the class. Skipping analysis.")
        df_consistency = None

    # 6. [新增输出] 确定性分析 (回答问题2)
    # 请确保你已经将 analyze_certainty 方法添加到了 FinalFanVoteEstimator 类中
    try:
        df_certainty = estimator.analyze_certainty(names)

        # 打印确定性分析的摘要
        if df_certainty is not None and not df_certainty.empty:
            print("\n[Certainty Analysis Summary]")
            # 确定性最高的 (通常是被淘汰者)
            print("\nTop 3 Most Certain Estimates (Highest Certainty Score):")
            print(df_certainty.sort_values("Certainty", ascending=False).head(3)[
                      ['Week', 'Contestant', 'Certainty']].to_string(index=False))
            # 确定性最低的 (通常是安全区的领先者)
            print("\nTop 3 Least Certain Estimates (Lowest Certainty Score):")
            print(df_certainty.sort_values("Certainty", ascending=True).head(3)[
                      ['Week', 'Contestant', 'Certainty']].to_string(index=False))
            # 保存到文件
            df_certainty.to_csv(f"season_{season_num}_certainty_analysis.csv", index=False)
            print(
                f"\nFull certainty report for Season {season_num} saved to 'season_{season_num}_certainty_analysis.csv'")

    except AttributeError:
        print("\n[ERROR] `analyze_certainty` method not found in the class. Skipping analysis.")
        df_certainty = None

    return {
        "season": season_num,
        "results": results,
        "consistency": df_consistency,
        "certainty": df_certainty
    }
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

        result = None
        try:
            result = analyze_with_final_model(season)
        except Exception as e:
            print(f"ERROR processing season {season}: {e}")
            import traceback

            traceback.print_exc()

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
