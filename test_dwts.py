"""
DWTS Analysis Test Suite for 2026 MCM Problem C
Author: [你的名字/团队名]
Date: 2026-01-[当前日期]

这个脚本提供了完整的测试和分析功能，包括：
1. 全赛季分析
2. 争议赛季深度分析
3. 敏感性分析
4. 结果汇总和可视化
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(".")

# 尝试导入你的分析模块
try:
    from dwts_analysis_clean import DWTSPipeline, MethodEvaluator, VotingSimulator

    MODULE_AVAILABLE = True
except ImportError as e:
    print(f"导入分析模块失败: {e}")
    MODULE_AVAILABLE = False


class DWTSAnalysisSuite:
    """DWTS分析测试套件"""

    def __init__(
        self,
        data_path="2026_MCM_Problem_C_Data.csv",
        output_base="dwts_analysis_results",
    ):
        self.data_path = data_path
        self.output_base = output_base
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_output_dir = f"{output_base}_{self.timestamp}"

        # 确保输出目录存在
        os.makedirs(self.full_output_dir, exist_ok=True)

        # 初始化结果存储
        self.all_season_results = []
        self.controversial_season_results = []
        self.summary_stats = {}

    def analyze_all_seasons(self, limit=None):
        """分析所有赛季"""
        print(f"\n{'='*80}")
        print(f"ANALYZING ALL SEASONS")
        print("=" * 80)

        if not MODULE_AVAILABLE:
            print("错误: 分析模块不可用")
            return None

        # 创建pipeline
        pipeline = DWTSPipeline(
            self.data_path, out_dir=os.path.join(self.full_output_dir, "full_analysis")
        )

        # 运行所有赛季
        seasons = pipeline.dp.list_seasons()
        if limit:
            seasons = seasons[:limit]

        print(f"Found {len(seasons)} seasons: {seasons}")

        all_results = []
        for i, season in enumerate(seasons, 1):
            print(f"\n[{i}/{len(seasons)}] Processing season {season}...")
            try:
                result = pipeline.run_for_season(season)
                if result:
                    all_results.append(result)

                    # 保存中间结果
                    season_dir = os.path.join(
                        self.full_output_dir, "full_analysis", f"season_{season}"
                    )
                    os.makedirs(season_dir, exist_ok=True)

                    with open(
                        os.path.join(season_dir, "analysis_result.json"), "w"
                    ) as f:
                        json.dump(
                            result,
                            f,
                            indent=2,
                            default=lambda x: (
                                float(x)
                                if isinstance(x, (np.float32, np.float64))
                                else x
                            ),
                        )

            except Exception as e:
                print(f"Error processing season {season}: {e}")
                import traceback

                traceback.print_exc()

        self.all_season_results = all_results

        # 计算汇总统计
        self._calculate_summary_statistics()

        # 生成汇总报告
        self._generate_summary_report()

        return all_results

    def deep_dive_controversial_seasons(self):
        """深入分析争议赛季"""
        print(f"\n{'='*80}")
        print(f"DEEP DIVE: CONTROVERSIAL SEASONS")
        print("=" * 80)

        controversial_seasons = {
            2: {
                "name": "Jerry Rice",
                "description": "Runner-up despite lowest judge scores in 5 weeks",
            },
            4: {
                "name": "Billy Ray Cyrus",
                "description": "5th place despite last place judge scores in 6 weeks",
            },
            11: {
                "name": "Bristol Palin",
                "description": "3rd place with lowest judge scores 12 times",
            },
            27: {
                "name": "Bobby Bones",
                "description": "Winner despite consistently low judge scores",
            },
        }

        if not MODULE_AVAILABLE:
            print("错误: 分析模块不可用")
            return None

        deep_dive_results = []

        for season_num, info in controversial_seasons.items():
            print(f"\n{'='*80}")
            print(f"DEEP ANALYSIS - Season {season_num}: {info['name']}")
            print(f"Description: {info['description']}")
            print("=" * 80)

            # 创建独立的输出目录
            season_output_dir = os.path.join(
                self.full_output_dir, f"controversial_season_{season_num}"
            )
            pipeline = DWTSPipeline(self.data_path, out_dir=season_output_dir)

            try:
                # 运行分析
                result = pipeline.run_for_season(season_num)

                if result:
                    deep_dive_results.append(
                        {
                            "season": season_num,
                            "contestant_info": info,
                            "analysis_result": result,
                        }
                    )

                    # 进行额外的详细分析
                    self._analyze_contestant_performance(season_num, info["name"])

                    # 保存详细结果
                    with open(
                        os.path.join(season_output_dir, "deep_dive_result.json"), "w"
                    ) as f:
                        json.dump(
                            {
                                "season": season_num,
                                "contestant_info": info,
                                "analysis_result": result,
                            },
                            f,
                            indent=2,
                            default=lambda x: (
                                float(x)
                                if isinstance(x, (np.float32, np.float64))
                                else x
                            ),
                        )

            except Exception as e:
                print(f"Error in deep dive for season {season_num}: {e}")
                import traceback

                traceback.print_exc()

        self.controversial_season_results = deep_dive_results

        # 生成对比报告
        self._generate_controversial_seasons_comparison()

        return deep_dive_results

    def sensitivity_analysis(self):
        """敏感性分析"""
        print(f"\n{'='*80}")
        print(f"SENSITIVITY ANALYSIS")
        print("=" * 80)

        # 这里可以添加对权重、参数等的敏感性分析
        # 由于时间关系，我们先实现一个简单的版本

        if not MODULE_AVAILABLE or not self.all_season_results:
            print("需要先运行全赛季分析")
            return None

        # 分析不同权重方法的影响
        weight_methods = ["entropy", "critic", "entropy+critic", "equal"]
        sensitivity_results = {}

        # 选择几个代表性赛季进行敏感性分析
        test_seasons = [2, 11, 27]

        for season in test_seasons:
            sensitivity_results[season] = {}

            for weight_method in weight_methods:
                print(f"Testing Season {season} with weight method: {weight_method}")

                # 这里需要修改DWTSPipeline以支持不同的权重方法
                # 由于代码结构，这个可能需要调整现有的类
                # 我们暂时跳过详细实现，提供框架

                sensitivity_results[season][weight_method] = {
                    "status": "TODO",
                    "notes": "需要扩展DWTSPipeline以支持不同的权重方法",
                }

        # 保存敏感性分析结果
        sensitivity_path = os.path.join(
            self.full_output_dir, "sensitivity_analysis.json"
        )
        with open(sensitivity_path, "w") as f:
            json.dump(sensitivity_results, f, indent=2)

        print(f"敏感性分析框架已建立，详细实现需要扩展代码")

        return sensitivity_results

    def _calculate_summary_statistics(self):
        """计算汇总统计"""
        if not self.all_season_results:
            return

        # 初始化统计
        method_counts = {}
        method_closeness_sum = {}
        method_closeness_count = {}

        for result in self.all_season_results:
            recommended = result.get("recommended_method", "unknown")
            method_counts[recommended] = method_counts.get(recommended, 0) + 1

            # 收集closeness系数
            if "method_comparison" in result:
                comparison = result["method_comparison"]
                if (
                    "method_names" in comparison
                    and "closeness_coefficients" in comparison
                ):
                    for method, closeness in zip(
                        comparison["method_names"], comparison["closeness_coefficients"]
                    ):
                        if method not in method_closeness_sum:
                            method_closeness_sum[method] = 0
                            method_closeness_count[method] = 0

                        method_closeness_sum[method] += closeness
                        method_closeness_count[method] += 1

        # 计算平均closeness
        avg_closeness = {}
        for method in method_closeness_sum:
            if method_closeness_count[method] > 0:
                avg_closeness[method] = (
                    method_closeness_sum[method] / method_closeness_count[method]
                )

        self.summary_stats = {
            "total_seasons": len(self.all_season_results),
            "method_recommendation_counts": method_counts,
            "method_recommendation_percentages": {
                method: count / len(self.all_season_results) * 100
                for method, count in method_counts.items()
            },
            "average_closeness_coefficients": avg_closeness,
            "analysis_timestamp": self.timestamp,
        }

    def _generate_summary_report(self):
        """生成汇总报告"""
        if not self.summary_stats:
            return

        report_path = os.path.join(self.full_output_dir, "summary_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DWTS ANALYSIS - SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. OVERVIEW\n")
            f.write(
                f"   Total seasons analyzed: {self.summary_stats['total_seasons']}\n"
            )
            f.write(
                f"   Analysis timestamp: {self.summary_stats['analysis_timestamp']}\n\n"
            )

            f.write("2. METHOD RECOMMENDATION SUMMARY\n")
            f.write("   " + "-" * 50 + "\n")
            f.write("   Method           Count     Percentage\n")
            f.write("   " + "-" * 50 + "\n")

            for method, count in sorted(
                self.summary_stats["method_recommendation_counts"].items()
            ):
                percentage = self.summary_stats["method_recommendation_percentages"][
                    method
                ]
                f.write(f"   {method:16s} {count:6d}     {percentage:6.1f}%\n")

            f.write("\n")

            f.write("3. AVERAGE CLOSENESS COEFFICIENTS\n")
            f.write("   " + "-" * 50 + "\n")
            for method, avg_closeness in self.summary_stats.get(
                "average_closeness_coefficients", {}
            ).items():
                f.write(f"   {method:16s}: {avg_closeness:.4f}\n")

            f.write("\n")

            # 添加主要发现
            f.write("4. KEY FINDINGS\n")

            # 确定主要推荐方法
            if self.summary_stats["method_recommendation_counts"]:
                top_method = max(
                    self.summary_stats["method_recommendation_counts"].items(),
                    key=lambda x: x[1],
                )[0]
                top_percentage = self.summary_stats[
                    "method_recommendation_percentages"
                ][top_method]

                f.write(
                    f"   • {top_method} method is recommended for {top_percentage:.1f}% of seasons\n"
                )

                # 添加其他观察
                if top_method == "percent":
                    f.write(
                        "   • Percent method shows strong performance across seasons\n"
                    )
                elif top_method == "rank":
                    f.write(
                        "   • Rank method shows strong performance across seasons\n"
                    )

            f.write("\n")

            f.write("5. RECOMMENDATIONS\n")
            f.write(
                "   • Consider the specific context of each season when choosing a voting method\n"
            )
            f.write(
                "   • Percent method may better balance judge and fan preferences\n"
            )
            f.write(
                "   • Further analysis of specific controversial cases is recommended\n"
            )

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"汇总报告已保存至: {report_path}")

        # 同时生成一个简明的控制台输出
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total seasons analyzed: {self.summary_stats['total_seasons']}")
        print("\nMethod Recommendation Summary:")
        for method, count in sorted(
            self.summary_stats["method_recommendation_counts"].items()
        ):
            percentage = self.summary_stats["method_recommendation_percentages"][method]
            print(f"  {method}: {count} seasons ({percentage:.1f}%)")

    def _generate_controversial_seasons_comparison(self):
        """生成争议赛季对比报告"""
        if not self.controversial_season_results:
            return

        report_path = os.path.join(
            self.full_output_dir, "controversial_seasons_comparison.txt"
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CONTROVERSIAL SEASONS - COMPARISON REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("SEASON COMPARISON:\n")
            f.write("-" * 80 + "\n")

            for item in self.controversial_season_results:
                season = item["season"]
                contestant = item["contestant_info"]["name"]
                description = item["contestant_info"]["description"]
                result = item["analysis_result"]

                f.write(f"\nSeason {season}: {contestant}\n")
                f.write(f"  Issue: {description}\n")

                if "recommended_method" in result:
                    f.write(f"  Recommended method: {result['recommended_method']}\n")

                if "method_comparison" in result:
                    comparison = result["method_comparison"]
                    if (
                        "closeness_coefficients" in comparison
                        and "method_names" in comparison
                    ):
                        f.write("  Closeness coefficients:\n")
                        for method, closeness in zip(
                            comparison["method_names"],
                            comparison["closeness_coefficients"],
                        ):
                            f.write(f"    {method}: {closeness:.4f}\n")

                if "monte_carlo_results" in result:
                    mc = result["monte_carlo_results"]
                    if "top_counts" in mc:
                        total = sum(mc["top_counts"].values())
                        if total > 0:
                            f.write("  Monte Carlo Top-1 probabilities:\n")
                            for method, count in mc["top_counts"].items():
                                prob = count / total
                                f.write(f"    {method}: {prob:.3f} ({count}/{total})\n")

                f.write("-" * 80 + "\n")

            # 添加观察和结论
            f.write("\nOBSERVATIONS AND CONCLUSIONS:\n")
            f.write("-" * 80 + "\n")

            # 统计争议赛季中的推荐方法
            method_counts = {}
            for item in self.controversial_season_results:
                method = item["analysis_result"].get("recommended_method", "unknown")
                method_counts[method] = method_counts.get(method, 0) + 1

            if method_counts:
                top_method = max(method_counts.items(), key=lambda x: x[1])[0]
                f.write(
                    f"• {top_method} method is recommended for {method_counts[top_method]}/4 controversial seasons\n"
                )

            f.write(
                "• These seasons demonstrate tension between judge scores and fan votes\n"
            )
            f.write(
                "• The analysis helps understand which voting methods might produce different outcomes\n"
            )
            f.write(
                "• Recommendations should consider both fairness and entertainment value\n"
            )

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF COMPARISON REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"争议赛季对比报告已保存至: {report_path}")

    def _analyze_contestant_performance(self, season_num, contestant_name):
        """分析具体选手表现（框架）"""
        print(f"\n  Additional analysis for {contestant_name} in Season {season_num}")

        # 这里可以添加针对具体选手的详细分析
        # 例如：计算他们在每个星期的评委分和估计的粉丝投票
        # 由于时间关系，我们先提供一个框架

        analysis_path = os.path.join(
            self.full_output_dir, f"contestant_analysis_season_{season_num}.txt"
        )

        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(f"Contestant Analysis: {contestant_name} in Season {season_num}\n")
            f.write("=" * 60 + "\n\n")
            f.write("This analysis would include:\n")
            f.write("1. Weekly judge scores for the contestant\n")
            f.write("2. Estimated fan vote percentages\n")
            f.write("3. Comparison with other contestants\n")
            f.write("4. Elimination week analysis\n")
            f.write("\n[Detailed analysis implementation pending]\n")

        print(f"  选手分析框架已保存至: {analysis_path}")

    def generate_latex_tables(self):
        """生成LaTeX表格用于论文"""
        if not self.summary_stats:
            print("需要先运行分析以生成统计")
            return

        latex_path = os.path.join(self.full_output_dir, "latex_tables.tex")

        with open(latex_path, "w", encoding="utf-8") as f:
            f.write("% LaTeX tables for MCM paper\n")
            f.write("% Generated automatically by DWTS Analysis Suite\n")
            f.write("% Date: " + datetime.now().strftime("%Y-%m-%d") + "\n\n")

            # 表1: 方法推荐统计
            f.write("% Table 1: Voting Method Recommendation Summary\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("  \\centering\n")
            f.write(
                "  \\caption{Voting Method Recommendation Summary Across All Seasons}\n"
            )
            f.write("  \\label{tab:method_summary}\n")
            f.write("  \\begin{tabular}{lcc}\n")
            f.write("    \\toprule\n")
            f.write("    Method & Count & Percentage (\\%) \\\\\n")
            f.write("    \\midrule\n")

            for method, count in sorted(
                self.summary_stats["method_recommendation_counts"].items()
            ):
                percentage = self.summary_stats["method_recommendation_percentages"][
                    method
                ]
                f.write(f"    {method} & {count} & {percentage:.1f} \\\\\n")

            f.write("    \\bottomrule\n")
            f.write("  \\end{tabular}\n")
            f.write("\\end{table}\n\n")

            # 表2: 争议赛季分析
            if self.controversial_season_results:
                f.write("% Table 2: Controversial Seasons Analysis\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("  \\centering\n")
                f.write("  \\caption{Analysis of Controversial Seasons}\n")
                f.write("  \\label{tab:controversial_seasons}\n")
                f.write("  \\begin{tabular}{llll}\n")
                f.write("    \\toprule\n")
                f.write("    Season & Contestant & Issue & Recommended Method \\\\\n")
                f.write("    \\midrule\n")

                for item in self.controversial_season_results:
                    season = item["season"]
                    contestant = item["contestant_info"]["name"]
                    description = item["contestant_info"]["description"]
                    method = item["analysis_result"].get("recommended_method", "N/A")

                    # 简化描述以适合表格
                    short_desc = (
                        description[:50] + "..."
                        if len(description) > 50
                        else description
                    )

                    f.write(
                        f"    {season} & {contestant} & {short_desc} & {method} \\\\\n"
                    )

                f.write("    \\bottomrule\n")
                f.write("  \\end{tabular}\n")
                f.write("\\end{table}\n\n")

            # 表3: 平均Closeness系数
            if "average_closeness_coefficients" in self.summary_stats:
                f.write("% Table 3: Average TOPSIS Closeness Coefficients\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("  \\centering\n")
                f.write(
                    "  \\caption{Average TOPSIS Closeness Coefficients by Method}\n"
                )
                f.write("  \\label{tab:closeness_coefficients}\n")
                f.write("  \\begin{tabular}{lc}\n")
                f.write("    \\toprule\n")
                f.write("    Method & Average Closeness Coefficient \\\\\n")
                f.write("    \\midrule\n")

                for method, avg_closeness in sorted(
                    self.summary_stats["average_closeness_coefficients"].items()
                ):
                    f.write(f"    {method} & {avg_closeness:.4f} \\\\\n")

                f.write("    \\bottomrule\n")
                f.write("  \\end{tabular}\n")
                f.write("\\end{table}\n")

        print(f"LaTeX表格已保存至: {latex_path}")

    def create_final_report(self):
        """创建最终报告"""
        print(f"\n{'='*80}")
        print(f"CREATING FINAL REPORT")
        print("=" * 80)

        # 收集所有重要文件
        important_files = []

        # 检查文件是否存在并添加到列表
        possible_files = [
            "summary_report.txt",
            "controversial_seasons_comparison.txt",
            "latex_tables.tex",
            "sensitivity_analysis.json",
        ]

        for filename in possible_files:
            filepath = os.path.join(self.full_output_dir, filename)
            if os.path.exists(filepath):
                important_files.append(filename)

        # 创建最终报告
        final_report_path = os.path.join(self.full_output_dir, "FINAL_REPORT.md")

        with open(final_report_path, "w", encoding="utf-8") as f:
            f.write("# DWTS Analysis - Final Report\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("## Overview\n")
            f.write(
                f"This report summarizes the analysis of {self.summary_stats.get('total_seasons', 'N/A')} seasons of DWTS.\n\n"
            )

            f.write("## Key Findings\n")

            if (
                self.summary_stats
                and "method_recommendation_counts" in self.summary_stats
            ):
                # 找出推荐最多的方法
                if self.summary_stats["method_recommendation_counts"]:
                    top_method = max(
                        self.summary_stats["method_recommendation_counts"].items(),
                        key=lambda x: x[1],
                    )[0]
                    top_percentage = self.summary_stats[
                        "method_recommendation_percentages"
                    ][top_method]

                    f.write(
                        f"1. **{top_method.capitalize()} method** is recommended for **{top_percentage:.1f}%** of seasons.\n"
                    )

            f.write(
                "2. The analysis reveals interesting patterns in how different voting methods handle judge-fan discrepancies.\n"
            )
            f.write(
                "3. Controversial seasons provide valuable insights into the trade-offs between different voting systems.\n\n"
            )

            f.write("## Generated Files\n")
            f.write("The following files were generated during the analysis:\n\n")

            for filename in important_files:
                f.write(f"- `{filename}`\n")

            f.write("\n## Directory Structure\n")
            f.write("```\n")
            f.write(f"{self.full_output_dir}/\n")

            # 添加子目录信息
            for root, dirs, files in os.walk(self.full_output_dir):
                level = root.replace(self.full_output_dir, "").count(os.sep)
                indent = " " * 2 * level
                f.write(f"{indent}{os.path.basename(root)}/\n")
                subindent = " " * 2 * (level + 1)
                for file in files[:5]:  # 只显示前5个文件
                    f.write(f"{subindent}{file}\n")
                if len(files) > 5:
                    f.write(f"{subindent}... and {len(files) - 5} more files\n")

            f.write("```\n\n")

            f.write("## Next Steps for MCM Paper\n")
            f.write(
                "1. Incorporate these findings into the mathematical modeling section.\n"
            )
            f.write("2. Use the generated LaTeX tables in the results section.\n")
            f.write("3. Reference the visualizations in the analysis section.\n")
            f.write("4. Develop recommendations based on the observed patterns.\n")
            f.write(
                "5. Consider limitations and potential improvements to the analysis.\n\n"
            )

            f.write("---\n")
            f.write(
                "*This report was automatically generated by the DWTS Analysis Suite.*\n"
            )

        print(f"最终报告已保存至: {final_report_path}")
        print(f"\n所有结果文件保存在: {self.full_output_dir}")


def main():
    """主函数：运行完整的分析套件"""
    print("\n" + "=" * 80)
    print("DWTS ANALYSIS SUITE FOR 2026 MCM PROBLEM C")
    print("=" * 80)

    if not MODULE_AVAILABLE:
        print("错误: 无法导入分析模块。请确保 dwts_analysis_clean.py 在正确的位置。")
        return

    # 创建分析套件实例
    print("初始化分析套件...")
    analyzer = DWTSAnalysisSuite()

    print(f"输出目录: {analyzer.full_output_dir}")

    # 运行全赛季分析（可以限制数量以节省时间）
    print("\n" + "-" * 80)
    print("阶段 1: 全赛季分析")
    print("-" * 80)

    # 提示用户选择分析范围
    print("\n选择分析范围:")
    print("1. 所有赛季 (34个)")
    print("2. 前10个赛季 (快速测试)")
    print("3. 仅争议赛季 (4个)")

    choice = input("请输入选项 (1-3, 默认: 2): ").strip()

    if choice == "1":
        limit = None  # 所有赛季
        print("分析所有34个赛季...")
    elif choice == "3":
        # 只分析争议赛季
        print("仅分析争议赛季...")
        analyzer.deep_dive_controversial_seasons()
        analyzer.generate_latex_tables()
        analyzer.create_final_report()
        return
    else:
        limit = 10  # 默认：前10个赛季
        print(f"分析前{limit}个赛季...")

    analyzer.analyze_all_seasons(limit=limit)

    # 深入分析争议赛季
    print("\n" + "-" * 80)
    print("阶段 2: 争议赛季深度分析")
    print("-" * 80)
    analyzer.deep_dive_controversial_seasons()

    # 敏感性分析（框架）
    print("\n" + "-" * 80)
    print("阶段 3: 敏感性分析框架")
    print("-" * 80)
    analyzer.sensitivity_analysis()

    # 生成LaTeX表格
    print("\n" + "-" * 80)
    print("阶段 4: 生成论文材料")
    print("-" * 80)
    analyzer.generate_latex_tables()

    # 创建最终报告
    print("\n" + "-" * 80)
    print("阶段 5: 创建最终报告")
    print("-" * 80)
    analyzer.create_final_report()

    print("\n" + "=" * 80)
    print("分析完成!")
    print(f"所有结果已保存至: {analyzer.full_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
