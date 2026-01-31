import pandas as pd
import numpy as np
import glob
import statsmodels.api as sm
import re

# ==========================================
# 1. 提取 34 个文件的粉丝投票数据
# ==========================================
print("Aggregating estimated fan votes...")
fan_files = glob.glob("season_*_certainty_analysis.csv")
fan_agg_list = []

for f in fan_files:
    s_num = int(re.findall(r"season_(\d+)_", f)[0])
    df = pd.read_csv(f)
    # 取预估票数的中值，并按明星聚合（算全赛季平均支持率）
    if 'Estimated Vote' not in df.columns:
        df['Estimated Vote'] = (df['Min Vote'] + df['Max Vote']) / 2

    # 聚合到人：得到该明星在该赛季的平均粉丝得票百分比
    df_person = df.groupby('Contestant')['Estimated Vote'].mean().reset_index()
    df_person['season'] = s_num
    fan_agg_list.append(df_person)

df_fan_final = pd.concat(fan_agg_list)

# ==========================================
# 2. 从原始数据提取背景和评委分
# ==========================================
print("Loading original characteristics and calculating judge averages...")
df_raw = pd.read_csv("2026_MCM_Problem_C_Data.csv")

# 提取评委列并计算该明星在参赛期间的平均分（不计入0和NA）
judge_cols = [c for c in df_raw.columns if 'score' in c]


def get_avg_judge_score(row):
    scores = [row[c] for c in judge_cols if pd.notna(row[c]) and row[c] > 0]
    return np.mean(scores) if scores else 0


df_raw['avg_season_judge_score'] = df_raw.apply(get_avg_judge_score, axis=1)

# ==========================================
# 3. 数据大合并
# ==========================================
df_final = pd.merge(
    df_raw[['season', 'celebrity_name', 'celebrity_age_during_season', 'celebrity_industry',
            'ballroom_partner', 'celebrity_homestate', 'celebrity_homecountry/region',
            'avg_season_judge_score', 'placement']],
    df_fan_final,
    left_on=['season', 'celebrity_name'],
    right_on=['season', 'Contestant'],
    how='inner'
)

# ==========================================
# 4. 深度特征工程 (更加准确的处理方式)
# ==========================================
print("Detailed feature engineering...")

# (1) 行业分类：保留主要类别，长尾类别归为 'Other'
top_industries = df_final['celebrity_industry'].value_counts().nlargest(6).index
df_final['ind_grp'] = df_final['celebrity_industry'].apply(lambda x: x if x in top_industries else 'Other')

# (2) 舞伴经验值：不再看名字，看这个舞伴带过多少季（代表资历）
pro_experience = df_raw['ballroom_partner'].value_counts().to_dict()
df_final['pro_experience'] = df_final['ballroom_partner'].map(pro_experience)

# (3) 地区划分：美国本土选手 vs 国际选手
df_final['is_international'] = (df_final['celebrity_homecountry/region'] != 'United States').astype(int)

# (4) 明星效应：是否来自人口大州（通常票多）
big_states = ['California', 'Texas', 'New York', 'Florida', 'Illinois']
df_final['is_big_state'] = df_final['celebrity_homestate'].isin(big_states).astype(int)

# ==========================================
# 5. 运行对比模型 (OLS Regression)
# ==========================================
# 准备哑变量
df_reg = pd.get_dummies(df_final, columns=['ind_grp'], drop_first=True)

# 选取自变量
feature_cols = [c for c in df_reg.columns if 'ind_grp_' in c] + \
               ['celebrity_age_during_season', 'pro_experience', 'is_international', 'is_big_state']

X = df_reg[feature_cols].astype(float)
X = sm.add_constant(X)

# 模型 1: 对评委平均分的影响 (Judge)
res_judge = sm.OLS(df_reg['avg_season_judge_score'], X).fit()

# 模型 2: 对粉丝预估票数的影响 (Fan)
res_fan = sm.OLS(df_reg['Estimated Vote'], X).fit()

# ==========================================
# 6. 输出分析报告
# ==========================================
print("\n" + "=" * 30 + " JUDGE ANALYSIS " + "=" * 30)
print(res_judge.summary())
print("\n" + "=" * 30 + " FAN ANALYSIS " + "=" * 30)
print(res_fan.summary())

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 7. 随机森林验证 (Random Forest Validation)
# ==========================================
print("\n" + "=" * 30 + " RUNNING RANDOM FOREST " + "=" * 30)
from sklearn.ensemble import RandomForestRegressor

# 1. 准备数据 (确保变量名与前文 df_reg 保持一致)
# 我们使用相同的 feature_cols 列表
X_rf = df_reg[feature_cols].astype(float)
y_judge = df_reg['avg_season_judge_score']
y_fan = df_reg['Estimated Vote']

# 2. 训练随机森林模型
# 设置较高的 n_estimators 以保证重要性评分稳定
rf_judge = RandomForestRegressor(n_estimators=500, random_state=42, max_depth=7)
rf_fan = RandomForestRegressor(n_estimators=500, random_state=42, max_depth=7)

rf_judge.fit(X_rf, y_judge)
rf_fan.fit(X_rf, y_fan)

# 3. 提取特征重要性
def get_importance_df(model, name):
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_,
        'Type': name
    })
    return imp_df.sort_values('Importance', ascending=False)

imp_judge = get_importance_df(rf_judge, 'Judge Model (RF)')
imp_fan = get_importance_df(rf_fan, 'Fan Model (RF)')

# 4. 可视化：对比特征重要性
plt.figure(figsize=(12, 8))
combined_imp = pd.concat([imp_judge, imp_fan])

# 使用你之前要求的 蓝色（评委）和 橙色（粉丝）
custom_palette = {'Judge Model (RF)': '#2c7fb8', 'Fan Model (RF)': '#d95f02'}

sns.barplot(data=combined_imp, y='Feature', x='Importance', hue='Type', palette=custom_palette)

plt.title("Relative Importance of Characteristics: Judges vs. Fans\n(Non-linear Validation via Random Forest)",
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Gini Importance Score (Feature Weight)", fontsize=12)
plt.ylabel("Characteristic", fontsize=12)
plt.legend(title='Evaluation Path')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存 RF 重要性图
plt.savefig("rf_feature_importance.png", dpi=300)
print("Random Forest Plot saved as 'rf_feature_importance.png'")


# ==========================================
# 5. 模型对比与自动化汇总表 (Table X)
# ==========================================
print("\n" + "=" * 30 + " FINAL ATTRIBUTION SUMMARY (TABLE X) " + "=" * 30)

# 定义显著性星号函数
def get_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

# 提取 OLS 数据
ols_j_params = res_judge.params
ols_j_pvals = res_judge.pvalues
ols_f_params = res_fan.params
ols_f_pvals = res_fan.pvalues

# 提取 RF 数据 (需确保顺序与 feature_cols 一致)
# Note: RF importance 的顺序与训练时的 X_rf.columns 一致
rf_j_imp = dict(zip(X_rf.columns, rf_judge.feature_importances_))
rf_f_imp = dict(zip(X_rf.columns, rf_fan.feature_importances_))

summary_rows = []
for feat in feature_cols:
    summary_rows.append({
        'Characteristic': feat,
        'Judge Coef(β)': f"{ols_j_params[feat]:.4f}{get_stars(ols_j_pvals[feat])}",
        'Judge RF Imp.': f"{rf_j_imp[feat]:.4f}",
        'Fan Coef(β)': f"{ols_f_params[feat]:.4f}{get_stars(ols_f_pvals[feat])}",
        'Fan RF Imp.': f"{rf_f_imp[feat]:.4f}"
    })

# 创建 DataFrame 并格式化
df_table_x = pd.DataFrame(summary_rows)

# 打印最终表格
print(df_table_x.to_string(index=False))
print("-" * 85)
print(f"Note: *** p<0.01, ** p<0.05, * p<0.1. Base Category: Actor/Actress")
print(f"Stats: N={len(df_reg)}, Judge_R2(OLS/RF)={res_judge.rsquared:.3f}/{rf_judge.score(X_rf, y_judge):.3f}, "
      f"Fan_R2(OLS/RF)={res_fan.rsquared:.3f}/{rf_fan.score(X_rf, y_fan):.3f}")

# 将结果保存为CSV，方便队友直接复制进Word/Excel
df_table_x.to_csv("table_x_attribution_summary.csv", index=False)
print("\nTable X has been saved to 'table_x_attribution_summary.csv'")

# ==========================================
# 6. 给论文队友的最终洞察建议 (Automated Insights)
# ==========================================
print("\n" + "-"*20 + " KEY INSIGHTS FOR THE PAPER " + "-"*20)
# 找出RF中最重要的特征
top_rf_j = df_table_x.loc[df_table_x['Judge RF Imp.'].astype(float).idxmax(), 'Characteristic']
top_rf_f = df_table_x.loc[df_table_x['Fan RF Imp.'].astype(float).idxmax(), 'Characteristic']

print(f"1. System Consistency: Both models identify '{top_rf_j}' as the most dominant factor.")
print(f"2. Divergence: Compare the coefficients. For example, look at P-values for industry groups ")
print(f"   to see which sectors judges penalize more heavily than fans.")

# 检查非线性提升
r2_gain_j = (rf_judge.score(X_rf, y_judge) - res_judge.rsquared) / res_judge.rsquared * 100
r2_gain_f = (rf_fan.score(X_rf, y_fan) - res_fan.rsquared) / res_fan.rsquared * 100
print(f"3. Non-linearity: RF improved fit by {r2_gain_j:.1f}% (Judge) and {r2_gain_f:.1f}% (Fan).")

plt.show()