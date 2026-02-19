import pandas as pd
import matplotlib.pyplot as plt

# ============ Load data ============
ob = pd.read_csv("data/Obligor_panel.csv")

# ============ Q1: Average PD by rating ============
avg_pd_by_rating = (
    ob.groupby("rating")["PD"]
      .agg(mean="mean", count="count")
      .reset_index()
      .sort_values("mean")
)

print("\n=== Q1: Average PD by rating ===")
print(avg_pd_by_rating.to_string(index=False))

# ============ Q2: Quarterly average PD with GFC/COVID shading ============
q_avg = (
    ob.groupby("date")
      .agg(avg_PD=("PD", "mean"),
           is_gfc=("is_gfc", "max"),
           is_covid=("is_covid", "max"))
      .reset_index()
)

# date 是类似 2005Q2，这里转成 period 方便排序
q_avg["date_period"] = pd.PeriodIndex(q_avg["date"], freq="Q")
q_avg = q_avg.sort_values("date_period")

# --- helper: find contiguous shaded intervals ---
def contiguous_intervals(periods):
    """periods: list/array of pandas Period (sorted). return list of (start, end) Period inclusive."""
    if len(periods) == 0:
        return []
    periods = list(periods)
    intervals = []
    start = prev = periods[0]
    for p in periods[1:]:
        if p == prev + 1:
            prev = p
        else:
            intervals.append((start, prev))
            start = prev = p
    intervals.append((start, prev))
    return intervals

gfc_periods = q_avg.loc[q_avg["is_gfc"] == 1, "date_period"].tolist()
covid_periods = q_avg.loc[q_avg["is_covid"] == 1, "date_period"].tolist()

gfc_intervals = contiguous_intervals(sorted(gfc_periods))
covid_intervals = contiguous_intervals(sorted(covid_periods))

# x轴用 timestamp（季度起始日期）
x = q_avg["date_period"].dt.to_timestamp()
y = q_avg["avg_PD"]

plt.figure(figsize=(12, 5))
plt.plot(x, y)

# 阴影：GFC
for (s, e) in gfc_intervals:
    plt.axvspan(s.start_time, e.end_time, alpha=0.2)

# 阴影：COVID
for (s, e) in covid_intervals:
    plt.axvspan(s.start_time, e.end_time, alpha=0.2)

plt.title("Quarterly Average PD (All Obligors)")
plt.xlabel("Quarter")
plt.ylabel("Average PD")

plt.tight_layout()
plt.savefig("outputs/Q2_avg_PD_timeseries.png", dpi=200)
print("\n=== Q2 saved figure: Q2_avg_PD_timeseries.png ===")

import numpy as np
from scipy.stats import norm

# ============ Q3: Quarterly average stressed PD for multiple rhos ============
alpha = 0.999
z_alpha = norm.ppf(alpha)
rhos = [0.1, 0.2, 0.3, 0.4]

# 避免 PD=0 或 1 导致 norm.ppf 无穷
pd_clip = ob["PD"].clip(1e-12, 1 - 1e-12)
z_pd = norm.ppf(pd_clip)

# 计算每个 rho 下的 stressed PD（逐行）
for rho in rhos:
    stressed = norm.cdf((z_pd + np.sqrt(rho) * z_alpha) / np.sqrt(1 - rho))
    ob[f"PD_stress_rho_{rho}"] = stressed

# 按季度取平均（并保留危机指示用于阴影）
q3 = (
    ob.groupby("date")
      .agg(**{f"avg_PD_stress_rho_{rho}": (f"PD_stress_rho_{rho}", "mean") for rho in rhos},
           is_gfc=("is_gfc", "max"),
           is_covid=("is_covid", "max"))
      .reset_index()
)

q3["date_period"] = pd.PeriodIndex(q3["date"], freq="Q")
q3 = q3.sort_values("date_period")

# 阴影区间（复用 Q2 的函数）
gfc_periods = q3.loc[q3["is_gfc"] == 1, "date_period"].tolist()
covid_periods = q3.loc[q3["is_covid"] == 1, "date_period"].tolist()

gfc_intervals = contiguous_intervals(sorted(gfc_periods))
covid_intervals = contiguous_intervals(sorted(covid_periods))

x = q3["date_period"].dt.to_timestamp()

plt.figure(figsize=(12, 5))

# 画 4 条线（不指定颜色，让 matplotlib 自动分配）
for rho in rhos:
    plt.plot(x, q3[f"avg_PD_stress_rho_{rho}"], label=f"rho={rho}")

# 阴影：GFC
for (s, e) in gfc_intervals:
    plt.axvspan(s.start_time, e.end_time, alpha=0.2)

# 阴影：COVID
for (s, e) in covid_intervals:
    plt.axvspan(s.start_time, e.end_time, alpha=0.2)

plt.title("Quarterly Average Stressed PD (Vasicek Mapping)")
plt.xlabel("Quarter")
plt.ylabel("Average Stressed PD")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/Q3_avg_stressed_PD_timeseries.png", dpi=200)
print("\n=== Q3 saved figure: Q3_avg_stressed_PD_timeseries.png ===")

# ============ Q4: Construct TTC PD ============

# 对每个 obligor 计算时间平均 PD
ttc_pd = (
    ob.groupby("obligorID")["PD"]
      .mean()
      .reset_index()
      .rename(columns={"PD": "PD_TTC"})
)

print("\n=== Q4: First 10 TTC PDs ===")
print(ttc_pd.head(10))

from scipy.stats import norm
import numpy as np

# ============ Q5: Basel stressed PD using TTC ============

alpha = 0.999
z_alpha = norm.ppf(alpha)

rho = 0.2  # 题目没有指定，但通常选一个代表值；后面Q6会用多个

# 防止0或1
ttc_pd["PD_TTC_clipped"] = ttc_pd["PD_TTC"].clip(1e-12, 1-1e-12)

z_ttc = norm.ppf(ttc_pd["PD_TTC_clipped"])

ttc_pd["PD_stress"] = norm.cdf(
    (z_ttc + np.sqrt(rho) * z_alpha) / np.sqrt(1 - rho)
)

print("\n=== Q5: First 10 Basel stressed PDs (rho=0.2) ===")
print(ttc_pd[["obligorID", "PD_TTC", "PD_stress"]].head(10))

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============ Q6: 3D plot of stressed PD as function of rating and rho ============

# 1) 准备：把 PD_TTC 合并回 obligor panel（让每一行都有 PD_TTC）
ob_with_ttc = ob.merge(ttc_pd[["obligorID", "PD_TTC"]], on="obligorID", how="left")

# 2) 计算多个 rho 下的 stressed PD（基于 TTC，不是 PIT）
alpha = 0.999
z_alpha = norm.ppf(alpha)
rhos = [0.1, 0.2, 0.3, 0.4]

pd_ttc_clip = ob_with_ttc["PD_TTC"].clip(1e-12, 1 - 1e-12)
z_ttc = norm.ppf(pd_ttc_clip)

for rho in rhos:
    ob_with_ttc[f"PD_stress_TTC_rho_{rho}"] = norm.cdf(
        (z_ttc + np.sqrt(rho) * z_alpha) / np.sqrt(1 - rho)
    )

# 3) rating 顺序（按信用等级从好到坏）
rating_order = ["AAA","AA","A","BBB","BB","B","CCC","CC","C","D"]
ob_with_ttc["rating"] = pd.Categorical(ob_with_ttc["rating"], categories=rating_order, ordered=True)

# 4) 生成 rating × rho 的平均 stressed PD 表
rows = []
for rho in rhos:
    tmp = (ob_with_ttc
           .groupby("rating")[f"PD_stress_TTC_rho_{rho}"]
           .mean()
           .reset_index()
           .rename(columns={f"PD_stress_TTC_rho_{rho}": "avg_PD_stress"}))
    tmp["rho"] = rho
    rows.append(tmp)

q6_tbl = pd.concat(rows, ignore_index=True).dropna()

print("\n=== Q6: Average stressed PD by rating and rho (based on TTC) ===")
print(q6_tbl.head(20))

# 5) 画 3D surface
# 构造网格：X=rating index, Y=rho, Z=avg stressed PD
rating_to_x = {r:i for i, r in enumerate(rating_order)}
q6_tbl["x"] = q6_tbl["rating"].map(rating_to_x)

# pivot 成矩阵
Z = q6_tbl.pivot(index="rho", columns="rating", values="avg_PD_stress").reindex(index=rhos, columns=rating_order).values
X, Y = np.meshgrid(np.arange(len(rating_order)), np.array(rhos))

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)

ax.set_title("Stressed PD (Basel Vasicek) by Rating and Asset Correlation")
ax.set_xlabel("Rating")
ax.set_ylabel("rho")
ax.set_zlabel("Avg Stressed PD")

ax.set_xticks(np.arange(len(rating_order)))
ax.set_xticklabels(rating_order)

plt.tight_layout()
plt.savefig("outputs/Q6_3D_stressed_PD.png", dpi=200)
print("\n=== Q6 saved figure: Q6_3D_stressed_PD.png ===")

# ============ Q7: Total EAD by credit rating (merge rating from obligor panel) ============

import pandas as pd
import matplotlib.pyplot as plt

# 读 facility & obligor（ob 你前面已经读过，如果没读就再读一次也行）
fac = pd.read_csv("data/facility_quarter_panel.csv")
ob = pd.read_csv("data/Obligor_panel.csv")

# 从 obligor panel 取 (obligorID, date, rating) 去重
rating_map = (
    ob[["obligorID", "date", "rating"]]
    .drop_duplicates(subset=["obligorID", "date"])
)

# merge：把 rating 合并到 facility 数据
fac2 = fac.merge(rating_map, on=["obligorID", "date"], how="inner")

# 检查是否有没匹配上的 rating
missing_rate = fac2["rating"].isna().mean()
print(f"\nQ7: share of facility rows with missing rating after merge = {missing_rate:.4%}")

# 按 rating 汇总 EAD
ead_by_rating = (
    fac2.groupby("rating")["EAD"]
        .sum()
        .reset_index()
)

# 排序：按信用等级从好到坏
rating_order = ["AAA","AA","A","BBB","BB","B","CCC","CC","C","D"]
ead_by_rating["rating"] = pd.Categorical(ead_by_rating["rating"], categories=rating_order, ordered=True)
ead_by_rating = ead_by_rating.sort_values("rating")

print("\n=== Q7: Total EAD by rating ===")
print(ead_by_rating.to_string(index=False))

# 画图
plt.figure(figsize=(10,5))
plt.bar(ead_by_rating["rating"].astype(str), ead_by_rating["EAD"])
plt.title("Total Exposure at Default (EAD) by Credit Rating")
plt.xlabel("Credit Rating")
plt.ylabel("Total EAD")
plt.tight_layout()
plt.savefig("outputs/Q7_total_EAD_by_rating.png", dpi=200)
print("\n=== Q7 saved figure: Q7_total_EAD_by_rating.png ===")

# ============ Q8: Facility-level Unexpected Loss (UL), rho=0.2 ============

import numpy as np
import pandas as pd

# 读 facility 数据
fac = pd.read_csv("data/facility_quarter_panel.csv")

# 1) 计算 LGD^DT：只用 default events 且在 GFC 或 COVID 窗口内
lgd_dt_sample = fac[
    (fac["is_default_event_row"] == 1) &
    ((fac["is_gfc"] == 1) | (fac["is_covid"] == 1)) &
    (fac["realized_LGD"].notna())
].copy()

LGD_DT = lgd_dt_sample["realized_LGD"].mean()
print(f"\n=== Q8: LGD^DT (avg realized_LGD in GFC/COVID defaults) = {LGD_DT:.6f} ===")
print(f"Q8: default-event rows used for LGD^DT = {len(lgd_dt_sample)}")

# 2) 准备 obligor-level PD_TTC 和 PD_stress(rho=0.2)
# ttc_pd 你在 Q4 已经生成过；Q5 生成了 PD_stress（rho=0.2）
# 这里确保只保留需要的列
obligor_pd = ttc_pd[["obligorID", "PD_TTC", "PD_stress"]].copy()
obligor_pd["PD_diff"] = obligor_pd["PD_stress"] - obligor_pd["PD_TTC"]

# 3) merge 到 facility panel（按 obligorID）
fac_ul = fac.merge(obligor_pd, on="obligorID", how="left")

missing_pd = fac_ul["PD_TTC"].isna().mean()
print(f"Q8: share of facility rows with missing PD_TTC after merge = {missing_pd:.4%}")

# 4) 计算 UL（facility-quarter level）
fac_ul["UL"] = fac_ul["EAD"] * LGD_DT * fac_ul["PD_diff"]

print("\n=== Q8: UL summary ===")
print(fac_ul["UL"].describe())

# （可选）保存结果，后面 Q9/Q10 会用到
fac_ul.to_csv("data/facility_with_UL.csv", index=False)
print("\n=== Q8 saved file: facility_with_UL.csv ===")

# ============ Q9: Histogram of facility-level UL (rho=0.2) with percentile cutoffs ============

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 Q8 输出（如果你没保存，也可以直接用 fac_ul，但读csv最稳）
fac_ul = pd.read_csv("data/facility_with_UL.csv")

# 取 UL 并去掉缺失
ul = fac_ul["UL"].dropna()

# 计算分位点：10, 20, 40, 60, 80, 90
p10, p20, p40, p60, p80, p90 = ul.quantile([0.10, 0.20, 0.40, 0.60, 0.80, 0.90])

print("\n=== Q9: UL percentiles ===")
print(f"10th={p10:.6f}, 20th={p20:.6f}, 40th={p40:.6f}, 60th={p60:.6f}, 80th={p80:.6f}, 90th={p90:.6f}")

# 限制样本在 10th–90th 之间
ul_trim = ul[(ul >= p10) & (ul <= p90)]

# 画直方图
plt.figure(figsize=(10,5))
plt.hist(ul_trim, bins=40)

# 标出 20/40/60/80 分位数
for val, lab in [(p20, "20th"), (p40, "40th"), (p60, "60th"), (p80, "80th")]:
    plt.axvline(val, linestyle="--", linewidth=1, label=lab)

plt.title("Histogram of Facility-level Unexpected Loss (UL)\n(Restricted to 10th–90th Percentiles)")
plt.xlabel("Unexpected Loss (UL)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/Q9_UL_hist_10_90_with_percentiles.png", dpi=200)
print("\n=== Q9 saved figure: Q9_UL_hist_10_90_with_percentiles.png ===")

# ============ Q10: 3D Aggregate UL by Rating and rho (2025Q4 snapshot) ============

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 1) 读原始 facility panel（有 EAD、date、obligorID）
fac = pd.read_csv("data/facility_quarter_panel.csv")

# 2) 读 obligor panel（提供 rating）
ob = pd.read_csv("data/Obligor_panel.csv")
rating_map = ob[["obligorID", "date", "rating"]].drop_duplicates(["obligorID", "date"])

# merge rating 到 facility（inner join，保证没有 missing rating）
fac2 = fac.merge(rating_map, on=["obligorID", "date"], how="inner")

# 3) 只取 2025Q4 snapshot
snap = fac2[fac2["date"] == "2025Q4"].copy()
print(f"\nQ10: 2025Q4 snapshot size (after rating merge) = {len(snap)}")

# 4) 准备 obligor-level PD_TTC（来自 Q4 的 ttc_pd）
# 确保 PD_TTC 不为0/1
ttc = ttc_pd[["obligorID", "PD_TTC"]].copy()
ttc["PD_TTC"] = ttc["PD_TTC"].clip(1e-12, 1 - 1e-12)

alpha = 0.999
z_alpha = norm.ppf(alpha)
rho_list = [0.1, 0.2, 0.3, 0.4]

# rating 排序（与前面一致）
rating_order = ["AAA","AA","A","BBB","BB","B","CCC","CC","C","D"]
snap["rating"] = pd.Categorical(snap["rating"], categories=rating_order, ordered=True)

# 5) 对每个 rho：算 PD_stress(TTC, rho)，算 UL_rho，再按 rating 求和
results = []
for rho in rho_list:
    # stressed PD and PD_diff at obligor level
    z_ttc = norm.ppf(ttc["PD_TTC"])
    pd_stress = norm.cdf((z_ttc + np.sqrt(rho) * z_alpha) / np.sqrt(1 - rho))
    ttc[f"PD_diff_{rho}"] = pd_stress - ttc["PD_TTC"]

    # merge PD_diff 到 snapshot
    tmp = snap.merge(ttc[["obligorID", f"PD_diff_{rho}"]], on="obligorID", how="left")

    # 计算 UL (facility level), 使用 Q8 的 LGD_DT 常数
    tmp["UL_rho"] = tmp["EAD"] * LGD_DT * tmp[f"PD_diff_{rho}"]

    # aggregate UL: sum across facilities by rating
    agg = tmp.groupby("rating", observed=True)["UL_rho"].sum().reset_index()
    agg["rho"] = rho
    results.append(agg)

agg_all = pd.concat(results, ignore_index=True)

# 6) 构造 3D surface 的 Z 矩阵
rho_vals = rho_list
ratings = rating_order

Z = agg_all.pivot(index="rho", columns="rating", values="UL_rho").reindex(index=rho_vals, columns=ratings).fillna(0.0).values
X, Y = np.meshgrid(np.arange(len(ratings)), np.array(rho_vals))

# 7) 画 3D surface
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)

ax.set_title("Aggregate Unexpected Loss (UL) by Rating and Asset Correlation (2025Q4)")
ax.set_xlabel("Rating")
ax.set_ylabel("rho")
ax.set_zlabel("Aggregate UL")

ax.set_xticks(np.arange(len(ratings)))
ax.set_xticklabels(ratings)

plt.tight_layout()
plt.savefig("outputs/Q10_3D_aggregate_UL.png", dpi=200)
print("\n=== Q10 saved figure: Q10_3D_aggregate_UL.png ===")