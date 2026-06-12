"""
starter.py — Trajectory Eval Pipeline (Mini)
============================================

目标:用 4 个 pandas 核心 API 把这套流程跑通
  pd.read_parquet  ->  df.merge  ->  df.groupby().agg()  ->  df.to_parquet

跑之前先:
    python generate_data.py

然后照着 TODO 一个个填。每个 task 末尾有 hint。
不会做就先把 hint 注释取消看一眼,但尽量自己写。
"""

import numpy as np
import pandas as pd


# ============================================================
# Task 1 — Load & Inspect (pd.read_parquet)
# ============================================================
# 读两个 parquet,打印 shape / dtypes / head / describe
# 重点检查:
#   - manifest 有多少 scene
#   - predictions 有没有 NaN
#   - 每个 model 有多少行 (用 value_counts)

# TODO 1.1: 读 scenes_manifest.parquet 和 predictions.parquet
manifest = ...        # <-- 改这里
predictions = ...     # <-- 改这里

# TODO 1.2: 打印两个 df 的 shape 和 dtypes
# TODO 1.3: predictions["model"].value_counts() 看每个模型多少行
# TODO 1.4: 检查 NaN: predictions.isna().sum()

# hint:
# manifest = pd.read_parquet("scenes_manifest.parquet")
# predictions = pd.read_parquet("predictions.parquet")


# ============================================================
# Task 2 — Per-model overall metrics (groupby + agg, 基础版)
# ============================================================
# 算每个 model 的总体平均 ADE / FDE / miss_rate
# 期待输出 (大致):
#                     ade      fde   miss_rate
#   model
#   baseline_lstm    1.95     4.10     0.62
#   mtr_v2           1.05     2.20     0.32
#   vectornet        1.40     2.95     0.45

# TODO 2.1: 用 groupby("model").agg(...) 写出来
per_model = ...  # <-- 改这里

# 进阶 — 用命名聚合 (named aggregation),同时拿 mean / std / count:
#   df.groupby("model").agg(
#       ade_mean=("ade", "mean"),
#       ade_std=("ade", "std"),
#       n=("ade", "count"),
#   )
# TODO 2.2: 写一个 per_model_detailed,带 mean / std / count

per_model_detailed = ...  # <-- 改这里


# ============================================================
# Task 3 — Join manifest with predictions (merge)
# ============================================================
# predictions 没有 weather / scenario_type / city 这些信息
# 要回答"模型在 rain 比 clear 差多少"必须先 merge

# TODO 3.1: predictions LEFT JOIN manifest on scene_id
df = ...  # <-- 改这里

# Sanity check:
# - merge 后行数应该 == len(predictions)
# - df 现在应该有 weather / scenario_type / city / num_agents 这几列
# - 没有 scene_id 应该匹配不上(全部 left join 成功)
# TODO 3.2: assert len(df) == len(predictions)
# TODO 3.3: assert df["weather"].isna().sum() == 0

# hint:
# df = predictions.merge(manifest, on="scene_id", how="left")


# ============================================================
# Task 4 — Conditional performance (multi-key groupby) ★核心★
# ============================================================
# 这是 AV 研究里最常用的一步:模型在 (weather, scenario) 切片下的表现
# 用来发现模型的弱区(对应你 OOD detection 的动机)

# TODO 4.1: 按 (model, weather) 分组,算 mean ADE / mean FDE
by_model_weather = ...  # <-- 改这里
# print(by_model_weather)
#
# 看看 rain 和 night 的 ADE 是不是明显比 clear 高?
# 看看 mtr_v2 在 night 的恶化幅度比 lstm 小还是大?

# TODO 4.2: 按 (model, scenario_type) 分组同样算
by_model_scenario = ...  # <-- 改这里

# 进阶可视化(可选):
# by_model_weather["ade"].unstack("weather").plot.bar()


# ============================================================
# Task 5 — Find hard scenes (OOD candidates) ★和 DC-MMD 相关★
# ============================================================
# 思路:
#   1) 对每个 scene_id,取所有 model 的 ADE 平均 (= scene_difficulty)
#   2) 排序取 top 5% 当 OOD candidate
#   3) 把这些 scene 的 metadata join 回来看分布(是不是都集中在 rain/night?)

# TODO 5.1: 按 scene_id 分组,算每个 scene 的 mean_ade across all models & agents
scene_difficulty = ...   # <-- 改这里
# 应该长这样: Series 或 DataFrame, index = scene_id, value = mean_ade

# TODO 5.2: 选出 top 5% 最难的 scene
threshold = ...   # <-- scene_difficulty 的 95-th percentile
hard_scenes = ...  # <-- scene_id 列表 (或 DataFrame)

# TODO 5.3: 把 hard_scenes 和 manifest merge 回来,看 weather 分布
# 如果你的 data generator 没改,应该会看到 rain/night 占比远高于全集
hard_with_meta = ...  # <-- 改这里
# print(hard_with_meta["weather"].value_counts(normalize=True))
# print(manifest["weather"].value_counts(normalize=True))   # 作对比

# hint:
# scene_difficulty = df.groupby("scene_id")["ade"].mean()
# threshold = scene_difficulty.quantile(0.95)
# hard_ids = scene_difficulty[scene_difficulty >= threshold].index
# hard_with_meta = manifest[manifest["scene_id"].isin(hard_ids)]


# ============================================================
# Task 6 — Save outputs (to_parquet)
# ============================================================
# 把两个 report 存成 parquet,模拟你跑完一个 experiment 后存 result 的动作

# TODO 6.1: per_model_detailed 存成 report_per_model.parquet
# TODO 6.2: hard_with_meta 存成 ood_candidates.parquet

# 注意:groupby 的结果 index 是 model,存 parquet 前可能要 reset_index()
# per_model_detailed.reset_index().to_parquet("report_per_model.parquet", index=False)


# ============================================================
# Bonus — Stretch goals (做完上面再来)
# ============================================================
# B1: pivot_table  画一张 "model x weather" 的 mean ADE 表
#     pd.pivot_table(df, values="ade", index="model", columns="weather", aggfunc="mean")
#
# B2: 给 ADE 分桶 (binning)  用 pd.cut(df["ade"], bins=[0,1,2,5,np.inf])
#     再 groupby bin 看每个区间多少 agent  这就是 ADE distribution
#
# B3: 用 .transform("mean") 做归一化  每个 agent 的 ADE 除以它所在 scene 的 mean ADE
#     得到 "relative difficulty"  这是 anomaly detection 里常用的 score
#
# B4: predictions 里 fde 有 NaN  你的 groupby 默认怎么处理的?
#     试试 .agg(["mean", "count", "size"])  对比 count vs size 体会差别
