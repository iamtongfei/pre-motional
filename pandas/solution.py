"""
solution.py — 参考答案 (做完 starter 再看!)
============================================
"""
import numpy as np
import pandas as pd


# ============================================================
# Task 1 — Load & Inspect
# ============================================================
manifest = pd.read_parquet("scenes_manifest.parquet")
predictions = pd.read_parquet("predictions.parquet")

print("=" * 60)
print("Task 1: Load & Inspect")
print("=" * 60)
print(f"manifest shape:    {manifest.shape}")
print(f"predictions shape: {predictions.shape}")
print(f"\npredictions dtypes:\n{predictions.dtypes}")
print(f"\nmodel counts:\n{predictions['model'].value_counts()}")
print(f"\nNaN per column:\n{predictions.isna().sum()}")


# ============================================================
# Task 2 — Per-model metrics
# ============================================================
print("\n" + "=" * 60)
print("Task 2: Per-model metrics")
print("=" * 60)

# 简单版
per_model = predictions.groupby("model").agg(
    {"ade": "mean", "fde": "mean", "miss_rate": "mean"}
)
print("\nSimple version:")
print(per_model.round(3))

# 命名聚合 (推荐写法,列名可控)
per_model_detailed = predictions.groupby("model").agg(
    ade_mean=("ade", "mean"),
    ade_std=("ade", "std"),
    fde_mean=("fde", "mean"),
    fde_std=("fde", "std"),
    miss_rate=("miss_rate", "mean"),
    n=("ade", "count"),
)
print("\nNamed aggregation:")
print(per_model_detailed.round(3))


# ============================================================
# Task 3 — Merge
# ============================================================
print("\n" + "=" * 60)
print("Task 3: Merge")
print("=" * 60)

df = predictions.merge(manifest, on="scene_id", how="left")
assert len(df) == len(predictions), "merge 改变了行数!检查 key 是否唯一"
assert df["weather"].isna().sum() == 0, "有 scene 没匹配上"
print(f"merged df shape: {df.shape}")
print(f"new columns: {[c for c in df.columns if c not in predictions.columns]}")


# ============================================================
# Task 4 — Conditional performance
# ============================================================
print("\n" + "=" * 60)
print("Task 4: Conditional performance")
print("=" * 60)

by_model_weather = df.groupby(["model", "weather"]).agg(
    ade=("ade", "mean"),
    fde=("fde", "mean"),
    n=("ade", "count"),
)
print("\nBy (model, weather):")
print(by_model_weather.round(3))

by_model_scenario = df.groupby(["model", "scenario_type"]).agg(
    ade=("ade", "mean"),
    fde=("fde", "mean"),
    n=("ade", "count"),
)
print("\nBy (model, scenario_type):")
print(by_model_scenario.round(3))

# pivot 看 model x weather 一目了然
print("\nPivot (mean ADE):")
print(
    pd.pivot_table(df, values="ade", index="model", columns="weather", aggfunc="mean")
    .round(3)
)


# ============================================================
# Task 5 — Find hard scenes (OOD candidates)
# ============================================================
print("\n" + "=" * 60)
print("Task 5: OOD candidates")
print("=" * 60)

scene_difficulty = df.groupby("scene_id")["ade"].mean().sort_values(ascending=False)
threshold = scene_difficulty.quantile(0.95)
hard_ids = scene_difficulty[scene_difficulty >= threshold].index
hard_with_meta = manifest[manifest["scene_id"].isin(hard_ids)].copy()
hard_with_meta["mean_ade"] = hard_with_meta["scene_id"].map(scene_difficulty)
hard_with_meta = hard_with_meta.sort_values("mean_ade", ascending=False)

print(f"threshold (95th pct mean ADE): {threshold:.3f}")
print(f"# hard scenes: {len(hard_with_meta)}")
print("\nHard scenes weather distribution:")
print(hard_with_meta["weather"].value_counts(normalize=True).round(3))
print("\nAll scenes weather distribution (for comparison):")
print(manifest["weather"].value_counts(normalize=True).round(3))
print(
    "\n-> rain/night 在 hard set 里占比明显高于全集,"
    "这就是 OOD detection 想抓的 distribution shift"
)


# ============================================================
# Task 6 — Save
# ============================================================
print("\n" + "=" * 60)
print("Task 6: Save reports")
print("=" * 60)

per_model_detailed.reset_index().to_parquet("report_per_model.parquet", index=False)
hard_with_meta.to_parquet("ood_candidates.parquet", index=False)
print("[OK] report_per_model.parquet")
print("[OK] ood_candidates.parquet")


# ============================================================
# Bonus B3: relative difficulty using transform
# ============================================================
print("\n" + "=" * 60)
print("Bonus: per-scene normalized ADE (transform demo)")
print("=" * 60)

# transform 关键: 返回的 Series 和原 df 一样长,可以直接赋值回去
df["scene_mean_ade"] = df.groupby("scene_id")["ade"].transform("mean")
df["relative_ade"] = df["ade"] / df["scene_mean_ade"]
print(df[["scene_id", "agent_id", "model", "ade", "scene_mean_ade", "relative_ade"]].head())
print("\n=== Done ===")
