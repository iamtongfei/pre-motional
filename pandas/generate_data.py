"""
generate_data.py
----------------
生成两个 parquet 文件，模拟 AV trajectory prediction 评测的真实数据结构：

1. scenes_manifest.parquet
   每行 = 一个 scene 的 metadata
   columns: scene_id, city, weather, scenario_type, num_agents, duration_s

2. predictions.parquet
   每行 = 一个 (scene_id, agent_id, model) 三元组的预测结果
   columns: scene_id, agent_id, model, ade, fde, miss_rate

跑一次就行：python generate_data.py
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SCENES = 200

# ---------- 1. Manifest ----------
cities = RNG.choice(["boston", "singapore", "pittsburgh"], size=N_SCENES, p=[0.5, 0.3, 0.2])
weathers = RNG.choice(["clear", "rain", "night"], size=N_SCENES, p=[0.6, 0.25, 0.15])
scenarios = RNG.choice(
    ["intersection", "highway", "urban", "parking"],
    size=N_SCENES,
    p=[0.35, 0.2, 0.35, 0.1],
)
num_agents = RNG.integers(3, 25, size=N_SCENES)
duration_s = RNG.uniform(6.0, 20.0, size=N_SCENES).round(1)

manifest = pd.DataFrame(
    {
        "scene_id": [f"scene-{i:04d}" for i in range(N_SCENES)],
        "city": cities,
        "weather": weathers,
        "scenario_type": scenarios,
        "num_agents": num_agents,
        "duration_s": duration_s,
    }
)

# ---------- 2. Predictions ----------
# 每个 scene 选若干 agent,每个 agent 给 3 个 model 各跑一次
MODELS = ["baseline_lstm", "vectornet", "mtr_v2"]
# 模型的 baseline 难度系数(越高越差):lstm 最弱, mtr 最强
MODEL_DIFFICULTY = {"baseline_lstm": 1.4, "vectornet": 1.0, "mtr_v2": 0.75}
# 场景难度系数: rain/night/intersection 更难
WEATHER_FACTOR = {"clear": 1.0, "rain": 1.35, "night": 1.5}
SCENARIO_FACTOR = {"highway": 0.8, "urban": 1.0, "intersection": 1.25, "parking": 0.9}

rows = []
for _, scene in manifest.iterrows():
    # 每个 scene 抽 3-8 个 agent
    n_eval_agents = min(scene["num_agents"], RNG.integers(3, 9))
    weather_f = WEATHER_FACTOR[scene["weather"]]
    scenario_f = SCENARIO_FACTOR[scene["scenario_type"]]

    for agent_idx in range(n_eval_agents):
        for model in MODELS:
            model_f = MODEL_DIFFICULTY[model]
            # ADE around 0.5-3.0 m, FDE 是 ADE 的 ~2x
            base_ade = RNG.gamma(shape=2.0, scale=0.5)
            ade = base_ade * model_f * weather_f * scenario_f
            fde = ade * RNG.uniform(1.8, 2.4)
            miss_rate = float(fde > 2.0)  # >2m 算 miss
            rows.append(
                {
                    "scene_id": scene["scene_id"],
                    "agent_id": f"agent-{agent_idx:02d}",
                    "model": model,
                    "ade": round(ade, 3),
                    "fde": round(fde, 3),
                    "miss_rate": miss_rate,
                }
            )

predictions = pd.DataFrame(rows)

# ---------- 3. Inject some NaN (real datasets always have them) ----------
nan_idx = RNG.choice(predictions.index, size=int(0.02 * len(predictions)), replace=False)
predictions.loc[nan_idx, "fde"] = np.nan

# ---------- 4. Write ----------
manifest.to_parquet("scenes_manifest.parquet", index=False)
predictions.to_parquet("predictions.parquet", index=False)

print(f"[OK] manifest:    {manifest.shape}  -> scenes_manifest.parquet")
print(f"[OK] predictions: {predictions.shape}  -> predictions.parquet")
print("\nmanifest preview:")
print(manifest.head())
print("\npredictions preview:")
print(predictions.head())
