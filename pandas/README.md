# Pandas × Trajectory Eval — Mini Lab

练熟 pandas 四个核心 API,场景是你每天在 AV 研究里都在做的事:

| pandas API | 你的研究里对应什么 |
|---|---|
| `pd.read_parquet` | 读 nuScenes manifest / 模型 prediction 输出 |
| `df.merge` | 把 scene metadata 拼到 prediction results 上 |
| `df.groupby(...).agg(...)` | 按 (model, weather, scenario) 算 ADE/FDE 切片 |
| `df.to_parquet` | 把 eval report / OOD candidate list 存盘 |

---

## 项目场景

你有 3 个模型在 nuScenes-like 数据集上跑 trajectory prediction:
`baseline_lstm` / `vectornet` / `mtr_v2`

两个 parquet:

**`scenes_manifest.parquet`** — 每行 1 个 scene 的 metadata
```
scene_id | city | weather | scenario_type | num_agents | duration_s
```

**`predictions.parquet`** — 每行 1 个 (scene, agent, model) 的预测结果
```
scene_id | agent_id | model | ade | fde | miss_rate
```

你的工作 = 把这两张表 join 起来,算出 conditional metrics,挑出 hard scenes 当 OOD candidate。

---

## 怎么用

```bash
# 0. (一次性) 装 parquet engine,你的研究环境多半已经有
pip install pandas pyarrow

# 1. 生成数据
python generate_data.py

# 2. 打开 starter.py,照着 6 个 TODO Task 一个个填
#    每个 Task 都有 hint 注释在末尾,实在卡住再看

# 3. 跑参考答案对照
python solution.py
```

---

## Task 路线图(渐进式)

### Task 1 — Load & Inspect ★基础★
`pd.read_parquet` + `.head() / .dtypes / .describe() / .isna().sum() / .value_counts()`
熟悉两张表长啥样,检查 NaN。

### Task 2 — Per-model overall metrics ★groupby 基础★
```python
df.groupby("model").agg(
    ade_mean=("ade", "mean"),
    ade_std=("ade", "std"),
    n=("ade", "count"),
)
```
**重点**:用**命名聚合** (named aggregation) 而不是 `.agg({...})` —— 列名可控,代码 self-documenting。
研究 paper 里 Table 1 通常长这样。

### Task 3 — Merge ★必杀技★
```python
df = predictions.merge(manifest, on="scene_id", how="left")
```
**Sanity check 三连**(养成习惯,merge 翻车很常见):
- `len(df) == len(predictions)` — 没有意外的笛卡尔积膨胀
- `df["weather"].isna().sum() == 0` — 所有 scene 都匹配到了
- merge 前先 `manifest["scene_id"].is_unique`,确保 right 表 key 唯一

`how` 参数记忆口诀:
- `left` — 保住 predictions,manifest 缺就 NaN(默认选这个)
- `inner` — 只留 match 上的
- `outer` — 都留
- `right` — 反过来

### Task 4 — Conditional performance ★最高频★
```python
df.groupby(["model", "weather"]).agg(ade=("ade","mean"))
```
这就是 paper 里 Table 2 的来源:模型在 weather × scenario × city 切片下分别多好/多差。
做完后用 `.unstack()` 或 `pd.pivot_table()` 转成矩阵视图,更直观。

### Task 5 — Find hard scenes ★和你 DC-MMD 直接相关★
1. 每个 scene_id 的 mean ADE → `scene_difficulty`
2. 取 top 5% (`.quantile(0.95)`)
3. join 回 manifest 看这些 hard scene 的 weather 分布
4. 对比全集分布 —— 如果 rain/night 比例显著上升,**这就是 distribution shift 的数据证据**

你 DC-MMD 监控的就是这种 shift,只不过你是 sequential / online 检测,这里是 offline batch 分析。
本质同一件事:在 metric space 里找 outlier。

### Task 6 — Save ★IO★
```python
per_model_detailed.reset_index().to_parquet("report_per_model.parquet", index=False)
```
**坑点**:`groupby` 的结果 index 是 group key,直接 `to_parquet` 会把 index 吞掉(默认 `index=True`)。
研究里 reproducibility 角度,**`reset_index() + index=False` 是更安全的默认**。

---

## Bonus / Stretch

做完 6 个 Task 还有余力:

- **B1 `pivot_table`** — Task 4 用 pivot 重写一遍,体会 long → wide
- **B2 `pd.cut`** — 给 ADE 分桶看 distribution: `pd.cut(df["ade"], bins=[0,1,2,5,np.inf])`
- **B3 `transform`** — `df.groupby("scene_id")["ade"].transform("mean")` 返回和原 df 等长的 Series,直接赋值回去做 per-scene normalization。这是 anomaly score 的常见做法
- **B4 NaN 处理** — `predictions["fde"]` 故意混了 NaN,试试 `.agg(["mean", "count", "size"])` 对比 `count` (非 NaN) vs `size` (全部) 的差别

---

## 核心 mental model

写 pandas 时脑子里一直想这个 split-apply-combine 三段式:

```
原 DataFrame
    │
    ├── split  ───> groupby(key)        # 按 key 切成多个组
    │
    ├── apply  ───> .agg / .transform / .apply  # 每组做事
    │
    └── combine ──> 自动拼回去
```

- `.agg` — **reduces** 每组(N 行 → 1 行),输出小
- `.transform` — **broadcasts** 每组(N 行 → N 行),输出和原 df 等长
- `.apply` — 最灵活也最慢,能不用就不用

记住这三个的区别,90% 的 groupby 场景都不会写错。

---

## 一句话总结

> Parquet 进,Parquet 出,中间 merge + groupby 是 pandas 的灵魂。
> 你的整条 eval pipeline = read 多个 parquet → merge → 多层 groupby → to_parquet。
> 这 4 个 API 练熟,你就有了所有 AV 研究 offline analysis 的基础肌肉记忆。
