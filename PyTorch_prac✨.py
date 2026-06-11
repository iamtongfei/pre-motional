'''
🔑 几个被很多教程忽略的「最优写法」我特别强调了

from_numpy vs torch.tensor(np.array) —— 前者零拷贝,后者会复制并 warn
.reshape() vs .view() —— 平时用 reshape(自动处理 contiguous),只在性能极致场景用 view
weights_only=True —— PyTorch 2.4+ 加载 checkpoint 的安全默认
take_along_dim vs gather —— 新版 API 语义更清晰,推荐
Hook 用 context manager 包装 —— 防止忘记 remove() 的内存泄漏
masked_fill(~mask, -inf) + softmax —— 处理 attention padding 的标准 idiom
@torch.inference_mode() —— 比 no_grad() 更激进的部署装饰器
'''

"""
================================================================================
PyTorch 实战练习脚本 (Self-Verifying Runnable Script)
场景: 自动驾驶轨迹预测 (Trajectory Prediction)
--------------------------------------------------------------------------------
用法:
  $ python pytorch_practice.py                  # 跑完整脚本看所有 print
  或者复制每个 SECTION 到 IPython 里一段段试

每个 section 格式:
  [一句话最优写法]
  [代码: 最优写法 + print 验证]
  [常见错对比 (commented out)]

所有 section 共用同一份 toy data,前后串联,模拟真实 trajectory pipeline.
================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

torch.manual_seed(42)
np.random.seed(42)

def banner(title: str):
    print(f"\n{'='*68}\n  {title}\n{'='*68}")

def show(name: str, t, *, head=False):
    """统一格式打印 tensor 信息"""
    if isinstance(t, torch.Tensor):
        info = f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"
        if t.requires_grad:
            info += ", requires_grad=True"
        print(f"  {name:30s} {info}")
        if head and t.numel() <= 30:
            print(f"  {' '*30} values={t.flatten().tolist()}")
    else:
        print(f"  {name:30s} {type(t).__name__}: {t}")


# ================================================================
#  SECTION 0: 造统一的 toy scene (后面所有 section 都用这份数据)
# ================================================================
banner("SECTION 0: 统一 Toy Scene (B=2 场景, N=3 agent, T_p=4 历史, T_f=6 未来)")

B, N, T_p, T_f, K, D = 2, 3, 4, 6, 3, 2
#                                    ↑ K=3 模态, D=2 (x,y) 坐标

# ★ 最优:numpy 数据用 from_numpy (零拷贝),指定 dtype 用 .float()
past_np = np.random.randn(B, N, T_p, D).astype(np.float32)
past    = torch.from_numpy(past_np)                    # 共享内存,无拷贝

future_gt = torch.randn(B, N, T_f, D)                  # ground truth
pred      = torch.randn(B, N, K, T_f, D)               # 多模态预测
probs     = torch.randn(B, N, K).softmax(dim=-1)       # 每个 agent 的 K 个模态概率

# 真实场景里 agent 数量不固定,要 padding mask
# 假设场景 0 只有 2 个 agent (第 3 个 是 padding), 场景 1 全部 3 个 agent 都真实
agent_mask = torch.tensor([
    [True, True, False],     # scene 0: agent 2 是 padding
    [True, True, True ],     # scene 1: 都是真的
])  # shape (B, N) bool

# 时间步 mask: 假设场景 0 agent 0 只有 3 帧历史,其余都满 4 帧
valid_mask = torch.ones(B, N, T_p, dtype=torch.bool)
valid_mask[0, 0, 0] = False                            # 把场景 0 agent 0 的第 0 帧标无效

show("past",         past)
show("future_gt",    future_gt)
show("pred",         pred)
show("probs",        probs)
show("agent_mask",   agent_mask)
show("valid_mask",   valid_mask)
# 预期:
#   past         shape=(2, 3, 4, 2), dtype=torch.float32
#   future_gt    shape=(2, 3, 6, 2), dtype=torch.float32
#   pred         shape=(2, 3, 3, 6, 2), dtype=torch.float32
#   probs        shape=(2, 3, 3), dtype=torch.float32
#   agent_mask   shape=(2, 3),    dtype=torch.bool
#   valid_mask   shape=(2, 3, 4), dtype=torch.bool


# ================================================================
#  SECTION 1: Tensor Creation —— 最优 vs 常见错
# ================================================================
banner("SECTION 1: Tensor Creation")

# ★ 最优 1: 从 numpy 用 from_numpy (零拷贝!)
arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
t1 = torch.from_numpy(arr)
print(f"  from_numpy 共享内存吗? {t1.data_ptr() == arr.__array_interface__['data'][0]}")
# True - 修改 arr 会影响 t1

# ✗ 常见错: torch.tensor(np.array(...)) — 会复制 + 给个 warning
# t_bad = torch.tensor(arr)   # 不推荐 (PyTorch 会 warn: 建议 from_numpy 或 clone().detach())

# ★ 最优 2: zeros_like / ones_like / empty_like 跟着别的 tensor 造
buffer = torch.zeros_like(past)                # shape/dtype/device 全跟 past 一致
show("zeros_like(past)", buffer)

# ✗ 不推荐: torch.zeros(*past.shape, dtype=past.dtype, device=past.device)  ← 啰嗦

# ★ 最优 3: 直接造时显式给 dtype/device 避免后面再转
labels = torch.zeros(B, N, dtype=torch.long)          # CrossEntropy 标签
mask   = torch.ones(B, N, dtype=torch.bool)
coords = torch.empty(B, N, 2)                          # 不需要清零时用 empty 更快

show("labels (long)", labels)
show("mask (bool)",   mask)
show("coords",        coords)


# ================================================================
#  SECTION 2: Device Management
# ================================================================
banner("SECTION 2: Device Management")

# ★ 最优:一次性确定 device,后面都用同一个
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  current device: {device}")

# 移动单个 tensor
past_dev = past.to(device, non_blocking=True)   # ★ 配合 DataLoader pin_memory=True 用
show("past_dev", past_dev)

# ★ 最优:整个 batch 是 dict 时,写一个通用搬运函数
def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj

batch = {'past': past, 'future': future_gt, 'agent_mask': agent_mask}
batch = to_device(batch, device)
print(f"  batch['past'].device = {batch['past'].device}")

# ✗ 常见错: 对 tensor 写 t.to(device) 不接住返回值
# past.to(device)         # ← 这行没用!tensor 的 .to() 不是 in-place
# past = past.to(device)  # ★ 必须这样


# ================================================================
#  SECTION 3: Shapes & Views (view vs reshape, contiguous 陷阱)
# ================================================================
banner("SECTION 3: Shapes & Views")

# ★ 最优:加 batch 维用 unsqueeze - 多加一个带1的维度
single = torch.randn(N, T_p, D)                # (3, 4, 2)
single_batch = single.unsqueeze(0)             # (1, 3, 4, 2)
show("unsqueeze(0)", single_batch)

# ★ 最优:压扁多余的 dim 用 squeeze, 去掉所有带1的维度
x = torch.randn(B, N, 1, T_f, D)               # (2, 3, 1, 6, 2)
x_sq = x.squeeze(2)                            # (2, 3, 6, 2)
show("squeeze(2)", x_sq)

# ★ 最优:agent 维和 batch 维合并送进 per-agent encoder
flat = past.reshape(B * N, T_p, D)             # (6, 4, 2) - 每个 agent 当独立样本
show("reshape (B,N→B*N)", flat)
# 还原
restored = flat.reshape(B, N, T_p, D)
print(f"  还原后 == past?  {torch.equal(restored, past)}")  # True

# ⚠ Contiguous 陷阱: permute 后用 view 会报错
y = past.permute(0, 2, 1, 3)                   # (B, T_p, N, D), 非 contiguous
print(f"  permute 后 is_contiguous? {y.is_contiguous()}")  # False

# ✗ 错: y.view(B, T_p, -1)  → RuntimeError
# ★ 最优: 两种解法
y_fix1 = y.reshape(B, T_p, -1)                 # reshape 自动处理 (可能拷贝)
y_fix2 = y.contiguous().view(B, T_p, -1)       # 显式拷贝再 view (语义更明确)
print(f"  reshape 结果 shape:    {tuple(y_fix1.shape)}")
print(f"  contiguous+view shape: {tuple(y_fix2.shape)}")

# 速查表:
#   .unsqueeze(dim)     加一个长度=1 的维度
#   .squeeze(dim)       去掉长度=1 的维度
#   .reshape(...)       通用 reshape,自动处理 non-contiguous (★ 平时用这个)
#   .view(...)          要求 contiguous,但语义更明确 / 速度略快
#   .permute(*dims)     任意维度重排 (不复制,变 non-contiguous)
#   .transpose(d1, d2)  交换两维 (不复制,变 non-contiguous)
#   .contiguous()       强制连续 (复制一份)


# ================================================================
#  SECTION 4: Tensor Operations (cat / stack / broadcast / einsum)
# ================================================================
banner("SECTION 4: Tensor Operations")

# ─── cat vs stack ───────────────────────────────────────
a = torch.zeros(B, N, T_p, D)
b = torch.ones(B, N, T_f, D)

# ★ cat: 沿已有维度拼,维度数不变
full = torch.cat([a, b], dim=2)                # (B, N, T_p+T_f, D) = (2, 3, 10, 2)
show("cat([past, future])", full)

# ★ stack: 新建一个维度堆叠
modes = [torch.randn(B, N, T_f, D) for _ in range(K)]      # K 个 (B, N, T_f, D)
stacked = torch.stack(modes, dim=2)            # (B, N, K, T_f, D) = (2, 3, 3, 6, 2)
show("stack K modes", stacked)

# ─── Broadcasting ────────────────────────────────────────
# ★ 用 unsqueeze 显式扩维,可读性 > 隐式
ego = torch.tensor([5.0, 5.0])                 # 自车位置 (D,) = (2,)
# 想做 past - ego, ego 必须能 broadcast 到 (B, N, T_p, D)
ego_expanded = ego.view(1, 1, 1, D)            # 显式 (1, 1, 1, 2)
relative = past - ego_expanded                 # broadcast → (B, N, T_p, D)
show("past - ego", relative)

# Broadcasting 三规则:从右往左对齐,要么相等 / 要么其中一个=1 / 要么不存在
# (B, N, T, D) - (1, 1, 1, D) ✓
# (B, N, T, D) - (D,)         ✓ (左侧自动补 1)
# (B, N, T, D) - (T, D)       ✓ (左侧补到 (1,1,T,D))

# ─── einsum 神器 ─────────────────────────────────────────
# 任务:把每个 scene 的所有 agent 轨迹绕原点旋转 (每个 scene 不同旋转矩阵)
R = torch.tensor([
    [[1.0, 0.0], [0.0, 1.0]],                  # scene 0: 不转
    [[0.0, -1.0], [1.0, 0.0]],                 # scene 1: 转 90°
])                                              # (B, 2, 2)

# 不用 einsum: past @ R^T  + 各种 broadcast → 容易写错
# ★ einsum 一行搞定
rotated = torch.einsum('bij,bntj->bnti', R, past)   # (B, N, T_p, D)
show("rotated", rotated)
# 解读:
#   R:    'bij'    → (B, 2, 2)
#   past: 'bntj'   → (B, N, T_p, 2)
#   out:  'bnti'   → (B, N, T_p, 2)  共享 b,被求和的 j 消失

# 验证: scene 1 应该把 (x, y) 变成 (-y, x)
print(f"  原 past[1,0,0]:     {past[1,0,0].tolist()}")
print(f"  旋转后 rotated[1,0,0]: {rotated[1,0,0].tolist()}  ← 应该是 (-y, x)")


# ================================================================
#  SECTION 5: Reduction —— ADE / FDE / minADE_K 实战
# ================================================================
banner("SECTION 5: Reduction (ADE / FDE)")

# ★ 关键心法:dim= 决定"沿哪一维压扁",keepdim 保留维度方便后续 broadcast

# ─── ADE (Average Displacement Error) ───────────────────
# 单模态 (取 K=0 的预测来对比)
pred_single = pred[:, :, 0, :, :]              # (B, N, T_f, D)
err = (pred_single - future_gt).norm(dim=-1)   # ★ norm 沿 D=2 压扁,得 (B, N, T_f) 距离
show("err (L2 per step)", err)

ade_per_agent = err.mean(dim=-1)               # (B, N) - 每个 agent 的平均位移误差
show("ade_per_agent", ade_per_agent)

# ─── FDE (Final Displacement Error) ─────────────────────
fde_per_agent = (pred_single[..., -1, :] - future_gt[..., -1, :]).norm(dim=-1)   # (B, N)
show("fde_per_agent", fde_per_agent)

# ─── minADE_K: 多模态选 best ────────────────────────────
# pred: (B, N, K, T_f, D),  future_gt: (B, N, T_f, D)
gt_expanded = future_gt.unsqueeze(2)            # (B, N, 1, T_f, D)
err_k = (pred - gt_expanded).norm(dim=-1)       # (B, N, K, T_f)
ade_k = err_k.mean(dim=-1)                      # (B, N, K)
min_ade, best_mode_idx = ade_k.min(dim=-1)      # (B, N), (B, N)
show("min_ade (B, N)", min_ade)
show("best_mode_idx", best_mode_idx, head=True)
# 解读:best_mode_idx[b,n] = 第 b 个 scene 第 n 个 agent 最准的模态编号

# ★ 全局 minADE_K 标量
overall_minade = min_ade.mean()
print(f"  Overall minADE_K = {overall_minade.item():.4f}")

# 速查:
#   .sum(dim=-1)              沿最后一维求和,出 (..., )
#   .sum(dim=-1, keepdim=True) 保留维度,出 (..., 1) — 适合后续 broadcast
#   .mean(dim=(2,3))          沿多个维度求平均
#   .norm(dim=-1)             L2 范数(等价 (x**2).sum(dim=-1).sqrt())
#   .max(dim) / .min(dim)     返回 (values, indices) tuple


# ================================================================
#  SECTION 6: Masking (Padding Mask 实战)
# ================================================================
banner("SECTION 6: Masking")

# 真实场景常见:有些 agent 是 padding,loss 不能算它们
# ★ 最优:用 bool mask + 算术 (向量化,GPU 友好)

def masked_mean(values, mask):
    """
    values: (..., D)
    mask:   (...,) bool
    返回:  values 在 mask=True 位置的平均
    """
    masked = values * mask.float().unsqueeze(-1)    # 用 unsqueeze 对齐到 values 末尾
    return masked.sum(dim=-2) / mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
    #     ↑ 沿被 mask 的维度求和         ↑ clamp 防除零

# 实战:masked ADE,只算真实 agent
err_per_agent = err.mean(dim=-1)                   # (B, N) 每个 agent 的 ADE
# ★ 屏蔽掉 padding agent
err_masked = err_per_agent * agent_mask.float()
ade_real = err_masked.sum() / agent_mask.sum().clamp(min=1)
print(f"  Naive ADE (含 padding): {err_per_agent.mean().item():.4f}")
print(f"  Masked ADE (排除 padding): {ade_real.item():.4f}")

# ─── masked_fill: 给 padding 位置填 -inf,配合 softmax ───
scores = torch.randn(B, N, N)                       # 假设这是 attention logits
# 想屏蔽掉 key 是 padding 的位置
# agent_mask: (B, N), True=真实
key_mask = ~agent_mask.unsqueeze(1)                 # (B, 1, N), True=要屏蔽
scores_masked = scores.masked_fill(key_mask, float('-inf'))
attn = scores_masked.softmax(dim=-1)                # padding 位置自动变 0
print(f"  attn[0] (scene 0, agent 2 是 padding):")
print(f"  {attn[0]}")
# 预期: 每行第 2 列 (对应 padding agent) 都是 0

# ─── torch.where: 三元运算 ────────────────────────────────
# 想法: 大于阈值的位置取 x, 否则取 y
threshold = 1.0
clipped = torch.where(scores > threshold, scores, torch.zeros_like(scores))
show("clipped", clipped)


# ================================================================
#  SECTION 7: argmax / topk / gather (多模态预测的核心)
# ================================================================
banner("SECTION 7: argmax / topk / gather")

# ─── argmax: 取最大值的 index ─────────────────────────
best_mode = probs.argmax(dim=-1)                    # (B, N) — 概率最大的模态
show("best_mode (argmax)", best_mode, head=True)

# ─── topk: 取前 k 个 ──────────────────────────────────
top_probs, top_idx = probs.topk(2, dim=-1)          # 前 2 个,值 + 索引
show("top_probs", top_probs, head=True)
show("top_idx",   top_idx,   head=True)

# ─── gather: 用 index 取出对应元素 (multi-modal 必杀技) ──
# 任务: pred (B, N, K, T_f, D),用 best_mode (B, N) 取出对应模态的轨迹
# 期望结果 shape: (B, N, T_f, D)

# ★ 最优方法 1: gather (经典)
# best_mode 现在是 (B, N),要 reshape 到 (B, N, 1, T_f, D) 才能 gather
idx = best_mode.view(B, N, 1, 1, 1).expand(B, N, 1, T_f, D)   # (B, N, 1, T_f, D)
best_pred = pred.gather(dim=2, index=idx).squeeze(2)          # (B, N, T_f, D)
show("best_pred (via gather)", best_pred)

# ★ 最优方法 2: take_along_dim (新版 API,语义更清晰,推荐 PyTorch ≥ 1.10)
idx2 = best_mode[..., None, None, None].expand(B, N, 1, T_f, D)
best_pred2 = pred.take_along_dim(idx2, dim=2).squeeze(2)
print(f"  两种方法结果一致? {torch.equal(best_pred, best_pred2)}")

# gather 读法:
#   out[b][n][0][t][d] = pred[b][n][ index[b][n][0][t][d] ][t][d]
#   index 在 dim=2 上指明要取的位置,其他维度照常


# ================================================================
#  SECTION 8: dtype 管理
# ================================================================
banner("SECTION 8: dtype 管理")

# ★ 黄金法则:label 用 long, mask 用 bool, coords 用 float32

# 常见坑 1: numpy bool 数组转过来变 uint8
mask_uint8 = torch.tensor([1, 0, 1], dtype=torch.uint8)
print(f"  uint8 mask 取反: {(~mask_uint8).tolist()}  ← 是按位取反不是逻辑取反!")
# tensor([-2, -1, -2]) - 不是 [0, 1, 0]!
mask_bool = mask_uint8.bool()
print(f"  bool  mask 取反: {(~mask_bool).tolist()}  ← 这才对")

# 常见坑 2: CrossEntropyLoss 报 "expected long"
logits = torch.randn(4, 10)
labels_float = torch.tensor([1.0, 2.0, 3.0, 4.0])    # ✗ float 不行
labels_long  = labels_float.long()                    # ★ 转 long
loss = F.cross_entropy(logits, labels_long)
print(f"  cross_entropy with long labels: {loss.item():.4f}")

# 常见坑 3: 混合精度训练前要确认 dtype
print(f"  past.dtype: {past.dtype}")
print(f"  past.half().dtype: {past.half().dtype}  ← AMP 用这个")

# 速查:
#   .float()  → torch.float32  (默认浮点)
#   .half()   → torch.float16  (AMP / FP16)
#   .double() → torch.float64
#   .long()   → torch.int64    (label 用)
#   .int()    → torch.int32
#   .bool()   → torch.bool     (mask 用)


# ================================================================
#  SECTION 9: 变长序列 (pad_sequence + mask)
# ================================================================
banner("SECTION 9: 变长序列处理")

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# 真实场景:3 个 agent 历史长度分别 [3, 7, 5]
trajs = [
    torch.randn(3, D),
    torch.randn(7, D),
    torch.randn(5, D),
]
lengths = torch.tensor([3, 7, 5])

# ★ 最优:pad_sequence 一行搞定
padded = pad_sequence(trajs, batch_first=True)      # (3, 7, 2) — pad 到最长
show("padded", padded)

# ★ 同步生成 mask
T_max = padded.size(1)
mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)
show("mask", mask, head=True)
# 预期 mask[0] = [T, T, T, F, F, F, F]    (长度 3)
#      mask[1] = [T, T, T, T, T, T, T]    (长度 7)
#      mask[2] = [T, T, T, T, T, F, F]    (长度 5)

# ─── pack/pad 配合 LSTM ──────────────────────────────────
lstm = nn.LSTM(input_size=D, hidden_size=8, batch_first=True)
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
out_packed, (h, c) = lstm(packed)
out_padded, out_lengths = pad_packed_sequence(out_packed, batch_first=True)
show("lstm out_padded", out_padded)
show("h (last hidden)", h)
# 优势:LSTM 只在有效位置算,padding 不参与计算图


# ================================================================
#  SECTION 10: .detach() / .cpu() / .numpy() 链
# ================================================================
banner("SECTION 10: tensor → numpy 标准链")

# 模拟一个带 grad 的 tensor
x = torch.randn(2, 3, requires_grad=True)
y = (x ** 2).sum()
print(f"  y.requires_grad: {y.requires_grad}")

# ✗ 直接转 numpy 会报错
try:
    arr = y.numpy()
except Exception as e:
    print(f"  ✗ y.numpy() 报错: {type(e).__name__}: {e}")

# ★ 最优链条:detach → cpu → numpy
arr = y.detach().cpu().numpy()
print(f"  ★ y.detach().cpu().numpy(): {arr}  type={type(arr).__name__}")

# ★ 0-d tensor 用 .item()
print(f"  ★ y.item(): {y.item():.4f}")

# 速查:
#   tensor → python scalar    .item()           (仅 0-d)
#   tensor → python list      .tolist()         (任意维)
#   tensor → numpy            .detach().cpu().numpy()
#   training loss logging     running_loss += loss.item()   ← 不写 .item() 会显存爆炸


# ================================================================
#  SECTION 11: Autograd 基础
# ================================================================
banner("SECTION 11: Autograd")

# 模型参数自动 requires_grad=True,你通常不用手动管
# 手动管的情况:对输入算梯度 (saliency / adversarial / RND analyzer)

x = past.clone().requires_grad_(True)              # ★ in-place 开启
print(f"  x.requires_grad: {x.requires_grad}")

# 简单 forward: 计算每个 agent 轨迹的 L2 总长度
score = x.pow(2).sum()                              # 标量
score.backward()
show("x.grad (输入梯度)", x.grad)
# x.grad 形状和 x 一样,可以看哪些时间步对 score 影响大

# 关键概念: .grad 是累加的
x.grad.zero_()                                      # ★ 用前清零
y = x.sum()
y.backward()
print(f"  二次 backward 后 x.grad[0,0,0]: {x.grad[0,0,0].tolist()}")
# 因为 zero_() 了,这次只反映 y 的梯度,不会和上次累加


# ================================================================
#  SECTION 12: no_grad / inference_mode / eval()
# ================================================================
banner("SECTION 12: no_grad / inference_mode")

# 三个 idiom,功能不同:
#   model.eval()           切换 Dropout/BN 到推理模式 (影响数值)
#   with torch.no_grad():  关 autograd (省显存,不影响数值)
#   with torch.inference_mode(): 比 no_grad 更激进,但限制更严

# ★ 标准推理 idiom:三件套
def standard_eval(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)

# ★ 部署 idiom:用 inference_mode (PyTorch ≥ 1.9)
@torch.inference_mode()
def deployed_predict(model, x):
    return model(x)

# 验证 no_grad 关 autograd
with torch.no_grad():
    z = past.clone().requires_grad_(True)           # 这里设置 requires_grad 也没用
    w = z.sum()
    print(f"  no_grad 内,w.requires_grad: {w.requires_grad}")   # False!

w_outside = past.clone().requires_grad_(True).sum()
print(f"  no_grad 外,w.requires_grad: {w_outside.requires_grad}") # True


# ================================================================
#  SECTION 13: Model Save / Load
# ================================================================
banner("SECTION 13: Save / Load")

# 造一个 mini 模型 (encoder + head),后面 hook 和 RND section 都会用
class TinyEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden=8):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        # x: (..., in_dim)
        return F.relu(self.lin2(F.relu(self.lin1(x))))


class TrajPredictor(nn.Module):
    def __init__(self, hidden=8, T_f=6, D=2):
        super().__init__()
        self.encoder = TinyEncoder(in_dim=D, hidden=hidden)
        # 把过去 T_p 步全部 encode 后 pool 再预测未来 T_f 步
        self.head = nn.Linear(hidden, T_f * D)
        self.T_f, self.D = T_f, D

    def forward(self, past):
        # past: (B, N, T_p, D)
        feat = self.encoder(past)                   # (B, N, T_p, hidden)
        feat = feat.mean(dim=2)                     # (B, N, hidden)  — 沿时间 pool
        out  = self.head(feat)                      # (B, N, T_f * D)
        return out.view(*out.shape[:-1], self.T_f, self.D)   # (B, N, T_f, D)


model = TrajPredictor()
with torch.no_grad():
    pred_out = model(past)
show("model 输出", pred_out)

# ★ 最优保存:只存 state_dict (权重),不存整个模型对象
torch.save(model.state_dict(), '/tmp/traj_pred.pth')

# ★ 最优加载:先建结构 → load_state_dict → to(device) → eval()
loaded = TrajPredictor()
state = torch.load('/tmp/traj_pred.pth', map_location='cpu', weights_only=True)
loaded.load_state_dict(state)
loaded.to(device)
loaded.eval()

# 验证两个模型一致
with torch.no_grad():
    a = model(past)
    b = loaded(past)
print(f"  原模型 vs 加载模型输出一致? {torch.allclose(a, b)}")

# ★ 部分加载 (fine-tune 场景):strict=False
state_subset = {k: v for k, v in state.items() if 'encoder' in k}
fresh = TrajPredictor()
missing, unexpected = fresh.load_state_dict(state_subset, strict=False)
print(f"  缺失 keys (head 没加载): {missing}")
print(f"  多余 keys: {unexpected}")


# ================================================================
#  SECTION 14: Forward Hooks (Analyzer 工作的核心)
# ================================================================
banner("SECTION 14: Forward Hooks")

# ─── 用法 1: 抓中间特征 (你做 RND analyzer 必用) ──────
features = {}

def make_save_hook(name):
    def hook(module, inp, out):
        features[name] = out.detach()               # ★ 必须 detach!不然 leak 计算图
    return hook

# 给 encoder 挂 hook,抓它的输出
handle = model.encoder.register_forward_hook(make_save_hook('encoder_out'))

# 正常 forward,hook 自动触发
_ = model(past)
show("捕获的 encoder 输出", features['encoder_out'])

# ★ 用完一定 remove,否则后面所有 forward 都被它干扰
handle.remove()

# ─── 用法 2: 修改输出 (encoder ablation) ──────────────
def ablate_hook(module, inp, out):
    """把 encoder 输出的某个隐藏单元置 0,看影响"""
    out = out.clone()                               # ★ 不要直接改 out (in-place 危险)
    out[..., 0] = 0                                 # 第 0 个 hidden 维置零
    return out                                      # 返回修改后的 → 替换原 output

handle = model.encoder.register_forward_hook(ablate_hook)
with torch.no_grad():
    out_ablated = model(past)
    out_normal  = model.eval()(past)
handle.remove()                                     # ★ 用完清掉

# 验证消融有效:ablated ≠ normal
# 注意:remove 后 normal 已经是正常的,但因为消融时 hook 还在,out_ablated 是消融过的
print(f"  消融后输出有差异? {not torch.allclose(out_ablated, out_normal)}")

# ─── 用法 3: 一次性抓所有层的激活 (debug 神技) ──────
class ActivationProbe:
    """用 context manager 包装,自动 remove hooks"""
    def __init__(self, model, layer_types=(nn.Linear,)):
        self.model = model
        self.types = layer_types
        self.handles = []
        self.activations = {}

    def __enter__(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, self.types):
                h = mod.register_forward_hook(
                    lambda m, i, o, n=name: self.activations.update({n: o.detach()})
                )
                self.handles.append(h)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()

with ActivationProbe(model) as probe:
    _ = model(past)
print(f"  抓到的层: {list(probe.activations.keys())}")
for name, act in probe.activations.items():
    show(f"  {name}", act)


# ================================================================
#  SECTION 15: Dataclass with Tensors
# ================================================================
banner("SECTION 15: Dataclass")

@dataclass
class Scene:
    """一个 batch 的所有 tensor 打包,比 dict 更安全"""
    past:       torch.Tensor    # (B, N, T_p, 2)
    future_gt:  torch.Tensor    # (B, N, T_f, 2)
    agent_mask: torch.Tensor    # (B, N) bool
    valid_mask: torch.Tensor    # (B, N, T_p) bool
    map_feats:  Optional[torch.Tensor] = None

    def __post_init__(self):
        # 自动校验 batch 维一致
        B = self.past.shape[0]
        assert self.future_gt.shape[0] == B,  f"future_gt batch {self.future_gt.shape[0]} != {B}"
        assert self.agent_mask.shape[0] == B
        assert self.valid_mask.shape[0] == B

    def to(self, device):
        """复刻 nn.Module.to() 风格"""
        return Scene(
            past=self.past.to(device),
            future_gt=self.future_gt.to(device),
            agent_mask=self.agent_mask.to(device),
            valid_mask=self.valid_mask.to(device),
            map_feats=None if self.map_feats is None else self.map_feats.to(device),
        )

    @property
    def num_real_agents(self) -> int:
        return int(self.agent_mask.sum().item())

scene = Scene(past=past, future_gt=future_gt, agent_mask=agent_mask, valid_mask=valid_mask)
print(f"  scene.num_real_agents: {scene.num_real_agents}")   # 2 + 3 = 5
scene_on_dev = scene.to(device)
print(f"  scene_on_dev.past.device: {scene_on_dev.past.device}")


# ================================================================
#  SECTION 16: 综合 —— RND OOD Analyzer (把所有概念串起来)
# ================================================================
banner("SECTION 16: 综合 — RND Analyzer")

# RND (Random Network Distillation) 用于 OOD 检测:
#   - 一个随机初始化、永不训练的 target 网络
#   - 一个训练去模仿 target 的 predictor 网络
#   - 在 in-distribution 数据上 predictor 学得好,score ≈ 0
#   - 在 OOD 数据上 predictor 没见过,score 高
class RNDAnalyzer:
    def __init__(self, traj_model: nn.Module, hidden_dim=8, device='cpu'):
        self.device = device
        self.model = traj_model.to(device).eval()           # § 13

        # Target: 随机初始化、冻结
        self.target = TinyEncoder(in_dim=D, hidden=hidden_dim).to(device).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)                          # § 11

        # 用 hook 捕获 traj_model 的 encoder 输出 (作为 predictor 的输出)
        self._captured = None
        self.handle = self.model.encoder.register_forward_hook(self._capture)   # § 14

    def _capture(self, module, inp, out):
        self._captured = out.detach()                        # § 14 (★ detach)

    @torch.inference_mode()                                   # § 12
    def score(self, scene: Scene) -> np.ndarray:              # § 15
        scene = scene.to(self.device)                         # § 2

        # 触发 hook 拿 predictor 特征
        _ = self.model(scene.past)
        predictor_feat = self._captured                        # (B, N, T_p, hidden)

        # Target 网络直接前向 (不需要 hook)
        target_feat = self.target(scene.past)                  # (B, N, T_p, hidden)

        # RND score = ||target - predictor||²,沿 hidden 维平均
        diff = (target_feat - predictor_feat).pow(2)           # § 4
        score = diff.mean(dim=-1)                              # § 5 — (B, N, T_p)

        # 沿时间用 valid_mask 做 masked mean
        score = (score * scene.valid_mask.float()).sum(dim=-1) \
              / scene.valid_mask.float().sum(dim=-1).clamp(min=1)   # § 6 — (B, N)

        # 屏蔽 padding agent (输出 NaN 方便识别)
        score = score.masked_fill(~scene.agent_mask, float('nan'))   # § 6

        return score.detach().cpu().numpy()                    # § 10

    def close(self):
        self.handle.remove()                                   # § 14 (清理)


# 跑起来
analyzer = RNDAnalyzer(model, hidden_dim=8, device=device)
ood_scores = analyzer.score(scene)
print(f"  RND scores shape: {ood_scores.shape}")
print(f"  RND scores:\n{ood_scores}")
print(f"  (注意 scene 0 agent 2 是 padding,应该是 NaN)")
analyzer.close()


# ================================================================
#  END
# ================================================================
banner("ALL SECTIONS DONE")
print("""
回顾这个脚本用到的 16 个核心点 (按 § 编号):
   § 0  统一 toy scene
   § 1  Tensor creation (from_numpy / zeros_like)
   § 2  Device management (.to + non_blocking)
   § 3  Shapes & views (reshape vs view + contiguous)
   § 4  Tensor ops (cat/stack/broadcast/einsum)
   § 5  Reduction (ADE/FDE/minADE_K)
   § 6  Masking (masked_fill, masked mean)
   § 7  argmax / topk / gather
   § 8  dtype 管理
   § 9  变长序列 (pad_sequence + pack)
   § 10 detach → cpu → numpy 链
   § 11 Autograd
   § 12 no_grad / inference_mode
   § 13 save / load (weights_only)
   § 14 Forward hooks (capture + ablate + probe)
   § 15 Dataclass
   § 16 综合: RNDAnalyzer

下一步建议:
  1. 真实跑通: python pytorch_practice.py
  2. 把每个 section 改改 (改 shape / 改 mask / 改 hook 行为),看 print 怎么变
  3. 接入真数据 (nuScenes / Argoverse) 把 toy 替换成真 batch
""")