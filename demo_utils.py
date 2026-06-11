"""demo_utils.py — 演示 utils/debug.py 全部 9 个工具.

跑这个脚本看每个函数的输出长什么样:
    $ python demo_utils.py

依赖结构 (放在同一个目录下):
    your_project/
    ├── utils/
    │   ├── __init__.py
    │   └── debug.py
    └── demo_utils.py     ← 本文件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ★ 这就是你想要的 import 方式
from utils import (
    banner, show, show_dict,
    summarize, compare,
    param_summary, grad_summary,
    timer, seed_all,
)


# ============================================================
#  Demo 1: seed_all (统一随机种子,放脚本最开头)
# ============================================================
banner("Demo 1: seed_all")
seed_all(42)


# ============================================================
#  Demo 2: banner + show (基础)
# ============================================================
banner("Demo 2: show — 基础打印")

# Toy trajectory tensors
B, N, T_p, T_f, D = 2, 3, 4, 6, 2
past       = torch.randn(B, N, T_p, D)
future_gt  = torch.randn(B, N, T_f, D)
agent_mask = torch.tensor([[True, True, False], [True, True, True]])
labels     = torch.tensor([0, 1, 2, 3])

show("past",       past)
show("future_gt",  future_gt)
show("agent_mask", agent_mask)
show("labels",     labels, head=True)    # head=True 时打印实际值
show("nothing",    None)                  # 也能打印 None
show("a list",     [1, 2, 3])             # 也能打印非 tensor


# ============================================================
#  Demo 3: show_dict (一次性看一个 batch)
# ============================================================
banner("Demo 3: show_dict")

batch = {
    'past': past,
    'future_gt': future_gt,
    'agent_mask': agent_mask,
    'scenario_id': 'scene_0042',          # 非 tensor 也能打
}
show_dict(batch)


# ============================================================
#  Demo 4: summarize (数值健康检查)
# ============================================================
banner("Demo 4: summarize — 数值健康")

summarize(past, "past")

# 故意造一个有 NaN 的 tensor 看效果
bad = past.clone()
bad[0, 0, 0, 0] = float('nan')
bad[0, 1, 0, 0] = float('inf')
summarize(bad, "bad (含 NaN/Inf)")


# ============================================================
#  Demo 5: compare (对比两个 tensor)
# ============================================================
banner("Demo 5: compare")

# 完全相同
compare(past, past.clone(), "past", "past_copy")

# 加点噪声
noisy = past + torch.randn_like(past) * 0.01
compare(past, noisy, "past", "noisy", atol=1e-6)

# 完全不同
compare(past, torch.randn_like(past), "past", "random")


# ============================================================
#  Demo 6: param_summary (模型参数统计)
# ============================================================
banner("Demo 6: param_summary")

class TinyTrajPredictor(nn.Module):
    def __init__(self, hidden=16, T_f=6, D=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, T_f * D)
        self.T_f, self.D = T_f, D

    def forward(self, past):
        feat = self.encoder(past).mean(dim=2)             # (B, N, hidden)
        out  = self.head(feat)                            # (B, N, T_f*D)
        return out.view(*out.shape[:-1], self.T_f, self.D)

model = TinyTrajPredictor()
param_summary(model)

# Freeze 一部分 demo
for p in model.encoder.parameters():
    p.requires_grad_(False)
print("\n  Freeze encoder 后:")
param_summary(model, show_layers=True)


# ============================================================
#  Demo 7: timer (计时)
# ============================================================
banner("Demo 7: timer")

# Unfreeze 回来
for p in model.parameters():
    p.requires_grad_(True)

with timer("forward"):
    pred = model(past)

with timer("forward + loss"):
    pred = model(past)
    loss = F.mse_loss(pred, future_gt)


# ============================================================
#  Demo 8: grad_summary (梯度健康检查)
# ============================================================
banner("Demo 8: grad_summary")

# 正常梯度
loss = F.mse_loss(model(past), future_gt)
loss.backward()
print("正常 backward 后:")
grad_summary(model)

# 模拟 NaN grad (人为注入)
model.zero_grad()
loss = F.mse_loss(model(past), future_gt)
loss.backward()
model.head.weight.grad[0, 0] = float('nan')         # 注入一个 NaN
print("\n人为注入 NaN 后:")
grad_summary(model, top_k=3)


# ============================================================
#  Demo 9: 综合 — 一个迷你训练循环用上所有工具
# ============================================================
banner("Demo 9: 综合训练循环 (所有工具串联)")

seed_all(123)

model = TinyTrajPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

param_summary(model)

for step in range(3):
    print(f"\n  --- Step {step} ---")

    optimizer.zero_grad()

    with timer(f"step {step} forward"):
        pred = model(past)

    loss = F.mse_loss(pred, future_gt)
    summarize(loss, f"loss step {step}")

    with timer(f"step {step} backward"):
        loss.backward()

    # 只在某步检查梯度
    if step == 0:
        print()
        grad_summary(model, top_k=3)

    optimizer.step()

banner("Done — 把 utils/ 整个目录复制到你的项目里就能用")
