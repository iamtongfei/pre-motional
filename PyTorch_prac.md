# PyTorch 核心 API 练习手册
## 场景:自动驾驶轨迹预测 + RND / Analyzer

> 把你列的 10 项 + 5 个补充项,全部用**轨迹预测**这一个场景串起来。
> 每一项格式:**概念** → **语法** → **轨迹场景例子** → **练习题**。

---

## 🔍 Part 0:你的清单覆盖度评估

| 你列的 | 评价 | 备注 |
|---|---|---|
| Tensor creation | ✅ 够 | |
| Device management | ✅ 够 | 加一个 `non_blocking=True` 会更好 |
| Autograd basics | ✅ 够 | RND 推理时其实不用算 grad,但理解很重要 |
| No-grad context | ✅ 够 | 别忘 `inference_mode()` 更省内存 |
| Model loading | ✅ 够 | `weights_only=True` 是新趋势 |
| Forward hooks | ✅ 够 | 但要细分 `register_forward_hook` vs `register_forward_pre_hook` |
| Tensor operations (cat/stack/einsum/broadcasting) | ✅ 够 | |
| Shapes (view/reshape/unsqueeze/squeeze) | ✅ 够 | 缺 `permute / transpose / contiguous` |
| Masking | ✅ 够 | 缺**padding mask**(轨迹任务里最常见) |
| Dataclasses with tensors | ✅ 够 | |

### 🚨 缺的 5 个(你做 trajectory / RND 一定会用)

| 缺什么 | 为什么必须会 |
|---|---|
| **Reduction 操作 `.mean(dim=)` / `.sum(dim=)` / `.norm(dim=)`** | 算 ADE/FDE 全靠它 |
| **`.detach()` / `.cpu()` / `.numpy()` 链** | 把 tensor 丢出计算图、转回 numpy 画图,日常高频 |
| **`torch.argmax` / `torch.topk` / `torch.gather`** | 多模态预测选 best-of-K 必用 |
| **dtype 管理 (`.float()`, `.long()`, `.bool()`)** | label / mask / coords 经常 dtype 不匹配报错 |
| **变长序列处理 (`pad_sequence`, `pack_padded_sequence`)** | 场景里 agent 数量、历史长度都不一样,padding 是核心问题 |

下面把 15 项全部讲完。

---

## 📐 统一场景:轨迹预测的张量约定

固定记住几个 shape,后面所有例子都基于这套:

```
─────────────────────────────────────────────────
 名字                Shape                  含义
─────────────────────────────────────────────────
 past               (B, N, T_p, 2)         历史轨迹
                                            B=batch, N=agents, T_p=past steps, 2=(x,y)
 future_gt          (B, N, T_f, 2)         未来真值
 pred               (B, N, K, T_f, 2)      K 模态预测
 pred_probs         (B, N, K)              K 个模态的概率
 valid_mask         (B, N, T_p) bool       每个时间步是否有效
 agent_mask         (B, N) bool            哪些 agent 槽位是真实的 (剩下是 padding)
─────────────────────────────────────────────────
 典型数值: B=32, N=64, T_p=8 (0.8s @ 10Hz), T_f=12 (1.2s @ 10Hz), K=6
```

---

## 1️⃣ Tensor Creation

**一句话**:`torch.tensor` 从已有数据复制,`torch.from_numpy` 共享内存(零拷贝),`torch.zeros/ones/empty/randn` 直接造。

```python
import torch
import numpy as np

# ① 从 Python list / scalar
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])         # shape (2, 2), dtype float32

# ② 从 numpy (零拷贝,⚠️ 修改 a 会改 npy)
npy = np.random.randn(32, 64, 8, 2).astype(np.float32)
past = torch.from_numpy(npy)                       # shape (32, 64, 8, 2)

# ③ 直接造
zero_mask = torch.zeros(32, 64, dtype=torch.bool)  # padding mask 初值
pred = torch.randn(32, 64, 6, 12, 2)               # 模拟随机预测

# ④ 跟着已有 tensor 的属性造(超常用)
new_pred = torch.zeros_like(past)                  # shape/dtype/device 全部跟 past
```

**轨迹场景例子**:你 dataloader 经常返回 numpy,batch 起来要转 tensor。
```python
def collate_fn(batch):
    past_np = np.stack([item['past'] for item in batch])  # (B, N, T_p, 2)
    return {
        'past': torch.from_numpy(past_np).float(),
        'mask': torch.from_numpy(np.stack([item['mask'] for item in batch])).bool()
    }
```

**🏋️ 练习**:造一个 (4, 8, 8, 2) 的过去轨迹张量,初值是 0,然后把 `[:, :, -1, :]` (最后一个时间步) 填成 1。
<details><summary>答案</summary>

```python
past = torch.zeros(4, 8, 8, 2)
past[:, :, -1, :] = 1.0   # 用切片赋值
```
</details>

---

## 2️⃣ Device Management

**一句话**:模型和数据**必须在同一个 device**,否则报错;搬运用 `.to(device)`。

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型
model = TrajPredictor().to(device)

# 数据 (⚠️ tensor 的 .to() 不是 in-place,必须接住!)
past = past.to(device)                              # 普通搬运
past = past.to(device, non_blocking=True)           # ★ DataLoader 配 pin_memory=True 时更快

# 检查 device
print(past.device)                                  # device(type='cuda', index=0)
```

**关于 `non_blocking=True`**:配合 DataLoader 的 `pin_memory=True`,可以让 CPU→GPU 传输异步进行,流水线效率更高。

**轨迹场景常见坑**:多 agent 数据有时 dataloader 给的是 dict,要逐项搬:
```python
batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
```

**🏋️ 练习**:写一个函数把任意嵌套 dict 里的所有 tensor 搬到 device。
<details><summary>答案</summary>

```python
def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj
```
</details>

---

## 3️⃣ Autograd Basics

**一句话**:`requires_grad=True` 的 tensor 参与运算时会建计算图,`.backward()` 反向填 `.grad`。

```python
# 通常你不需要手动开 requires_grad —— 模型参数自动是 True
# 你需要手动开的情况:对输入算梯度 (adversarial, saliency, RND...)

past = past.requires_grad_(True)            # in-place 开启
out = model(past)                           # 前向,建计算图
loss = out.sum()
loss.backward()                             # 反向

print(past.grad.shape)                      # (B, N, T_p, 2) — 输入梯度
```

**轨迹场景例子(RND-flavored)**:对输入的轨迹算梯度敏感度,定位"哪些时间点最重要"。
```python
past.requires_grad_(True)
embedding = encoder(past)                   # (B, N, D)
score = (embedding ** 2).sum(dim=-1)        # (B, N) 每个 agent 一个 score

# 看 score 对每个时间步的敏感度
score.sum().backward()
sensitivity = past.grad.norm(dim=-1)        # (B, N, T_p)
```

**🏋️ 练习**:为什么训练 loop 里要 `optimizer.zero_grad()`?如果不写会怎样?
<details><summary>答案</summary>

`.grad` 是**累加**的(`+=`)。不清零的话,这个 batch 的梯度会加到上个 batch 的残留梯度上,等价于用一个"被污染的"梯度更新。这是 PyTorch 故意这样设计的——为了支持 gradient accumulation (模拟大 batch)。
</details>

---

## 4️⃣ No-grad Context

**一句话**:推理/评估时关 autograd,**省显存 + 提速**。

```python
# 方式 1: with 上下文
with torch.no_grad():
    pred = model(past)

# 方式 2: 装饰器(整个函数都不建图)
@torch.no_grad()
def evaluate(model, loader):
    for batch in loader:
        ...

# 方式 3 (PyTorch ≥1.9): 比 no_grad 更激进的优化
with torch.inference_mode():
    pred = model(past)
```

`inference_mode` 比 `no_grad` 更省一点,但限制更严(返回的 tensor 之后不能再参与 autograd)。**部署时用 `inference_mode`,做研究分析时用 `no_grad`**。

**轨迹场景例子**:RND scoring 完全是推理,要包起来。
```python
@torch.no_grad()
def rnd_score(target_net, predictor_net, past):
    """高 score = OOD"""
    target_feat   = target_net(past)            # 随机固定网络
    predicted_feat = predictor_net(past)        # 训练过的预测网络
    return (target_feat - predicted_feat).pow(2).mean(dim=-1)
```

**🏋️ 练习**:`no_grad()` 和 `model.eval()` 区别?
<details><summary>答案</summary>

| | `no_grad()` | `model.eval()` |
|---|---|---|
| 作用 | 关 autograd,不建计算图 | 切 Dropout/BatchNorm 到推理模式 |
| 影响数值 | 不影响(只影响内存/速度) | **影响**(BN 用 running stats 而非 batch stats) |

两者**独立、推理时都要开**。
</details>

---

## 5️⃣ Model Loading

**一句话**:加载 = 先建空模型 → `load_state_dict` 填权重 → `eval()`。

```python
# 保存
torch.save(model.state_dict(), 'traj_pred.pth')

# 加载 (推荐流程)
model = TrajPredictor()                                          # 先建结构
state = torch.load('traj_pred.pth', map_location='cpu',
                   weights_only=True)                            # ★ 新版推荐,安全
model.load_state_dict(state)
model.to(device)
model.eval()                                                     # ⚠️ 推理必须!
```

**关于 `weights_only=True`**:PyTorch 2.4+ 强烈推荐——只反序列化 tensor,防止恶意 pickle 执行任意代码。下载 HuggingFace / 别人 checkpoint 时尤其要开。

**关于 `map_location`**:GPU 上训的模型在 CPU 机器上加载,必须 `map_location='cpu'`,否则报错。

**部分加载**(常见于 fine-tune / 加载预训练 encoder):
```python
state = torch.load('pretrained_encoder.pth', weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"missing keys: {missing}")           # 模型有但 state 没有的
print(f"unexpected keys: {unexpected}")     # state 有但模型没用上的
```

**🏋️ 练习**:你有一个预训练 encoder (`encoder.xxx` 开头的参数),想 freeze 它,只训 decoder。怎么做?
<details><summary>答案</summary>

```python
for name, param in model.named_parameters():
    if name.startswith('encoder.'):
        param.requires_grad = False

# optimizer 只收 decoder 参数
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3
)
# 也别忘 encoder.eval() 让它的 BN 也别乱动
model.encoder.eval()
```
</details>

---

## 6️⃣ Forward Hooks ⭐ (你 analyzer 工作的核心)

**一句话**:Hook = "在 forward 路上**埋监控点 / 改包裹**",不用动模型代码。

```python
# 三种 hook
module.register_forward_pre_hook(fn)        # 在 forward 之前  (fn 看到 input)
module.register_forward_hook(fn)            # 在 forward 之后  (fn 看到 input + output)
module.register_full_backward_hook(fn)      # 反向时           (fn 看到 grad)
```

### Hook 函数签名

```python
def fwd_hook(module, input, output):        # forward_hook
    # module:  当前模块
    # input:   tuple,即使只一个输入也是 tuple
    # output:  模块输出
    pass

def fwd_pre_hook(module, input):            # forward_pre_hook
    return modified_input                   # 返回 None = 不改;返回 tuple = 替换
```

### 用法 1: 提取中间特征

```python
features = {}
def save_feat(name):
    def hook(module, inp, out):
        features[name] = out.detach()       # ⚠️ detach!不然会 leak 计算图
    return hook

model.encoder.register_forward_hook(save_feat('encoder_out'))
_ = model(past)
print(features['encoder_out'].shape)         # 不动模型代码就拿到了 encoder 输出
```

### 用法 2: Encoder Ablation(消融实验)

```python
def ablate_encoder_dim(dim_idx):
    """把 encoder 输出某一维置零,看影响"""
    def hook(module, inp, out):
        out = out.clone()                    # 不要直接改 out!
        out[..., dim_idx] = 0
        return out                           # 返回修改后的 output
    return hook

handle = model.encoder.register_forward_hook(ablate_encoder_dim(42))
pred_ablated = model(past)
handle.remove()                              # ⚠️ 用完一定要 remove,否则后面所有 forward 都被改
```

### 用法 3: 抓所有线性层激活(RND analyzer 风)

```python
activations = {}
def attach(model):
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            h = mod.register_forward_hook(
                lambda m, i, o, n=name: activations.setdefault(n, []).append(o.detach())
            )
            handles.append(h)
    return handles
```

**🏋️ 练习**:写一个 hook,把所有 batch norm 层的输出标准化(已经是了)然后**再额外加一个噪声 N(0, 0.01)**。
<details><summary>答案</summary>

```python
def noisy_bn_hook(std=0.01):
    def hook(module, inp, out):
        return out + torch.randn_like(out) * std
    return hook

handles = []
for mod in model.modules():
    if isinstance(mod, nn.BatchNorm2d):
        handles.append(mod.register_forward_hook(noisy_bn_hook()))
# 用完: for h in handles: h.remove()
```
</details>

---

## 7️⃣ Tensor Operations (cat / stack / broadcast / einsum)

### `torch.cat` vs `torch.stack`

```
cat:   沿已有维度拼接 (维度数不变)
stack: 新建一个维度堆叠 (维度数+1)
```

```python
a = torch.randn(8, 2)       # (8, 2)
b = torch.randn(8, 2)       # (8, 2)

cat  = torch.cat([a, b], dim=0)    # (16, 2)  ← 沿时间拼
cat2 = torch.cat([a, b], dim=1)    # (8, 4)   ← 沿 feature 拼
stk  = torch.stack([a, b], dim=0)  # (2, 8, 2) ← 新出来个 batch 维
```

**轨迹场景**:把过去 + 未来拼起来形成 full trajectory:
```python
past   = torch.randn(B, N, 8, 2)
future = torch.randn(B, N, 12, 2)
full = torch.cat([past, future], dim=2)         # (B, N, 20, 2)

# 把 K 个模态的预测堆起来
mode_preds = [pred_head_k(feat) for _ in range(6)]   # 每个 (B, N, T_f, 2)
pred = torch.stack(mode_preds, dim=2)                # (B, N, 6, T_f, 2)
```

### Broadcasting(广播)

规则:从右往左对齐,对应维度要么相等、要么其中一个是 1、要么不存在。

```python
traj = torch.randn(B, N, T, 2)              # (B, N, T, 2)
ego  = torch.randn(B, 1, 1, 2)              # (B, 1, 1, 2) — 自车位置 broadcast 到所有 (N, T)
relative = traj - ego                        # (B, N, T, 2)  ← 自动广播

# 常见错误: ego = torch.randn(B, 2),做减法会报错
# 正确做法: 显式 reshape
ego = torch.randn(B, 2).view(B, 1, 1, 2)
```

### `einsum`(神器,但要熟悉)

```python
# 把每个 agent 的轨迹做一个旋转变换 (R: B x 2 x 2,traj: B x N x T x 2)
R = torch.randn(B, 2, 2)
traj = torch.randn(B, N, T, 2)

# 不用 einsum 要这样写:traj @ R.transpose(-1,-2).unsqueeze(1) ...
# einsum 一行搞定:
rotated = torch.einsum('bij,bntj->bnti', R, traj)     # (B, N, T, 2)
```

读法:`'bij,bntj->bnti'` 意思是
- 输入 1 维度命名 `b,i,j`(B×2×2)
- 输入 2 维度命名 `b,n,t,j`(B×N×T×2)
- 输出 `b,n,t,i`(共享的 `b` 保留,被求和的 `j` 消失)

**🏋️ 练习**:你有 K 个模态的预测 `pred` (B, N, K, T, 2),和模态概率 `probs` (B, N, K)。求**概率加权平均预测** `(B, N, T, 2)`。
<details><summary>答案</summary>

```python
# 方法 1: broadcasting + sum
weighted = (pred * probs.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)  # (B, N, T, 2)

# 方法 2: einsum (更清晰)
weighted = torch.einsum('bnk,bnktd->bntd', probs, pred)
```
</details>

---

## 8️⃣ Shapes and Views

### 核心 5 个操作

| 操作 | 干啥 | 内存 |
|---|---|---|
| `.view(...)` | reshape,但要求 contiguous | 不复制 |
| `.reshape(...)` | 类似 view,需要时自动 copy | 看情况 |
| `.unsqueeze(dim)` | 加一个长度为 1 的维度 | 不复制 |
| `.squeeze(dim)` | 删掉长度为 1 的维度 | 不复制 |
| `.permute(*dims)` | 维度重排 | 不复制(但变 non-contiguous) |
| `.transpose(d1, d2)` | 交换两个维度 | 不复制(变 non-contiguous) |
| `.contiguous()` | 强制连续(允许之后 view) | 复制 |

### 轨迹场景必用例子

```python
# 加 batch 维 (单样本推理)
single_past = torch.randn(64, 8, 2)              # (N, T, 2)
single_past = single_past.unsqueeze(0)            # (1, N, T, 2)

# 拍平 agent 维度送进 per-agent encoder
past = torch.randn(B, N, T, 2)
flat = past.view(B * N, T, 2)                     # 每个 agent 当成独立样本
out  = encoder(flat)                              # (B*N, D)
out  = out.view(B, N, -1)                         # 还原回 (B, N, D)
```

### `view` vs `reshape` 的坑

```python
x = torch.randn(B, N, T, 2)
y = x.permute(0, 2, 1, 3)        # (B, T, N, 2) — 但内存不连续了

z = y.view(B, T, -1)             # ❌ RuntimeError! permute 后不是 contiguous
z = y.reshape(B, T, -1)          # ✅ 自动 copy 一份再 reshape
z = y.contiguous().view(...)     # ✅ 显式 copy
```

**经验**:遇到这个错就加 `.contiguous()` 或换 `.reshape()`。

**🏋️ 练习**:你有一个 (B, T, N, 2) 的 tensor(以时间为主轴),需要送进一个 expects (B, N, T*2) 的 MLP。怎么转?
<details><summary>答案</summary>

```python
x = torch.randn(B, T, N, 2)
# 想法: 先 permute 到 (B, N, T, 2),再 reshape 把 (T, 2) 合并
y = x.permute(0, 2, 1, 3).contiguous().view(B, N, T * 2)
```
</details>

---

## 9️⃣ Masking

**一句话**:轨迹场景里,**真实数据永远是变长的**(agent 数变、历史长度变),mask 是核心工具。

### Boolean mask 选元素
```python
valid_mask = torch.tensor([[True, True, False, True]])     # (1, 4)
data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
selected = data[valid_mask]                                 # tensor([1., 2., 4.])  扁平化了!
```

### `masked_fill_`:把 mask=True 的位置填指定值
```python
scores = torch.randn(B, N)                  # 注意力 logits
agent_mask = torch.tensor(...).bool()        # (B, N), True=真实,False=padding

# 对 padding 位置填 -inf,softmax 后变 0
scores_masked = scores.masked_fill(~agent_mask, float('-inf'))
attn = scores_masked.softmax(dim=-1)
```

### `torch.where`:三元运算
```python
# 公式: where(condition, x, y) → condition 为 True 取 x 否则取 y
displacement = torch.where(valid_mask, pred - gt, torch.zeros_like(pred))
```

### ⭐ Padding Mask(轨迹任务最常用模式)

```python
# 算 masked MSE: 只在 valid 位置算 loss
def masked_mse(pred, gt, mask):
    """
    pred, gt:  (B, N, T, 2)
    mask:      (B, N, T) bool, True=有效
    """
    err = (pred - gt) ** 2                          # (B, N, T, 2)
    err = err.sum(dim=-1)                           # (B, N, T)
    err = err * mask.float()                        # 无效位置归 0
    return err.sum() / mask.sum().clamp(min=1)      # 只对有效位置求平均
```

**🏋️ 练习**:给定 `past (B, N, T, 2)` 和 `valid_mask (B, N, T)`,求**每个 agent 的平均速度**(只用 valid 的点)。
<details><summary>答案</summary>

```python
# 速度 = 相邻两帧的差
vel = past[:, :, 1:, :] - past[:, :, :-1, :]                     # (B, N, T-1, 2)
vel_mask = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]           # 两帧都 valid
speed = vel.norm(dim=-1)                                         # (B, N, T-1)
speed = speed * vel_mask.float()
mean_speed = speed.sum(dim=-1) / vel_mask.sum(dim=-1).clamp(min=1)  # (B, N)
```
</details>

---

## 🔟 Dataclasses with Tensors

**一句话**:用 dataclass 把"一个 scene 的所有 tensor"包起来,比 dict 更安全(类型检查 + 自动补全)。

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SceneTensors:
    past:        torch.Tensor                    # (B, N, T_p, 2)
    future_gt:   torch.Tensor                    # (B, N, T_f, 2)
    valid_mask:  torch.Tensor                    # (B, N, T_p) bool
    agent_mask:  torch.Tensor                    # (B, N) bool
    map_feats:   Optional[torch.Tensor] = None   # (B, M, D)

    def to(self, device):
        """复刻 nn.Module.to() 的行为"""
        return SceneTensors(
            past=self.past.to(device),
            future_gt=self.future_gt.to(device),
            valid_mask=self.valid_mask.to(device),
            agent_mask=self.agent_mask.to(device),
            map_feats=None if self.map_feats is None else self.map_feats.to(device),
        )

    @property
    def num_agents(self) -> int:
        return self.agent_mask.sum().item()
```

更高级一点:用 `pytree` 或 `torch.utils._pytree` 自动遍历(但通常自己写 `.to()` 就够了)。

**🏋️ 练习**:给上面的 `SceneTensors` 加一个 `__post_init__`,验证所有 tensor 的 batch 维度一致。
<details><summary>答案</summary>

```python
def __post_init__(self):
    B = self.past.shape[0]
    assert self.future_gt.shape[0] == B
    assert self.valid_mask.shape[0] == B
    assert self.agent_mask.shape[0] == B
    if self.map_feats is not None:
        assert self.map_feats.shape[0] == B
```
</details>

---

# 🆕 补充的 5 项

## 1️⃣1️⃣ Reduction 操作(`.mean(dim=)` / `.sum(dim=)` / `.norm(dim=)`)

**一句话**:`dim=` 决定**沿哪个维度"压扁"**,被压的维度消失或留 1。

```python
x = torch.randn(B, N, T, 2)         # (B, N, T, 2)
x.sum(dim=-1)                        # (B, N, T)   ← 压掉最后一维
x.sum(dim=-1, keepdim=True)          # (B, N, T, 1) ← 保留维度方便 broadcast
x.mean(dim=(2, 3))                   # (B, N)      ← 压两个维度
x.norm(dim=-1)                       # (B, N, T)   ← 沿 (x,y) 算 L2,变成距离
```

### 轨迹场景核心应用:**ADE / FDE 计算** ⭐

```python
def ade(pred, gt, mask):
    """
    pred, gt: (B, N, T, 2)
    mask:     (B, N, T) bool
    返回: 平均位移误差
    """
    err = (pred - gt).norm(dim=-1)                       # (B, N, T) 每个 timestep 的 L2
    err = err * mask.float()
    return err.sum() / mask.sum().clamp(min=1)

def fde(pred, gt, agent_mask):
    """
    最终时刻的位移误差。
    agent_mask: (B, N) bool
    """
    err = (pred[..., -1, :] - gt[..., -1, :]).norm(dim=-1)   # (B, N)
    err = err * agent_mask.float()
    return err.sum() / agent_mask.sum().clamp(min=1)
```

**🏋️ 练习**:写 **minADE_K**:K 个模态预测里,选 ADE 最小的那个作为分数。
<details><summary>答案</summary>

```python
def min_ade_k(pred, gt, mask):
    """
    pred: (B, N, K, T, 2)
    gt:   (B, N, T, 2)
    mask: (B, N, T)
    """
    gt_exp = gt.unsqueeze(2)                                  # (B, N, 1, T, 2)
    err = (pred - gt_exp).norm(dim=-1)                        # (B, N, K, T)
    err = err * mask.unsqueeze(2).float()                      # broadcast mask
    ade_per_mode = err.sum(dim=-1) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
                                                              # (B, N, K)
    min_ade, _ = ade_per_mode.min(dim=-1)                      # (B, N)
    return min_ade.mean()
```
</details>

---

## 1️⃣2️⃣ `.detach() / .cpu() / .numpy()` 链

**一句话**:把训练时的 tensor 安全地"丢出计算图、回 CPU、转 numpy"画图/存盘。

```python
loss_tensor = ...                # 0-d, 带 grad
# 错: numpy_loss = loss_tensor.numpy()  → 报错,因为还在计算图里 / 在 GPU 上
# 对:
numpy_loss = loss_tensor.detach().cpu().numpy()    # 标准链条
python_loss = loss_tensor.item()                    # 0-d 直接拿 python 数
```

| 方法 | 作用 |
|---|---|
| `.detach()` | 返回**共享存储**但脱离计算图的新 tensor(grad 不会传过去) |
| `.cpu()` | 搬到 CPU(如果已经在 CPU 上是 no-op) |
| `.numpy()` | 转 numpy 数组(必须在 CPU + 不带 grad) |
| `.item()` | 0-d tensor → Python scalar |
| `.tolist()` | 任意 tensor → 嵌套 list |

### 轨迹场景例子(可视化预测)

```python
@torch.no_grad()
def plot_prediction(model, batch):
    pred = model(batch['past'])              # (B, N, K, T, 2) on GPU
    pred_np = pred.detach().cpu().numpy()    # 标准转换
    past_np = batch['past'].cpu().numpy()    # 没 grad 就不用 detach

    # 现在可以用 matplotlib 画了
    for k in range(pred_np.shape[2]):
        plt.plot(pred_np[0, 0, k, :, 0], pred_np[0, 0, k, :, 1])
```

**🏋️ 练习**:为什么 `running_loss += loss` 错,而 `running_loss += loss.item()` 对?
<details><summary>答案</summary>

`loss` 还在计算图里。每个 batch 把这个 tensor 加进 `running_loss`,等于让 Python 持有了所有 batch 的计算图引用,显存会**线性爆炸**。`.item()` 把数字拿出来,Python int/float 和计算图无关。
</details>

---

## 1️⃣3️⃣ Argmax / Topk / Gather(多模态预测的核心)

```python
probs = torch.randn(B, N, K)                          # K 个模态概率 logits

# Argmax: 最大值的 index
best_mode = probs.argmax(dim=-1)                       # (B, N) int64

# Topk: 前 k 个最大值 + index
top_vals, top_idx = probs.topk(3, dim=-1)              # (B, N, 3) 各两个

# Gather: 用 index 从张量取对应元素 (比 fancy index 更通用)
# pred: (B, N, K, T, 2),想取 best_mode 对应的那个模态预测
idx = best_mode.view(B, N, 1, 1, 1).expand(B, N, 1, T, 2)
best_pred = pred.gather(dim=2, index=idx).squeeze(2)    # (B, N, T, 2)
```

### `gather` 怎么读
```
out[i][j][k] = input[i][j][index[i][j][k]]   # 沿 dim=2 取
```

### 轨迹场景:Winner-Take-All Loss
```python
# 算每个模态的 ADE
ade_per_mode = ...               # (B, N, K)
best_mode = ade_per_mode.argmin(dim=-1)                  # 最准的那个
# 只用最准模态算 loss (WTA),其他模态不更新
# 通常配 stop_gradient 一起用
```

**🏋️ 练习**:你有 `probs (B, N, K)` 和 `pred (B, N, K, T, 2)`,想取 **概率最高的 mode** 对应的预测。
<details><summary>答案</summary>

```python
best = probs.argmax(dim=-1)                                          # (B, N)
idx = best[..., None, None, None].expand(-1, -1, 1, T, 2)            # (B, N, 1, T, 2)
best_pred = pred.gather(dim=2, index=idx).squeeze(2)                  # (B, N, T, 2)
```
</details>

---

## 1️⃣4️⃣ dtype 管理

**一句话**:label 用 `long`,mask 用 `bool`,坐标用 `float32`(必要时 `float16`),**别让它们错位**。

```python
# 常见 dtype
torch.float32 / torch.float        # 默认浮点
torch.float16 / torch.half         # AMP 用
torch.float64 / torch.double       # 几乎不用
torch.int64   / torch.long         # ⭐ CrossEntropyLoss 的 label 必须是这个
torch.int32   / torch.int          # 一般索引也用 long
torch.bool                         # mask 专用

# 转换
labels.long()                       # → int64
mask.bool()                         # → bool
past.float()                        # → float32
past.to(dtype=torch.float16)        # 明确指定
```

### 轨迹场景常见 dtype 错误

```python
# ❌ mask 是 uint8 时,~mask 不是按位取反而是 -1, -2, ...
mask = torch.tensor([1, 0, 1], dtype=torch.uint8)
~mask                                # tensor([-2, -1, -2]) — 坑!

# ✅ 永远把 mask 转成 bool
mask = mask.bool()
~mask                                # tensor([False, True, False])
```

```python
# ❌ float mask × float tensor 在大 batch 上比 bool indexing 慢
loss = err[mask].mean()              # 慢
# ✅
loss = (err * mask.float()).sum() / mask.sum().clamp(min=1)   # 更快,可被 jit
```

**🏋️ 练习**:`CrossEntropyLoss` 报错 "expected long but got int"。怎么修?
<details><summary>答案</summary>

```python
labels = labels.long()      # int32 → int64
# 或更安全:
labels = labels.to(dtype=torch.long)
```
</details>

---

## 1️⃣5️⃣ 变长序列(轨迹任务的"隐藏 boss")

真实场景里每个 batch 的 agent 数量、历史长度都不同。两种应对方式:

### 方式 A: Padding 到固定长度 + mask(主流)

```python
# 假设有 4 个 scene,agent 数 [12, 8, 15, 6]
# 统一 pad 到 N_max=16,做 mask
```

PyTorch 提供 `nn.utils.rnn.pad_sequence`:
```python
from torch.nn.utils.rnn import pad_sequence

seqs = [torch.randn(t, 2) for t in [5, 8, 6]]    # 不同长度
padded = pad_sequence(seqs, batch_first=True)    # (3, 8, 2) 自动 pad 到最长

# 同步生成 mask
lengths = torch.tensor([5, 8, 6])
mask = torch.arange(8)[None, :] < lengths[:, None]    # (3, 8) bool
```

### 方式 B: `pack_padded_sequence` 给 RNN/LSTM 用

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
out, h = lstm(packed)               # LSTM 只在有效位置算
out, _ = pad_packed_sequence(out, batch_first=True)
```

> 你之前研究的 Social-LSTM、FQA 模型应该都用这一套。

### 方式 C: Nested Tensors(PyTorch 新功能,实验性)

```python
nt = torch.nested.nested_tensor([torch.randn(5, 2), torch.randn(8, 2)])
# 真正变长,无需 pad
```
还不太成熟,Transformer 里用得多,RNN 还是用 pack/pad。

**🏋️ 练习**:有 3 个 agent 的过去轨迹,长度分别 [4, 7, 5],每点是 (x, y)。pad 到统一长度并生成 valid_mask。
<details><summary>答案</summary>

```python
trajs = [torch.randn(t, 2) for t in [4, 7, 5]]
padded = pad_sequence(trajs, batch_first=True)    # (3, 7, 2)
lengths = torch.tensor([4, 7, 5])
mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]   # (3, 7) bool
```
</details>

---

## 🎯 综合练习:实现一个完整的 RND analyzer

把以上所有概念串起来。需求:
- 加载训练好的轨迹预测模型 + 一个固定的 random target encoder
- 用 forward hook 抓模型 encoder 的输出
- 对每个 agent 算 RND score = ||target_encoder(past) - hooked_encoder_out||²
- 返回 numpy 数组方便后处理

```python
import torch
import torch.nn as nn

class RNDAnalyzer:
    def __init__(self, traj_model, target_encoder, device='cuda'):
        self.model = traj_model.to(device).eval()
        self.target = target_encoder.to(device).eval()
        self.device = device

        # 1️⃣ Hook:抓 traj_model.encoder 的输出
        self._hooked_feat = None
        self.handle = self.model.encoder.register_forward_hook(self._save_feat)

    def _save_feat(self, module, inp, out):
        self._hooked_feat = out.detach()      # ⚠️ detach 防 leak

    @torch.no_grad()
    def score(self, past, agent_mask):
        """
        past:       (B, N, T_p, 2) on CPU or GPU
        agent_mask: (B, N) bool

        return: (B, N) numpy array, RND scores (高 = OOD)
        """
        # 2️⃣ Device
        past = past.to(self.device, non_blocking=True).float()

        # 3️⃣ Forward (触发 hook + 算 target)
        _ = self.model(past)
        hooked = self._hooked_feat                # (B, N, D)
        target = self.target(past)                # (B, N, D)

        # 4️⃣ Reduction: 算 RND score
        score = (target - hooked).pow(2).mean(dim=-1)   # (B, N)

        # 5️⃣ Mask 掉 padding agent
        score = score.masked_fill(~agent_mask.to(self.device), float('nan'))

        # 6️⃣ 转 numpy
        return score.detach().cpu().numpy()

    def close(self):
        self.handle.remove()                       # 清掉 hook
```

把每条注释对应回前面 15 项,你就能确认自己理解了。

---

## ✅ Checklist:这 15 项你都会了吗?

- [ ] 能讲清 `torch.tensor` 和 `from_numpy` 的区别(零拷贝)
- [ ] DataLoader 配 `pin_memory + non_blocking` 的搭配
- [ ] 知道什么时候输入要 `requires_grad_(True)`
- [ ] `no_grad` vs `eval` vs `inference_mode` 的差异
- [ ] `weights_only=True` 为什么是新默认
- [ ] 能用 hook 提中间特征 / 做消融
- [ ] `cat` vs `stack` 的维度差异
- [ ] Broadcasting 三规则 + 何时用 `unsqueeze`
- [ ] 一句 `einsum` 能读懂
- [ ] `view` 报错时知道加 `.contiguous()`
- [ ] 写得出 masked MSE
- [ ] dataclass 包 tensor + `.to()` 方法
- [ ] `dim=` 配 `keepdim=` 的用法
- [ ] `.detach().cpu().numpy()` 的标准链条
- [ ] `argmax / topk / gather` 三剑客
- [ ] `bool` mask vs `uint8` 的坑
- [ ] `pad_sequence + pack_padded_sequence` 的整套流程