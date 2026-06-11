# tensor
Core concepts to practice:

- Tensor creation: torch.zeros, torch.tensor, torch.from_numpy
- Device management: .to(device), torch.cuda.is_available()
- Autograd basics: requires_grad_(), .backward(), .grad
- No-grad context: @torch.no_grad() decorator and with torch.no_grad():
- Model loading: torch.load(), model.load_state_dict(), model.eval()
- Forward hooks: module.register_forward_hook(fn) — you'll use this for encoder
ablation
- Tensor operations: torch.cat, torch.stack, broadcasting, einsum
- Shapes and views: .view(), .reshape(), .unsqueeze(), .squeeze()
- Masking: torch.where, boolean masks, masked_fill
- Dataclasses with tensors: our codebase uses @dataclass classes that hold tensors

Recommended resources:

- PyTorch official 60-min blitz: [https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- "What is torch.nn really?" tutorial: [https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)
- Forward hooks tutorial: [https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)

# 1 Tensor creation: torch.zeros, torch.tensor, torch.from_numpy

```python
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

Random Tensor:
 tensor([[0.5615, 0.8933, 0.1537],
        [0.1362, 0.2967, 0.6852]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

NumPy array to Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

Changes in the NumPy array reflects in the tensor.

```python
np.add(n, 1, out=n)
print(f"t: {t}") # >>> t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
print(f"n: {n}") # >>> n: [2. 2. 2. 2. 2.]
```

# 2 Device management: .to(device), torch.cuda.is_available()

Each of them can be run on the GPU (at typically higher speeds than on a CPU). If you’re using Colab, allocate a GPU by going to Edit > Notebook Settings.

```python
# We move our tensor to the GPU if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")
```

参考2️⃣[tensor1.py](http://tensor1.py)

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 版本是 2.4.0，但这个 conda 环境的 torch 构建里没有包含 torch.accelerator 模块。用兼容性更好的写法替换一下
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")
```

**MPS 就是 GPU，只是苹果的叫法。**


|            | CUDA                                | MPS                             |
| ---------- | ----------------------------------- | ------------------------------- |
| 全称         | Compute Unified Device Architecture | Metal Performance Shaders       |
| 厂商         | NVIDIA                              | Apple                           |
| 适用硬件       | NVIDIA 独立显卡                         | Apple Silicon（M1/M2/M3）芯片内置 GPU |
| PyTorch 使用 | `torch.cuda`                        | `torch.backends.mps`            |
| 设备名        | `cuda:0`                            | `mps:0`                         |


**关键区别是架构：**

- **NVIDIA GPU（CUDA）**：独立显卡，有专属显存（VRAM），CPU 内存和 GPU 显存是分开的，数据需要通过 PCIe 总线来回传输
- **Apple Silicon GPU（MPS）**：集成在芯片里，和 CPU **共享同一块内存**（Unified Memory），没有独立显存，所以 `.to('mps')` 几乎不涉及数据拷贝

你的输出 `mps:0` 就是说 tensor 现在在 M 系列芯片的 GPU 核心上跑，用的是 Metal 框架加速计算。

# 3 Autograd basics: requires_grad_(), .backward(), .grad

[https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

#### Background

Neural networks (NNs) are a collection of nested functions that are executed on some input data. These functions are defined by parameters (consisting of weights and biases), which in PyTorch are stored in tensors.

Training a NN happens in two steps:

1. Forward Propagation: In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.
2. Backward Propagation: In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent.

结合这两个file:
tensor/PyTorch_prac✨.py
tensor/cifar10_v2.py

# 一句话总结

| API | 干啥 | 99% 训练里你的角色 |
|---|---|---|
| `tensor.requires_grad_(True)` | 给 tensor 打标"参与梯度计算" | **不用管** — 模型参数默认就是 True |
| `loss.backward()` | 反向传播,把梯度写进相关 tensor 的 `.grad` | **每个 batch 写一行**就够 |
| `tensor.grad` | 存梯度的属性,backward 后才有值 | **不用读** — optimizer 自己读 |

**所以你最常见的代码就这一行**:
```python
loss.backward()
```
**没了**。其他都是 PyTorch 自动处理的。

但有 **4 个场景** 你会真的手动碰 autograd —— 巧的是这些和你做的研究(OOD、RND、trajectory analyzer)正好相关。先讲三个 API 的签名,再讲场景。

---

## 1️⃣ `requires_grad_()` 详解

### 签名
```python
tensor.requires_grad_(mode: bool = True) -> Tensor  # in-place, 返回自身
tensor.requires_grad = True                          # 也可以直接赋值属性
```

`_` 后缀表示 **in-place**(直接改这个 tensor 本身)。

### 一个关键事实
**所有 `nn.Parameter`(也就是模型权重)默认 `requires_grad=True`,你不用动**。你需要碰它,只有两种情况:

| 场景 | 怎么用 |
|---|---|
| **冻结某些层**(fine-tune 时常见) | `p.requires_grad_(False)` |
| **对输入算梯度**(saliency / adversarial / RND) | `past.requires_grad_(True)` |

### 输入/输出
- 输入: `mode=True` 或 `False`(可选,默认 True)
- 输出: tensor 自身(方便链式 `x.requires_grad_(True).to(device)`)
- 副作用: 改了 tensor 的 `requires_grad` 属性

---

## 2️⃣ `.backward()` 详解

### 签名
```python
loss.backward(
    gradient=None,        # 非标量输出时才用
    retain_graph=False,   # True 时不释放计算图,可多次 backward
    create_graph=False,   # True 时建反向图 (用于高阶梯度)
    inputs=None,          # 只对指定 tensor 算梯度
)
```

### 参数到底是干啥的

| 参数 | 99% 场景 | 什么时候用 |
|---|---|---|
| `gradient` | 不传 | loss 是标量就不用;输出是 vector/tensor 时要传上游梯度 |
| `retain_graph` | 不传 | 同一个图要 backward 多次(对抗训练、某些 GAN) |
| `create_graph` | 不传 | meta-learning / 二阶导数 |
| `inputs` | 不传 | 大模型只关心某些层的梯度,省内存 |

### 输入/输出
- **输入**: `self` 必须是 scalar (0-d tensor) — 即 `loss.shape == ()`
- **输出**: `None` ⚠️
- **真正的"结果"**: 副作用 —— 把梯度写到计算图里所有 `requires_grad=True` 的叶子节点的 `.grad` 里

⚠️ **常见 bug**: 写 `grad = loss.backward()`,以为返回梯度 → 实际 `grad = None`。

### loss 必须是标量
```python
loss = criterion(pred, gt)        # () — 标量 ✓
loss.backward()

per_sample_loss = ((pred - gt)**2).sum(-1)   # (B,) — vector ✗
per_sample_loss.backward()         # RuntimeError
per_sample_loss.sum().backward()   # ✓ 先 reduce 到标量
```

---

## 3️⃣ `.grad` 详解

### 不是方法,是属性
```python
print(p.grad)                # 直接读
p.grad.zero_()               # 清零(in-place)
norm = p.grad.norm()         # 算梯度范数
```

### 关键事实

| 事实 | 含义 |
|---|---|
| backward 之前是 `None` | 没跑 backward 不要读 |
| shape 和原 tensor **完全一样** | `p.grad.shape == p.shape` |
| **是累加的**(`+=`) | 不 `zero_grad` 会跨 batch 累加 |
| 只有叶子 tensor 有 `.grad` | 中间结果默认没有(除非 `retain_grad()`) |

### 这就是为什么有 `optimizer.zero_grad()`
```python
for batch in loader:
    optimizer.zero_grad()   # 把所有 p.grad 设为 0(或者 None,新版默认 set_to_none=True)
    loss = ...
    loss.backward()         # p.grad += new_gradient
    optimizer.step()        # 读 p.grad 更新 p
```

---

# 🎯 真正会用到 autograd 手动操作的 4 个场景

## 场景 A: 冻结预训练 encoder(fine-tune 标配)

**你做 VLA / trajectory 时,加载预训练 vision encoder,只训 decoder 的情况**。

```python
# 冻结整个 encoder
for p in model.encoder.parameters():
    p.requires_grad_(False)

# ⚠️ optimizer 只收剩下能训的参数,否则它会试图更新冻结参数(虽然 grad=0,但浪费状态)
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable, lr=1e-4)

# ⚠️ encoder 里的 BatchNorm 也要 .eval(),否则 running stats 还在更新!
model.encoder.eval()
```

**只解冻最后一层**:
```python
for p in model.encoder.parameters():
    p.requires_grad_(False)
# 只放开最后一个 block
for p in model.encoder.layer4.parameters():
    p.requires_grad_(True)
```

---

## 场景 B: 输入梯度 / Saliency Analysis ⭐(你做 analyzer 会用)

**问题**:轨迹预测的输入是 `past (B, N, T_p, 2)`,模型给了一个预测。**哪些历史时间步对预测影响最大?**

答案:看输出对输入的梯度。

```python
model.eval()
past = past.detach().requires_grad_(True)   # ★ 关键

# Forward
pred = model(past)                            # (B, N, T_f, 2)

# 定义一个标量 score (比如某个 agent 未来位置的 L2)
score = pred[0, 0, -1, :].norm()              # scene 0, agent 0, 最终位置

# Backward 到输入
score.backward()

# past.grad 形状 = past 形状 = (B, N, T_p, 2)
importance = past.grad[0, 0].norm(dim=-1)     # (T_p,) — 每个历史时间步的重要性
print(importance)                              # 数值大的时间步对预测影响大
```

**注意**:推理时也用 `.requires_grad_()` + `backward()`,**不要包 `no_grad()`**(那样图都不建,backward 就没东西反向)。但模型 `eval()` 还是要的。

---

## 场景 C: RND-style OOD 分数(你的研究核心)

RND 的 score 是 `||target_net(x) - predictor_net(x)||²`。**纯推理,不需要 autograd**:

```python
@torch.no_grad()                              # ★ 关 autograd,省显存
def rnd_score(target, predictor, x):
    return (target(x) - predictor(x)).pow(2).mean(dim=-1)
```

**但训练 predictor 时需要**:
```python
# 训练 predictor 去模仿 target
for x in loader:
    optimizer.zero_grad()
    # ⚠️ target 永远不更新,用 no_grad 包起来切断它的图
    with torch.no_grad():
        target_feat = target(x)
    
    pred_feat = predictor(x)                  # 这里要建图
    loss = (target_feat - pred_feat).pow(2).mean()
    loss.backward()                            # 只反向到 predictor
    optimizer.step()
```

**这里的精妙之处**: `target_feat` 通过 `with torch.no_grad():` 没有 `requires_grad`,所以 backward 不会试图穿过 target 网络 → 自动只更新 predictor。

---

## 场景 D: 梯度累积(模拟大 batch)

**问题**:显存只够 batch=8,但论文说用 batch=64 收敛才好。怎么办?**累积 8 次再更新一次**。

```python
ACCUM = 8

optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss = criterion(model(batch['past']), batch['gt']) / ACCUM   # ★ 除以累积次数
    loss.backward()                            # .grad 自动 += 

    if (i + 1) % ACCUM == 0:
        optimizer.step()
        optimizer.zero_grad()                  # ★ 累积到位才清零
```

**为什么这能 work**:就是因为 `.grad` **天然是累加的**。这其实就是 PyTorch 设计 `.grad` 累加而不是覆盖的原因——直接支持这种模式。

---

# 🎯 你需要改吗?Verdict 表

| 任务 | 需要碰 autograd 吗? |
|---|---|
| 常规训练(SGD/Adam + loss.backward + step) | ❌ 不用,模板写就行 |
| 改用 AdamW 或不同 lr | ❌ 不用,只改 optimizer 参数 |
| 加 BatchNorm / Dropout | ❌ 不用 |
| 写新的 loss | ❌ 不用,只要 loss 是 tensor 运算就自动有 grad |
| **冻结 / 解冻部分模型** | ✅ 用 `requires_grad_(False/True)` |
| **算输入梯度做 saliency / adversarial** | ✅ 用 `input.requires_grad_(True)` |
| **梯度累积** | ✅ 利用 `.grad` 是累加的特性 |
| **梯度裁剪** | 半 ✅ 用 `nn.utils.clip_grad_norm_(model.parameters(), max_norm)` (一行) |
| **梯度可视化 / debug NaN** | ✅ 读 `p.grad`(我们的 `grad_summary` 工具就是这个) |
| GAN / meta-learning / 多次 backward | ✅ 用 `retain_graph=True` / `create_graph=True` |

---

# 🧠 心智模型(记住这个就够了)

```
计算图构建:
   x (requires_grad=True 的叶子) ─── op1 ─── op2 ─── ... ─── loss (标量)
   
loss.backward() 触发:
   反向走计算图,链式法则 → 把梯度填到所有 requires_grad=True 的叶子的 .grad 里
   
optimizer.step():
   读每个 p 的 .grad,按算法更新 p
```

**Autograd 是"自动"微分**,核心思路就是:你正向写计算,它自动帮你算反向梯度,你不需要手推链式法则。

所以平时你写的就是**正向**(forward),`.backward()` 一行触发反向,`.grad` 是结果,optimizer 消费 `.grad` 更新参数。**这就是全部循环**。

只有当你想"绕过"或"利用"这个循环的某个特性时(冻结、输入梯度、累积、可视化),你才会真的去碰 `requires_grad_()` / `.backward()` / `.grad`。

# **scalar (标量)** — what it actually means

## Definition

A **scalar (标量)** in PyTorch = a **0-dimensional tensor (零维张量)** = `shape == ()`.
Just a single number, no dimensions.

```python
torch.tensor(3.14).shape       # torch.Size([])    ← scalar (empty shape!)
torch.tensor([3.14]).shape     # torch.Size([1])   ← NOT scalar (1-D, one element)
torch.tensor([[3.14]]).shape   # torch.Size([1, 1]) ← NOT scalar (2-D)
```

### How to check

```python
loss.ndim == 0      # True if scalar
loss.shape == ()    # True if scalar
loss.dim() == 0     # True if scalar
```

⚠️ Key distinction: **a 1-D tensor with one element is NOT a scalar**. `shape=(1,)` ≠ `shape=()`.

---

## When loss IS scalar (99% of training)

Standard PyTorch loss functions use **`reduction='mean'`** by default → output is always scalar:

```python
criterion = nn.CrossEntropyLoss()        # default reduction='mean'
loss = criterion(logits, labels)         # shape () — scalar ✓
loss.backward()                          # works, no gradient param needed
```

| Loss function | Default reduction | Output |
|---|---|---|
| `nn.CrossEntropyLoss()` | `'mean'` | scalar `()` |
| `nn.MSELoss()` | `'mean'` | scalar `()` |
| `F.cross_entropy(...)` | `'mean'` | scalar `()` |
| `F.mse_loss(...)` | `'mean'` | scalar `()` |

Custom losses are also scalar when you end with `.mean()` / `.sum()`:

```python
loss = (pred - gt).pow(2).mean()                     # scalar ✓
loss = (err * mask).sum() / mask.sum().clamp(min=1)  # scalar ✓ (masked mean)
```

---

## When loss is NOT scalar (3 common cases)

### Case 1: `reduction='none'` (you opted out of reduction)
```python
criterion = nn.CrossEntropyLoss(reduction='none')
loss = criterion(logits, labels)    # shape (B,) — NOT scalar!
loss.backward()
# ✗ RuntimeError: grad can be implicitly created only for scalar outputs
```

You'd do this when you want **per-sample loss** (for hard-example mining, focal loss, weighted sampling).

### Case 2: forgot to reduce a custom loss
```python
err = (pred - gt).pow(2)             # shape (B, N, T, 2)
err.backward()                       # ✗ same error
```

### Case 3: multi-task losses kept as vector
```python
losses = torch.stack([loss_mse, loss_ce, loss_kld])   # shape (3,) — NOT scalar
losses.backward()                                      # ✗
```

The standard fix: just combine them into a scalar first.
```python
total = loss_mse + 0.1 * loss_ce + 0.01 * loss_kld   # scalar ✓
total.backward()                                      # works
```

---

## Why `.backward()` requires a scalar (math intuition)

For **scalar (标量)** output `L`, gradient w.r.t. parameter `x` is well-defined: one number per element of `x`.
$$
\frac{\partial L}{\partial x}, \quad \text{shape matches } x
$$

For **vector / tensor (向量/张量)** output `y`, the "gradient" is a **Jacobian (雅可比矩阵)** — one gradient row per output element. PyTorch refuses to handle this implicitly because:
- It's ambiguous which row you want
- For big outputs it's wasteful (you usually only want a sum/weighted combination)

So you have to tell PyTorch: "**collapse my vector output into a scalar this way**". That's what the `gradient` parameter does.

---

## The `gradient` parameter explained

```python
y.backward(gradient=g)
```

Mathematically equivalent to:
```python
(y * g).sum().backward()
```

So `gradient` is the **upstream gradient (上游梯度)** — the weights for combining vector outputs into a scalar.

### Three ways to handle a non-scalar `y`, all equivalent for backward:
```python
y = ...                                # shape (B,)

# Method 1: reduce to scalar first (★ most common)
y.sum().backward()

# Method 2: pass gradient=ones (mathematically same as Method 1)
y.backward(gradient=torch.ones_like(y))

# Method 3: weighted (different gradient for each output element)
weights = torch.tensor([1.0, 2.0, 0.5, ...])    # custom weighting
y.backward(gradient=weights)                     # same as (y * weights).sum().backward()
```

### When `gradient=` is actually useful

| Scenario | Code |
|---|---|
| **Computing one row of the Jacobian** (e.g., sensitivity of output `i` to all inputs) | `y.backward(gradient=one_hot_at_i)` |
| **Per-sample loss weighting** (hard examples weighted more) | `per_sample_loss.backward(gradient=weights)` |
| **Implementing custom backward logic** (rare in user code) | manual gradient injection |

For **trajectory prediction / OOD / RND research** — you almost never need it. **Reduce to scalar with `.mean()` or `.sum()` and just call `.backward()`**.

---

## The practical 99% rule

```python
loss = criterion(pred, gt)    # scalar  ← always make sure this is scalar
loss.backward()               # no params, just call it
```

If you ever get the error `grad can be implicitly created only for scalar outputs`, your fix is one of:
```python
loss.mean().backward()        # average over batch
loss.sum().backward()         # sum over batch
```

Don't reach for `gradient=...` unless you have a specific reason (Jacobian computation, per-sample weighting, custom backward logic). It's a power-user feature, not a daily tool.


# Part 1: **No-grad context** — 关 autograd 的几种姿势

## 一句话 summary

`torch.no_grad()` **关闭 autograd (自动微分)**: 在它的作用域里,所有 tensor 运算**不建计算图 (computation graph)**, 不分配梯度内存,也跑得更快。

## 它到底做什么 — Mechanics

PyTorch 默认每次 tensor 运算都会:
1. 记录这个运算节点 (operation node) — build the graph
2. 保存中间结果 (intermediate activations) — for backward later
3. 设置输出的 `requires_grad=True` if any input had it

`torch.no_grad()` **关掉这三件事**:
- ❌ No graph building
- ❌ No saved activations  
- ❌ Output is always `requires_grad=False`

结果是: **显存占用降低 ~50%, 速度快 10-30%, 但失去 backward 能力**.

```python
x = torch.randn(1000, 1000, requires_grad=True)

# 正常模式
y = (x ** 2).sum()
print(y.requires_grad)        # True
print(y.grad_fn)              # <SumBackward0> — 有计算图

# no_grad 模式
with torch.no_grad():
    y = (x ** 2).sum()
print(y.requires_grad)        # False ← 即使 x.requires_grad=True
print(y.grad_fn)              # None — 没有计算图
```

---

## 三种写法 — Three forms

### Form 1: **Context manager** (最常用)
```python
with torch.no_grad():
    pred = model(x)
    score = (pred - target).pow(2).mean()
# 出了 with 块, autograd 自动恢复
```

**作用域**: 只在 `with` 块内. **推荐用于**: 局部代码段, e.g. 函数中间的一部分.

### Form 2: **Decorator** (推荐用于整个函数都不需要 grad 的场景)
```python
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for x, y in loader:
        pred = model(x)
        correct += (pred.argmax(-1) == y).sum().item()
    return correct
```

整个函数体都被自动包在 `no_grad` 里. **推荐用于**: 评估函数, inference 函数, RND scoring function.

### Form 3: **Function call** (rarely used by itself)
```python
torch.no_grad()    # 单独调用其实没用,因为它返回的 context manager 没被进入
```
You'll basically never see this — it must be used as `with` or `@`.

---

## **`inference_mode()`** — no_grad 的"加强版" (PyTorch ≥ 1.9)

```python
with torch.inference_mode():
    pred = model(x)
```

比 `no_grad` 更激进的优化:
- 同样关 autograd
- **额外**关掉 version counter (用于检测 in-place 修改) → 内存再省一点
- **代价**: 在 `inference_mode` 里产生的 tensor 之后**不能**再参与 autograd

| 比较 | `no_grad` | `inference_mode` |
|---|---|---|
| 速度 | fast | **faster** |
| 显存 | low | **lower** |
| Tensor 之后能再用于 backward 吗? | ✓ Yes | ✗ No |
| 适用 | 评估 / debug / 分析 | 纯部署 / production |

**经验法则**:
- 做研究、可能要进一步分析 tensor → 用 `no_grad`
- 纯部署、明确不再需要 backward → 用 `inference_mode`

---

## **`no_grad`** vs **`model.eval()`** — 经典混淆点 ⚠️

这是 PyTorch 新手最常搞混的地方. **它们做两件完全不同的事**, 推理时**两个都要开**.

| | `with torch.no_grad():` | `model.eval()` |
|---|---|---|
| 作用对象 | 整个上下文的所有 tensor 运算 | 仅 model 内部的 Dropout / BatchNorm / LayerNorm |
| 关闭什么 | autograd (建图 + 梯度) | Dropout 的随机丢弃 + BN 切换到 running stats |
| 影响**速度/显存** | ✓ | ✗ |
| 影响**数值结果** | ✗ | ✓ (Dropout/BN 行为变了) |
| 影响**梯度计算** | ✓ | ✗ |

### 错误组合的后果

```python
# ❌ 只开 no_grad, 忘了 eval()
with torch.no_grad():
    pred = model(x)    # Dropout 仍在随机丢, BN 用 batch stats → 数值不稳

# ❌ 只开 eval, 忘了 no_grad
model.eval()
pred = model(x)        # 数值正确, 但显存爆涨 + 速度慢

# ✓ 正确组合
model.eval()
with torch.no_grad():
    pred = model(x)
```

### 你的 RND analyzer 应该这样写
```python
@torch.no_grad()
def rnd_score(target, predictor, x):
    target.eval()
    predictor.eval()
    return (target(x) - predictor(x)).pow(2).mean(dim=-1)
```

---

## 何时**不要**用 no_grad

```python
# ❌ Saliency / 输入梯度场景: 需要 backward, 不能用 no_grad
past.requires_grad_(True)
with torch.no_grad():
    pred = model(past)     # ← graph 没建!
pred.sum().backward()       # ✗ 报错: no grad accumulator

# ✓ 应该这样
past.requires_grad_(True)
model.eval()
pred = model(past)          # 正常建图
pred.sum().backward()       # ✓
```

---

# Part 2: **Model loading** — 加载训练好的模型

## 一句话 summary

Standard idiom 4 步:
```python
model = ModelClass()                                         # ① 建结构
state = torch.load(path, map_location='cpu',                  # ② 读文件
                   weights_only=True)
model.load_state_dict(state)                                  # ③ 灌权重
model.to(device).eval()                                       # ④ 上设备 + 切推理模式
```

---

## **`state_dict`** 是什么

A **state_dict** is just an `OrderedDict` from PyTorch's standard library: maps parameter name → tensor.

```python
model = nn.Linear(10, 2)
state = model.state_dict()
print(type(state))    # collections.OrderedDict
print(state)
# OrderedDict([
#   ('weight', tensor of shape (2, 10)),
#   ('bias',   tensor of shape (2,))
# ])
```

对于复杂模型, key 用点号分层:
```python
# 例如 ResNet-18:
# 'conv1.weight'
# 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var'
# 'layer1.0.conv1.weight'
# 'layer1.0.bn1.weight'
# ...
# 'fc.weight', 'fc.bias'
```

**关键事实**:
- state_dict 包含**所有 parameters** (weights, biases) **+ all buffers** (BN 的 running_mean / running_var)
- 不包含**模型结构** (architecture) — 加载时你必须自己重新建结构
- 也不包含 **optimizer state** (那是单独的 dict)

---

## **`torch.save()`** — 保存

### ✅ 推荐方式: 只存 state_dict (weights only)
```python
torch.save(model.state_dict(), 'model.pth')
```

### ✗ 不推荐: 存整个 model 对象
```python
torch.save(model, 'model.pth')    # ← 不推荐
```

为什么不推荐:
- 它用 `pickle` 序列化整个 Python 对象 → 依赖 file 路径、依赖 class 定义、依赖 Python 版本
- 换环境就崩 ("ModuleNotFoundError: No module named 'old_project.models'")
- 也有安全风险 (任意代码执行 via pickle)

### ✅ Resume training 场景: 存 checkpoint dict
```python
torch.save({
    'epoch': epoch,
    'model_state':     model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'best_metric':     best_metric,
}, 'checkpoint.pth')
```

---

## **`torch.load()`** — 读文件

```python
state = torch.load(path, map_location='cpu', weights_only=True)
```

### 关键参数

#### **`map_location`** — 控制 tensor 加载到哪个 device
```python
torch.load(path)                         # 加载到原 device (训练时的 device)
torch.load(path, map_location='cpu')     # ★ 推荐: 总是先到 CPU, 后续手动 .to(device)
torch.load(path, map_location='cuda:0')  # 直接到指定 GPU
```

**典型坑**: 在 GPU 上训的 checkpoint, 在 CPU 机器上加载时, 不指定 `map_location` 会报错.

**推荐做法**: 总是 `map_location='cpu'`, 后面手动控制 device — 行为可预测.

#### **`weights_only`** ⭐ (PyTorch ≥ 2.4 强烈推荐)
```python
torch.load(path, weights_only=True)    # ★ 推荐 (PyTorch 2.6+ 已经是默认)
```

- `weights_only=True`: 只 deserialize **tensors**, 拒绝任意 Python 对象 → 防止恶意 pickle 执行任意代码
- `weights_only=False`: 老行为, 可以加载任意 Python object (但有安全风险)
- 加载 HuggingFace / 第三方 checkpoint 时尤其重要

如果你存的是单纯的 state_dict, `weights_only=True` 一定 work. 如果存的是含自定义 class 的 dict, 可能要 `weights_only=False` 或者用 `torch.serialization.add_safe_globals(...)` 显式 allowlist.

---

## **`model.load_state_dict()`** — 灌权重

```python
missing, unexpected = model.load_state_dict(state, strict=True)
```

### **`strict=True`** vs **`strict=False`**

- `strict=True` (default): keys 必须**完全匹配**, 多一个少一个都报错
- `strict=False`: 允许不匹配, 返回 `(missing_keys, unexpected_keys)` 让你检查

```python
# 典型场景: 只加载预训练的 encoder, head 重新初始化
pretrained = torch.load('pretrained.pth', weights_only=True)
encoder_only = {k: v for k, v in pretrained.items() if k.startswith('encoder.')}

missing, unexpected = model.load_state_dict(encoder_only, strict=False)
print(f"Missing (head 部分): {missing}")           # 缺的 keys — 这些保持随机初始化
print(f"Unexpected (多余): {unexpected}")          # 多的 keys — 这些被忽略
```

### **DataParallel / DDP 的 `'module.'` 前缀坑**
用 `nn.DataParallel` 或 `DistributedDataParallel` 训练时, state_dict 的 keys 会自动加 `'module.'` 前缀. 加载时如果不用 DDP, 要手动剥掉:
```python
state = torch.load(path, weights_only=True)
state = {k.replace('module.', '', 1): v for k, v in state.items()}
model.load_state_dict(state)
```

---

## **`model.eval()`** — 切换推理模式

```python
model.eval()
```

具体做什么 (递归地, 对模型里所有子模块):
- **Dropout**: 关闭随机丢弃 → 推理时确定性
- **BatchNorm**: 用 **running_mean / running_var** (累积的全数据集统计) 而不是 batch 统计
- **其他**: LayerNorm/Embedding/Linear/Conv 等**不受影响** (它们没有 train/eval 之分)

要切回训练模式: `model.train()`.

**Source of truth**: `model.training` (bool) 告诉你当前模式.
```python
print(model.training)    # True 或 False
```

### Subtle 但常见的坑: BN 在 fine-tune 时

```python
# 冻结 encoder 训 head
for p in model.encoder.parameters():
    p.requires_grad_(False)

# ⚠️ 但 model.train() 时, encoder 的 BN 仍然在 update running_mean!
# 这会破坏预训练统计量

# ✓ 正确做法
model.train()              # head 部分用 train mode
model.encoder.eval()       # encoder 强制 eval, BN 不再 update
```

---

## 完整 idiom — Save & load checkpoint 全套

```python
# ─── 训练时保存 ─────────────────────────────────
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_metric):
    torch.save({
        'epoch': epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_metric':     best_metric,
    }, path)

# ─── 恢复训练 ──────────────────────────────────
def load_for_resume(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    model.to(device)
    return ckpt['epoch'], ckpt['best_metric']

# ─── 推理使用 ──────────────────────────────────
def load_for_inference(path, device):
    model = TrajPredictor()
    state = torch.load(path, map_location='cpu', weights_only=True)
    # 如果存的是 checkpoint dict 而非纯 state_dict:
    if 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    model.to(device).eval()      # ★ 推理两件套: device + eval
    return model
```

---

## 速查表 — Cheatsheet

### No-grad
```
Form                                   When to use
─────────────────────────────────────────────────────────
with torch.no_grad():                  局部代码段
@torch.no_grad()                       整个函数 (推荐用于 eval/score)
with torch.inference_mode():           纯部署 (PyTorch ≥ 1.9)

Pair with:                             model.eval() ← 永远配套
Don't use when:                        需要 backward (输入梯度场景)
```

### Model loading
```
Step                            Code
─────────────────────────────────────────────────────────────────────
1. Build architecture           model = MyModel()
2. Load state dict from disk    state = torch.load(path,
                                       map_location='cpu',
                                       weights_only=True)
3. Inject weights               model.load_state_dict(state)
                                # or: load_state_dict(state, strict=False)
4. To device + eval mode        model.to(device).eval()

What to save?                   model.state_dict()  ← 仅权重, 不要存 model 对象
Checkpoint for resuming?        dict of {model/optim/scheduler/epoch}
DDP 训练的 checkpoint?           剥 'module.' 前缀再 load
fine-tune 部分加载?              strict=False, 检查 missing/unexpected keys
```

---

# Final mental model

**No-grad** = "this code is for **using** the model, not learning from it". 关 autograd 拿速度和显存.

**Model loading** = "**rebuild the architecture, pour the weights in, freeze the random behavior**". 三步缺一不可: state_dict → load → eval.

两个一起常常作为**部署模板**:
```python
model = MyModel()
model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
model.to(device).eval()

@torch.no_grad()
def predict(x):
    return model(x.to(device))
```

这八行代码就是绝大多数 inference / evaluation script 的骨架.


# Part 1: **`.pth`** 是什么文件?

就是一个**文件名 + 约定俗成的后缀**. `.pth` 在 PyTorch 社区习惯标识 "saved PyTorch object" — 但**只是约定, 不强制**.

文件内部是一个 **pickled Python object (序列化的 Python 对象)** — 你用 `torch.save()` 存什么进去, 它就装什么. 通常是个 dict.

### 常见扩展名 (都只是命名习惯)
```
.pth, .pt      ─ PyTorch 社区
.bin           ─ HuggingFace 老格式
.safetensors   ─ HuggingFace 新格式 (更安全, 防 pickle 注入攻击)
.ckpt          ─ PyTorch Lightning 用
.h5            ─ Keras / TensorFlow
```

⚠️ 这些**只是后缀**, 实际内容由 `torch.save()` 决定. 你存成 `whatever.xyz` 也能 work — 只是代码可读性差.

---

# Part 2: **`state_dict`** vs **`checkpoint`** — 都装在 `.pth` 里, 内容不同

### `state_dict` — 仅模型权重 (轻量)
```python
torch.save(model.state_dict(), 'model.pth')
```
文件里实际装的内容:
```python
OrderedDict([
    ('encoder.lin1.weight', tensor(shape=(128, 2))),
    ('encoder.lin1.bias',   tensor(shape=(128,))),
    ('head.weight',         tensor(shape=(12, 128))),
    ('head.bias',           tensor(shape=(12,))),
])
```
Only weights. 适合**仅推理**.

### `checkpoint` — 训练状态的完整快照 (重量)
```python
torch.save({
    'epoch':           epoch,
    'model_state':     model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'best_metric':     best_metric,
}, 'checkpoint.pth')
```
文件里实际装的内容:
```python
{
    'epoch': 42,
    'model_state':     OrderedDict([...]),     # ← state_dict 包在里面
    'optimizer_state': {'state': {...}, 'param_groups': [...]},
    'scheduler_state': {...},
    'best_metric':     0.873,
}
```
包含**继续训练所需的一切**.

### "Checkpoint" 这个词的来历

来自电子游戏 / 数据库 — **a snapshot of state you can return to (可回溯的状态快照)**.

在 ML 训练里同理:
- 训练跑到 epoch 42 突然宕机 → 没 checkpoint 就全白费
- 第 60 个 epoch 验证集表现最好 → 保存为 "best.pth"
- 半年后想接着 fine-tune → 加载 checkpoint 接上

所以 **"checkpoint" = 训练过程中的一个可恢复时间点**.

---

# Part 3: 为什么可以"直接拿来用" — 概念性答案

> **神经网络的所有"知识"都存在它的 weights (权重) 里. 一旦存了 weights, 就有了模型学到的全部.**

### 类比: 做菜

| 比喻 | 对应 ML 概念 |
|---|---|
| **菜谱 (Recipe)** | 模型架构 — 多少层、什么层、怎么连 |
| **调料用量** (Seasoning amounts) | 训练学到的参数值 (weights) |
| **做菜过程** (Cooking) | 训练过程 |
| **反复试错调整调料** (Tasting & adjusting) | Gradient descent 在调整 weights |

任何人**有菜谱 (架构) + 有调料用量 (weights)** = 都能复现这道菜.

### 数学上

A neural network is a function $f_\theta(x)$ parametrized by $\theta$ (所有权重的集合). 训练做的是优化 $\theta$:
$$
\theta^* = \arg\min_\theta \mathbb{E}_{(x, y)} \big[ \text{loss}(f_\theta(x), y) \big]
$$

训练完毕后, 你有了 $\theta^*$. 之后所有推理就是套用 $f_{\theta^*}(x)$. 同样的 $x$ 永远得到同样的 output (deterministic).

**保存 $\theta^*$ = 保存模型的全部"学到的东西"**.

### PyTorch 的设计哲学

`nn.Module` 把模型拆成两块, 设计上**完全解耦**:

```
┌─────────────────┬──────────────────────┐
│   Architecture  │       State          │
│   (代码定义)     │   (state_dict 存)     │
├─────────────────┼──────────────────────┤
│  class MyNet:   │  {'lin1.weight':     │
│    lin1 = ...   │     tensor(...),     │
│    lin2 = ...   │   'lin1.bias':       │
│    fc = ...     │     tensor(...),     │
│                 │   ...}               │
└─────────────────┴──────────────────────┘
        ↓                    ↓
     Python class         .pth file
```

加载 = 把 `state_dict` 里的 tensor **按名字** copy 进 architecture 里对应的位置. 仅此而已.

---

# Part 4: 怎么拿来用 — 4 个典型场景

## Scenario A: 用你自己训好的模型 (最简单)

```python
# ─── 训练完毕时存 ──────────────────────────
torch.save(model.state_dict(), 'best_model.pth')

# ─── 之后 inference 脚本里 ────────────────
from my_models import TrajPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ① 用同样 config 重建结构
model = TrajPredictor(hidden_dim=128, T_f=12, D=2)

# ② 读文件
state = torch.load('best_model.pth',
                   map_location='cpu',
                   weights_only=True)

# ③ 灌权重
model.load_state_dict(state)

# ④ 上设备 + 切推理模式
model.to(device).eval()

# 现在可以用了
@torch.no_grad()
def predict(past):
    return model(past.to(device))
```

⚠️ **关键点 ①**: config 必须严格匹配训练时的! 如果训练时 `hidden_dim=128`, 加载时建 `hidden_dim=64` → tensor shape 对不上 → `RuntimeError`.

### Best practice: 把 config 也存进去
```python
torch.save({
    'model_state': model.state_dict(),
    'config': {'hidden_dim': 128, 'T_f': 12, 'D': 2},
}, 'best_model.pth')

# 加载
ckpt = torch.load('best_model.pth', weights_only=True)
model = TrajPredictor(**ckpt['config'])          # ★ 自动拿正确 config
model.load_state_dict(ckpt['model_state'])
```

这就是为什么很多代码库分两个文件: `model.pth` (weights) + `config.json` (架构超参).

## Scenario B: 训练中断 — Resume 接着训

需要**完整 checkpoint** (含 optimizer / scheduler / epoch).

```python
model     = TrajPredictor(**config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
start_epoch = 0

# 如果有 checkpoint, 加载继续
if os.path.exists('checkpoint.pth'):
    ckpt = torch.load('checkpoint.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

model.to(device)

for epoch in range(start_epoch, 100):
    train_one_epoch(...)
    scheduler.step()

    # 每个 epoch 存 checkpoint, 实现"宕机也不怕"
    torch.save({
        'epoch': epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }, 'checkpoint.pth')
```

**为什么 optimizer state 也要存**? Adam / SGD-with-momentum **本身有内部状态** (历史梯度的 running averages). 不存的话, resume 时这些状态被清零 → loss 突然变高 (像运动员热身被打断重做).

## Scenario C: torchvision 预训练模型 (你 CIFAR 脚本就用过这个)

torchvision 帮你**自动下载 + 加载** ImageNet 预训练权重.

```python
import torchvision.models as models

# 自动从 PyTorch 服务器下载 weights 到 ~/.cache/torch/hub/checkpoints/
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 已经是训好的, 直接用
model.eval()
```

背后: torchvision 维护了 URL → checkpoint 的映射. 第一次调用时下载, 之后从本地缓存读. 文件存在 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`.

可以打开看看里面是啥:
```python
state = torch.load('~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth', weights_only=True)
print(list(state.keys())[:5])
# ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']
```
就是 state_dict, 没什么神秘的.

## Scenario D: HuggingFace 模型 (你 VLA / VLM 工作会用)

```python
from transformers import AutoModel, AutoTokenizer

# 第一次跑自动下载到 ~/.cache/huggingface/
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model.eval()
inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

HuggingFace 的 `from_pretrained()` 帮你做了 4 步:
1. 下载 checkpoint (e.g., `pytorch_model.bin` 或 `model.safetensors`)
2. 下载 `config.json` (architecture 配置)
3. 用 config 建对应的 architecture
4. Load state_dict 进去

你只需要写一行.

---

# Part 5: 常见坑 — Common pitfalls

## Pitfall 1: Architecture mismatch ⭐ 最常见
```python
# 训练时
model = TrajPredictor(hidden_dim=128)
torch.save(model.state_dict(), 'model.pth')

# 加载时改错 config
model = TrajPredictor(hidden_dim=64)
model.load_state_dict(torch.load('model.pth', weights_only=True))
# ✗ RuntimeError: size mismatch for encoder.lin1.weight:
#   expected [64, 2] but loaded [128, 2]
```
**Fix**: 严格匹配 architecture, 或把 config 也存进 checkpoint.

## Pitfall 2: DDP `'module.'` 前缀
DDP 训出来的 state_dict, keys 自动多了 `'module.'` 前缀:
```python
# DDP 模型存的: 'module.encoder.lin1.weight'
# 单 GPU 模型期待: 'encoder.lin1.weight'

state = torch.load('ddp_model.pth', weights_only=True)
state = {k.replace('module.', '', 1): v for k, v in state.items()}
model.load_state_dict(state)
```

## Pitfall 3: 忘了切 `.eval()`
```python
model.load_state_dict(state)
# (忘了 model.eval())
pred = model(x)    # Dropout 还在随机丢, BN 用 batch stats → 数值不稳
```

## Pitfall 4: 想 resume 训练但只存了 state_dict
```python
torch.save(model.state_dict(), 'model.pth')  # 只有 weights, 没 optimizer 状态
# ...训练 100 epoch 后宕机...
# 加载: model OK, 但 optimizer 状态丢, scheduler 不知到哪了 → resume 效果断档
```

---

# Mental model — 一张图记住

```
.pth file (binary)
├── 装的是 state_dict 吗?
│       内容: {'layer.weight': tensor, ...}
│       用途: 推理 / fine-tune 时用
│
└── 装的是 checkpoint dict 吗?
        内容: {'epoch': ..., 'model_state': state_dict, 'optimizer_state': ...}
        用途: resume 中断的训练

加载流程 (永远 4 步):
  ① 用同样 config 重建 model architecture
  ② torch.load(path, map_location='cpu', weights_only=True)
  ③ model.load_state_dict(state)
  ④ model.to(device).eval()              ← 推理时
     or model.to(device)                  ← resume 训练时
```

下次看到任何 `.pth` 文件, 想了解里面装什么, **打开看**:
```python
obj = torch.load('mystery.pth', map_location='cpu', weights_only=True)
print(type(obj))                # OrderedDict? dict?
print(list(obj.keys())[:10])    # 前 10 个 keys
```
看 keys:
- 全是 `'layer.weight'` 风格 → 纯 state_dict
- 有 `'epoch'`, `'model_state'` 这种顶层 keys → checkpoint dict


# Part 1: **"idiom"** 是什么

In programming context, **idiom (惯用法)** = a **standard, recognizable code pattern (代码套路)** that experienced people write without thinking.

Not a language feature. Not a forced rule. Just **community consensus** on the best way to do X.

这个词**来自自然语言** — natural language 里 "idiom" 是**习语 / 惯用语** (e.g., "kick the bucket"), 字面 ≠ 实际意义. 在编程里类似 — 一个 pattern 被识别为**一个整体**, 不是从单个 token 推出来的.

### Python 通用 idiom 例子
```python
# Idiom: 文件 IO
with open('x.txt') as f:           # 用 with, 不手动 close
    data = f.read()

# Idiom: script 入口
if __name__ == '__main__':         # 这整个 pattern 就是 idiom
    main()
```

### PyTorch idiom 例子 (前几个回复都用过)
```python
# 5-step training idiom
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
optimizer.step()

# 4-step model loading idiom
model = MyModel()
state = torch.load(path, map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.to(device).eval()

# tensor → numpy idiom
arr = tensor.detach().cpu().numpy()

# Inference idiom (两件套)
model.eval()
with torch.no_grad():
    out = model(x)
```

每当我说 "**this is the standard idiom**", 意思是: **直接套用, 不用每次重新思考**.

---

# Part 2: `model.eval()` / `model.train()` — 到底切什么开关?

## 它实际做的事 (mechanics)

**很简单**: 翻转一个布尔标志 `model.training`.

```python
model = MyModel()
print(model.training)         # True (默认值)

model.eval()
print(model.training)         # False

model.train()
print(model.training)         # True
```

这个标志**递归地**应用到所有 submodule:
```python
model.eval()
print(model.encoder.training)    # False — 自动传播
print(model.encoder.lin1.training)  # False — 一路到底
```

## 翻这个开关有什么用?

PyTorch 某些 layer 会**检查 `self.training`**, 根据它**改变行为**. 主要是这两个:

### **Dropout**
```python
class Dropout:
    def forward(self, x):
        if self.training:
            return x * random_mask / (1 - p)   # 随机丢 + rescale
        else:
            return x                            # identity, 不做任何事
```

### **BatchNorm**
```python
class BatchNorm:
    def forward(self, x):
        if self.training:
            mean, var = x.mean(...), x.var(...)        # 用 batch 自己的统计
            self.running_mean += ...                    # 顺便累积更新 running stats
        else:
            mean, var = self.running_mean, self.running_var   # 用累积的 running stats
        return (x - mean) / sqrt(var) * gamma + beta
```

### 哪些层关心 `self.training`?

| 层 | 关心? | 行为差异 |
|---|---|---|
| **`nn.Dropout` / `Dropout2d`** | ✓ | train: 随机丢. eval: identity |
| **`nn.BatchNorm1d/2d/3d`** | ✓ | train: batch stats. eval: running stats |
| `nn.RNN/LSTM/GRU` (dropout>0) | ✓ | dropout 同上 |
| `nn.LayerNorm` | ✗ | 永远算当前输入 |
| `nn.GroupNorm` | ✗ | 同上 |
| `nn.Linear / Conv2d` | ✗ | 没区别 |
| `nn.ReLU / Sigmoid / Tanh` | ✗ | 没区别 |

⭐ **重要事实**: 如果你的模型**没有 Dropout 也没有 BatchNorm**, 那 `eval()` / `train()` **数值上没有任何差别**. 比如纯 ViT (用 LayerNorm) 在两种模式下输出完全一样.

## 什么时候用?

### 新建的 model 默认 `training=True`
```python
model = MyModel()         # 已经是 train 模式, 直接训就行
```

你**不需要**在训练开始前 explicitly 写 `model.train()`. 但写一下更明确, 也防止从 eval 状态切回来漏掉.

### 标准训练 + 评估循环
```python
for epoch in range(epochs):
    model.train()                # ★ 进训练阶段
    for batch in train_loader:
        # train step ...
        pass

    model.eval()                 # ★ 切评估
    with torch.no_grad():
        for batch in val_loader:
            # eval step ...
            pass
    # 下一轮 epoch 开头会重新 model.train() 切回来
```

⚠️ **必须切回 `train()`** — 评估完不切回来, 下一个 epoch 训练时 BN 还在用 running stats → loss 不收敛.

### Fine-tune 时 per-submodule 控制 (重要)
```python
# 冻结预训练 encoder 只训 head
for p in model.encoder.parameters():
    p.requires_grad_(False)      # 参数不更新

model.train()                    # head 部分进训练模式
model.encoder.eval()             # ★ 但 encoder 强制 eval 模式!
```

**为什么这样**? Encoder 的参数虽然 `requires_grad=False`, 但 **BatchNorm 的 `running_mean` / `running_var` 不是 parameter, 它们是 buffer**, 在 `training=True` 时**还会被更新**. 想保护预训练统计量, 必须让 encoder 保持 eval 模式.

### 与 `no_grad()` 的关系 (复习一下)

| | `model.eval()` | `with torch.no_grad():` |
|---|---|---|
| 影响什么 | Dropout / BN 的**内部行为** | autograd (不建计算图) |
| 影响**数值** | ✓ Yes | ✗ No |
| 影响**显存/速度** | 很少 | ✓ 大幅减少 |
| 推理时 | **都要开** | **都要开** |

---

# Part 3: `torch.einsum` — Einstein Summation

## Mental model

**`einsum` 是一套描述"如何组合多个 tensor"的紧凑符号**. 看上去像魔法, 本质就是 **3 条规则**.

### 语法
```python
torch.einsum('input1_dims, input2_dims, ... -> output_dims',
             tensor1, tensor2, ...)
```

每个**字母** = 一个 **dimension**. 字母告诉 PyTorch 这个维度的"角色".

### 3 条核心规则

1. **重复字母** (在不同输入间出现多次) → matched, 大小必须相同, **沿这个维度逐位相乘**
2. **只在输入出现, 不在输出** → 沿这个维度**求和 (summed out / contracted 收缩)**
3. **在输出里** → **保留** (这一维出现在结果里)

掌握这 3 条 = 能读懂任何 einsum.

## 从最简单开始

### Example 1: Vector sum
```python
v = torch.tensor([1., 2., 3.])
torch.einsum('i->', v)         # tensor(6.)
```
- `i` 在输入有, 输出没有 → **沿 i 求和** → 标量

### Example 2: Element-wise product
```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
torch.einsum('i,i->i', a, b)   # tensor([4., 10., 18.])
```
- 两输入都是 `'i'` → matched (逐位乘)
- `i` 在输出 → 保留

### Example 3: Dot product
```python
torch.einsum('i,i->', a, b)    # tensor(32.)  ← 1·4 + 2·5 + 3·6
```
- 同 Example 2, 但输出是空 → 多了一个"沿 i 求和"的动作

### Example 4: Matrix multiplication
```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
torch.einsum('ij,jk->ik', A, B)   # shape (2, 4), 等价 A @ B
```
- `j` 在两个输入都有 → matched
- `j` 不在输出 → 沿 j 求和
- `i, k` 在输出 → 保留

**这就是矩阵乘法的数学定义**: $(AB)_{ik} = \sum_j A_{ij} \cdot B_{jk}$

### Example 5: Transpose
```python
torch.einsum('ij->ji', A)      # shape (3, 2)
```
换字母顺序 = 维度重排 = transpose.

### Example 6: Trace
```python
M = torch.randn(5, 5)
torch.einsum('ii->', M)        # scalar (对角线之和)
```
- 同 tensor 内两个相同字母 → 取对角元素 (限定 i 一致)
- 输出空 → 求和

## 速查表

| 操作 | einsum | 等价写法 |
|---|---|---|
| Vector sum | `'i->'` | `v.sum()` |
| Element-wise mul | `'i,i->i'` | `a * b` |
| Dot product | `'i,i->'` | `(a*b).sum()` |
| Outer product | `'i,j->ij'` | `a[:,None] * b[None,:]` |
| Matrix mul | `'ij,jk->ik'` | `A @ B` |
| Batched matmul | `'bij,bjk->bik'` | `torch.bmm(A, B)` 或 `A @ B` |
| Transpose | `'ij->ji'` | `A.T` |
| Trace | `'ii->'` | `A.diagonal().sum()` |
| Diagonal extract | `'ii->i'` | `A.diagonal()` |
| Bilinear form $x^T A y$ | `'i,ij,j->'` | `x @ A @ y` |

## 你 trajectory 工作的实际例子

### Example 7: 每个 scene 不同旋转, 批量旋转所有 agent 轨迹
```python
R    = torch.randn(B, 2, 2)            # 每个 scene 一个旋转矩阵
past = torch.randn(B, N, T, 2)         # 过去轨迹

rotated = torch.einsum('bij,bntj->bnti', R, past)
# shape: (B, N, T, 2)
```

**逐字母解读**:
| 字母 | 在 R 中 | 在 past 中 | 在输出中 | 作用 |
|---|---|---|---|---|
| `b` | ✓ | ✓ | ✓ | matched + 保留 (每个 scene 用自己的 R) |
| `i` | ✓ | ✗ | ✓ | 只在 R + 保留 (输出坐标维) |
| `j` | ✓ | ✓ | ✗ | matched + **求和** (矩阵乘的"内层维") |
| `n` | ✗ | ✓ | ✓ | 只在 past + 保留 (agent 维) |
| `t` | ✗ | ✓ | ✓ | 只在 past + 保留 (时间维) |

**数学含义**: `output[b,n,t,i] = Σ_j R[b,i,j] * past[b,n,t,j]`,就是给每个 scene 的每个 agent 的每个时间步, 用对应 R 矩阵旋转 (x, y) 坐标.

### Example 8: 多模态预测的加权平均
```python
pred  = torch.randn(B, N, K, T, 2)         # K 个模态的预测
probs = torch.randn(B, N, K).softmax(-1)   # 每个模态的概率

weighted = torch.einsum('bnk,bnktd->bntd', probs, pred)
# shape: (B, N, T, 2) — 沿 K 加权求和
```

`k` matched + 求和 → 加权平均.
其他维度全保留.

### Example 9: Pairwise distance (你 DC-MMD 算 kernel 时用得到)
```python
A = torch.randn(M, T*2)                    # M 条 reference
B = torch.randn(N, T*2)                    # N 条 query

diff = A.unsqueeze(1) - B.unsqueeze(0)     # (M, N, T*2)
dist_sq = torch.einsum('mnd,mnd->mn', diff, diff)
# shape (M, N), dist_sq[m, n] = ||A[m] - B[n]||²
```

## 什么时候**不**用 einsum

`einsum` 通用但不一定最清晰. 简单操作直接用原生 API 更好读:

| 操作 | einsum 写法 | 更好的写法 |
|---|---|---|
| 标准矩阵乘法 | `'ij,jk->ik'` | `A @ B` ★ |
| Batched matmul | `'bij,bjk->bik'` | `A @ B` ★ |
| 沿一维求和 | `'ij->i'` | `M.sum(dim=1)` ★ |
| 3+ tensor 跨维度组合 | `'bij,bntj,btd->...'` | ★ **einsum 最清晰** |

**经验法则**: 简单 2-tensor 操作用 PyTorch 原生 API, **3+ tensor 跨多维 contraction 用 einsum**.

## 读 einsum 的训练方法

每次看到一个 einsum, 心里默念这 4 步:

```python
torch.einsum('bnk,bnktd->bntd', probs, pred)
              ──┬─ ──┬──   ──┬─
              输入1 输入2    输出

第1步: 哪些字母在所有输入都有 → matched
       b, n → 共享 / broadcast
第2步: 哪些字母只在输入,不在输出 → 求和
       k → 沿 k 收缩
第3步: 哪些在输出 → 保留
       b, n, t, d
第4步: 数学含义?
       output[b,n,t,d] = Σ_k  probs[b,n,k] * pred[b,n,k,t,d]
```

练 10 个例子就成肌肉记忆.

---

# 一句话总结

| 概念 | 一句话 |
|---|---|
| **idiom (惯用法)** | 社区公认的"标准代码套路", 跟着写最稳 |
| **`model.eval()` / `.train()`** | 翻 `self.training` 这个布尔标志, 影响 **Dropout / BatchNorm** 的内部行为 (其他层不在乎) |
| **`einsum`** | 用字母给维度命名 — **重复字母 matched, 不出现在输出的字母求和, 出现在输出的字母保留** |

下次看到这三个东西不会再卡壳了:
- 看到 "follow the idiom" → "OK, 标准套路, 照抄"
- 看到 `model.eval()` → "对, Dropout / BN 切到推理模式"
- 看到 `einsum('bij,bjk->bik', ...)` → "batched matmul, 沿 j 求和, b/i/k 保留"