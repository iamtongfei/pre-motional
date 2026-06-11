# PyTorch 入门:CIFAR-10 训练流程笔记

> 这份笔记按 **概念** 组织(不是按代码顺序),配合 `cifar10_annotated.py` 一起看。
> 已经懂 ML 概念了,所以只讲 **PyTorch API 的设计逻辑** 和 **新手坑**。

---

## 🎯 一句话总览

> **PyTorch 训练 = 五步循环:`zero_grad → forward → loss → backward → step`**

把这五步刻进 DNA,以后所有训练脚本都是它的变体。

---

## 📦 整个流程的数据流向

```
原始 PIL 图片
   │  transforms (ToTensor + Normalize)
   ▼
Tensor (3, 32, 32), 值域 [-1, 1]
   │  DataLoader (batching + shuffle)
   ▼
Batch (4, 3, 32, 32)
   │  .to(device)        ← 搬到 GPU
   ▼
GPU 上的 input
   │  net(inputs)         ← forward
   ▼
logits (4, 10)
   │  criterion(outputs, labels)
   ▼
loss (标量,带计算图)
   │  loss.backward()     ← 反向传播,填 .grad
   ▼
每个参数有了 .grad
   │  optimizer.step()    ← 用 .grad 更新参数
   ▼
模型参数被更新一次
```

---

## 1️⃣ Dataset / DataLoader / Transform

### 三者职责分工

| 组件 | 干什么 | 类比 |
|---|---|---|
| `Dataset` | 定义 "第 i 个样本是什么"(单样本) | 仓库里第 i 个货架 |
| `transform` | 对单个样本做预处理 | 出库前包装一下 |
| `DataLoader` | 批量、打乱、并行加载 | 物流系统 |

### ToTensor 干了两件事(常被忽略)
1. `(H, W, C)` → `(C, H, W)`(PyTorch 的 channel-first 约定)
2. `uint8 [0, 255]` → `float32 [0.0, 1.0]`(自动除 255)

### Normalize 的 mean/std
- 长度必须等于 channel 数(RGB 就是 3 个)
- 公式:`out = (in - mean) / std`
- 这里 `(0.5, 0.5, 0.5)` 把 `[0, 1]` 映射到 `[-1, 1]`
- **实战中常用 ImageNet 统计量**:
  ```python
  mean=[0.485, 0.456, 0.406]
  std =[0.229, 0.224, 0.225]
  ```

### DataLoader 的 `num_workers` 坑
- `num_workers > 0` 用子进程并行加载
- **Windows / Jupyter** 上可能崩 → 报错就设 0
- Linux + 命令行运行,可以设到 4~8

---

## 2️⃣ nn.Module:为什么要继承它?

继承 `nn.Module` 之后,你**白嫖**了一堆能力:

| 方法 | 干什么 |
|---|---|
| `.parameters()` | 递归收集所有 `nn.Parameter`(给 optimizer 用) |
| `.to(device)` | 把所有参数一键搬到 GPU |
| `.train()` / `.eval()` | 切换 Dropout / BatchNorm 模式 |
| `.state_dict()` | 导出权重 dict(给 `torch.save` 用) |
| `.load_state_dict()` | 加载权重 |

### `super().__init__()` 别忘
没调用父类构造函数 → `nn.Module` 的注册机制不工作 → `net.parameters()` 是空的 → 训练时 optimizer 什么也更新不了,模型纹丝不动。**这是最隐蔽的 bug 之一**。

### `forward()` 不要自己调
```python
# ❌ 不要这样
net.forward(x)

# ✅ 这样
net(x)   # 会触发 __call__,内部除了 forward 还会跑 hook、autograd 注册等
```

---

## 3️⃣ Conv2d / Linear shape 怎么算

### Conv2d (no padding) 输出尺寸
```
H_out = (H_in - kernel_size) / stride + 1
```

CIFAR-10 案例:
```
输入:        (B, 3, 32, 32)
conv1(5×5):  (B, 6, 28, 28)    # 32 - 5 + 1 = 28
pool(2×2):   (B, 6, 14, 14)    # 28 / 2 = 14
conv2(5×5):  (B, 16, 10, 10)   # 14 - 5 + 1 = 10
pool(2×2):   (B, 16, 5, 5)     # 10 / 2 = 5
flatten:     (B, 400)          # 16 × 5 × 5
fc1:         (B, 120)
fc2:         (B, 84)
fc3:         (B, 10)
```

### 🐛 新手最常报错
`RuntimeError: shapes don't match` —— 99% 是 fc1 的 `in_features` 算错了。
解决方法:在 forward 里临时 `print(x.shape)`,看清楚再写死那个数。

---

## 4️⃣ nn.Xxx vs F.xxx

| | nn.Module 类 | F.xxx 函数 |
|---|---|---|
| 例子 | `nn.ReLU()`, `nn.Conv2d` | `F.relu()`, `F.conv2d()` |
| 用法 | 在 `__init__` 里实例化 | 在 `forward` 里直接调 |
| 适合 | **有参数**的层 | **无参数**的操作 |

**经验法则**:
- 卷积、线性、BN、Dropout(有状态) → 用 `nn.Xxx`
- ReLU、Softmax、flatten(无状态) → `F.xxx` 也行,`nn.ReLU()` 也行,看习惯

---

## 5️⃣ 训练循环的五步(最重要的部分)

```python
optimizer.zero_grad()      # ① 清梯度
outputs  = net(inputs)     # ② forward
loss     = criterion(...)  # ③ 算 loss
loss.backward()            # ④ backward
optimizer.step()           # ⑤ 更新参数
```

### 为什么要 `zero_grad`?
PyTorch 的 `.grad` 是**累加**的(`+=`,不是 `=`)。
- 设计原因:支持"梯度累积"模拟大 batch
- 副作用:每个 batch 必须手动清零,否则梯度无限累加 → 训练发散

### `loss.backward()` 实际在做什么?
- 沿着计算图(从 `loss` 出发反向走)
- 对每个 `requires_grad=True` 的 tensor,把梯度写到它的 `.grad` 属性里
- 跑完之后,模型每个参数都有了梯度

### `optimizer.step()` 干嘛?
读每个参数的 `.grad`,按算法更新参数值(SGD: `p -= lr * p.grad`)。

### ⚠️ `running_loss += loss.item()` 不能省 `.item()`
```python
running_loss += loss          # ❌ loss 是带计算图的 tensor,会一直累积引用,显存爆炸
running_loss += loss.item()   # ✅ 抽出 Python float,断开图
```

---

## 6️⃣ CrossEntropyLoss 的关键细节

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
#                  ↑        ↑
#         (B, num_classes)  (B,) 整数 [0, num_classes)
```

- **输入是 raw logits**(不要 softmax!)
- **label 是整数**(不要 one-hot!)
- 内部 = `log_softmax + NLLLoss`,合并是为了**数值稳定性**

最常见错误:在网络最后加了 `softmax` 再传给 `CrossEntropyLoss` → 训练不动或非常慢。

---

## 7️⃣ 设备管理(device)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)

# 训练时
inputs = inputs.to(device)
labels = labels.to(device)
```

### 黄金法则
**模型在哪个 device,数据就要在哪个 device。** 不一致会报 `Expected all tensors to be on the same device`。

### `.to(device)` 行为差异
- 对 **模型**:in-place 修改,可以写 `net.to(device)` 也可以 `net = net.to(device)`
- 对 **tensor**:**不是 in-place**,必须接住 → `x = x.to(device)`

---

## 8️⃣ 推理模式:`no_grad` vs `eval()`

两个东西,**功能不重叠**,推理时两个都开:

| | `torch.no_grad()` | `net.eval()` |
|---|---|---|
| 作用 | 关 autograd,不建计算图 | 切换 Dropout / BatchNorm 到推理模式 |
| 影响 | 省显存 + 加速 | 影响数值结果 |
| 不开会怎样 | 显存涨、慢 | Dropout 还在随机丢、BN 用 batch 统计 → 结果不稳 |

```python
net.eval()
with torch.no_grad():
    for x, y in testloader:
        ...
```

> 这个教程的小网络没用 Dropout/BN 所以省了 `eval()`,但**养成两个都写的习惯**。

---

## 9️⃣ 保存与加载

```python
# 保存(推荐:只存权重)
torch.save(net.state_dict(), 'cifar10_net.pth')

# 加载
net = Net()                                       # 先构造同样结构的模型
net.load_state_dict(torch.load('cifar10_net.pth'))
net.eval()                                        # 推理前切模式
```

**不推荐** `torch.save(net, ...)`——会把整个 Python 对象 pickle 起来,依赖文件路径和类定义,换环境就崩。

---

## 🔟 这个教程没讲、但实战必备的东西

按重要性排:

1. **lr scheduler**:`optim.lr_scheduler.CosineAnnealingLR` 等,每个 epoch 后 `scheduler.step()`
2. **BatchNorm / LayerNorm**:稳定深层网络训练
3. **Data Augmentation**:`RandomCrop`, `RandomHorizontalFlip`, `RandAugment`
4. **更现代的 optimizer**:`optim.AdamW`(带 decoupled weight decay,你应该熟)
5. **Mixed precision**:`torch.cuda.amp.autocast()` + `GradScaler`,显存减半、速度翻倍
6. **梯度裁剪**:`nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)`
7. **TensorBoard / wandb 日志**:别只 print loss
8. **`pin_memory=True`**:DataLoader 加这个参数,GPU 训练略快

---

## ✅ Checklist:写训练循环时自检

- [ ] 模型和数据在同一个 device 上?
- [ ] `optimizer.zero_grad()` 写了吗?
- [ ] `loss.backward()` 和 `optimizer.step()` 顺序对?
- [ ] 评估时开了 `net.eval()` 和 `torch.no_grad()`?
- [ ] 累计 loss 写的是 `.item()` 还是 tensor?
- [ ] 模型最后一层有没有不小心加 softmax?
- [ ] fc1 的 `in_features` 算对了吗?
- [ ] DataLoader 在 Windows/Jupyter 上 `num_workers` 是不是 0?