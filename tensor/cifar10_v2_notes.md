# ResNet-18 CIFAR-10 训练脚本 · 深度笔记

> 三个核心问题:
> 1. **每步在干什么 + Tensor 怎么变化**
> 2. **哪些代码是固定模板 / 哪些是这里 specific**
> 3. **数值改了会怎样**

---

## 📐 第一部分:Tensor Shape 全程追踪

PyTorch 的核心心法:**写代码时脑子里要时刻有 shape**。

### 数据 pipeline 中的 shape 变化

```
┌──────────────────────────────────────────────────────────────┐
│  阶段              │  Shape             │  dtype     │  含义   │
├──────────────────────────────────────────────────────────────┤
│  原始 PIL 图片     │  H=32, W=32, C=3   │  uint8     │ HWC 顺序│
│  ToTensor() 后     │  (3, 32, 32)       │  float32   │ CHW + [0,1]│
│  Normalize() 后    │  (3, 32, 32)       │  float32   │ ~N(0,1) │
│  DataLoader batch  │  (128, 3, 32, 32)  │  float32   │ + B 维度│
│  labels            │  (128,)            │  int64     │ 0~9     │
└──────────────────────────────────────────────────────────────┘
```

**记住四个维度的顺序:`(B, C, H, W)`** — Batch, Channel, Height, Width。
这是 PyTorch 全局约定,所有 conv / pool / BN 都假设这个顺序。
(对比:TensorFlow 默认 `(B, H, W, C)`,NHWC)

### 模型内部的 shape 变化(改造版 ResNet-18)

```
输入 (128, 3, 32, 32)
   │
   ▼
conv1  Conv2d(3→64, 3×3, s=1, pad=1)       ┐
       (128, 64, 32, 32)                    │  Stem (头)
bn1    BatchNorm2d(64)                      │  原版会把 32→8,
relu   ReLU                                 │  改造后保持 32
maxpool nn.Identity()  ← 原版是 MaxPool     ┘
       (128, 64, 32, 32)
   │
   ▼
layer1 (2× BasicBlock, 64ch, stride=1)      → (128, 64,  32, 32)
layer2 (2× BasicBlock, 128ch, 第一个 s=2)   → (128, 128, 16, 16)
layer3 (2× BasicBlock, 256ch, 第一个 s=2)   → (128, 256,  8,  8)
layer4 (2× BasicBlock, 512ch, 第一个 s=2)   → (128, 512,  4,  4)
   │
   ▼
avgpool  AdaptiveAvgPool2d((1,1))           → (128, 512, 1, 1)
flatten                                      → (128, 512)
fc       Linear(512 → 10)                   → (128, 10)   ← logits
```

**关键观察**:
- 每个 stage 通道翻倍、空间减半 → "信息总量"大致守恒,但表示越来越抽象
- 最后 fc 之前的特征是 **512 维**,这是 ResNet-18 的"瓶颈宽度"
- AdaptiveAvgPool2d 是黄金设计:无论输入空间多大,输出永远 (1,1) → 模型适应任意输入尺寸

### Loss 计算时的 shape

```python
outputs = net(inputs)        # (128, 10) — 10 个 raw logit
labels  = ...                # (128,)    — 整数标签
loss    = criterion(outputs, labels)
                             # () — scalar tensor,带计算图
```

⚠️ **CrossEntropyLoss 的契约**:
- 输入必须是 **raw logits**(不要 softmax)
- 标签必须是 **整数索引**(不要 one-hot)
- 内部会自己做 `log_softmax + nll_loss`,数值更稳

---

## 🎯 第二部分:固定 vs Specific 对照

### A. [固定] 所有 PyTorch 训练脚本都长这样

这套骨架你在所有 PyTorch 代码里都会看到,死记硬背就行。

#### A.1 训练五步循环
```python
optimizer.zero_grad()                  # ① 清梯度
outputs = net(inputs)                  # ② forward
loss = criterion(outputs, labels)      # ③ 算 loss
loss.backward()                        # ④ backward
optimizer.step()                       # ⑤ 更新参数
```

#### A.2 train/eval 模式切换
```python
net.train()                            # 训练 epoch 开始前
net.eval()                             # 评估前
```
影响 **Dropout** 和 **BatchNorm** 的行为。即使当前网络没用它们,也养成习惯。

#### A.3 推理时关 autograd
```python
with torch.no_grad():
    outputs = net(images)
```

#### A.4 数据/模型同 device
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```

#### A.5 评估常用 idiom
```python
_, predicted = torch.max(outputs, 1)              # argmax over class dim
correct += (predicted == labels).sum().item()      # bool tensor → 累计 int
```

#### A.6 保存权重
```python
torch.save(net.state_dict(), 'model.pth')
```

#### A.7 scheduler.step() 的位置
**每个 epoch 末**调一次(注意不是每个 batch!除了 OneCycleLR 之类的特殊 scheduler)。

---

### B. [Specific] 这个脚本特有的部分

如果你换任务/换数据集/换网络,这些都要改。

#### B.1 Normalize 的 mean/std
```python
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
```
这是 **CIFAR-10 训练集的真实 RGB 通道统计量**,不是 magic number。
- ImageNet:`(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`
- MNIST:`(0.1307,)` / `(0.3081,)`(单通道)
- 自己的数据集:自己算一遍

#### B.2 RandomCrop + HorizontalFlip
是 **CIFAR / ImageNet 类自然图像**的标配,但:
- 文档/数字识别 → **绝不能** HorizontalFlip(6 翻成 9 是灾难)
- 医学影像 → 翻转规则要看模态(放射科有时可以,病理通常不行)
- 你做的轨迹预测 → 整套 augmentation 都得重新设计

#### B.3 conv1 + maxpool 的改造 ⭐ 最重要的"specific"
```python
net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
net.maxpool = nn.Identity()
```

**为什么必须改?** ResNet-18 原本是为 **ImageNet 224×224** 设计的。
| 阶段 | ImageNet 版 (224 输入) | 如果直接喂 CIFAR (32 输入) |
|---|---|---|
| conv1 (7×7, s=2) | 224 → 112 | 32 → 16 |
| maxpool (3×3, s=2) | 112 → 56 | 16 → 8 |
| 之后 4 个 stage 还要再缩 8 倍 | 56 → 7 (合理) | **8 → 1** (信息基本丢光) |

改成 3×3 / s=1 + Identity 之后,32×32 进 layer1,最后留 4×4,**信息更充足**。

**几乎所有"ResNet for CIFAR"的论文复现代码都做这个改造**(He Kaiming 原论文的 CIFAR 实验用的也是更小的 ResNet,不是 ResNet-18)。

#### B.4 num_classes=10
跟着任务走。CIFAR-100 就是 100,你的轨迹预测可能根本不是分类。

#### B.5 S3 上传
跟基础设施走,本地实验不需要。

---

### C. [可调] 超参数 —— 改不改取决于实验

这些不是"必须这样",但是经过千万人验证的"省心默认值"。

| 超参数 | 这里的值 | 为什么常这样 |
|---|---|---|
| `lr=0.1` (SGD) | 0.1 | ResNet+CIFAR+SGD 黄金值,论文同款 |
| `momentum=0.9` | 0.9 | 业界标准,几乎没人改 |
| `weight_decay=5e-4` | 5e-4 | ResNet 系列常用,Adam 系列常用 1e-2 + AdamW |
| `BATCH_SIZE=128` | 128 | 单卡显存够,且大到梯度估计稳定 |
| `EPOCHS=100` | 100 | ResNet-18+CIFAR 达 ~94% 的够用预算 |
| `T_max=EPOCHS` | 100 | "完整 cosine 周期 = 训练总长" |

---

## 📊 第三部分:数值变了会怎样

### `BATCH_SIZE = 128`

| 改成 | 效果 |
|---|---|
| **32** (变小) | 显存占用小,但梯度噪声大,通常 lr 也要按比例调小 → `lr * (32/128)` |
| **256** (变大) | 显存翻倍可能爆 OOM;梯度更稳但泛化可能略差(大 batch 容易陷入 sharp minima) |
| **8192** (超大) | 需要 warmup + 多卡训练。**线性 lr scaling rule**: lr 跟着 batch 同比例放大 |

🧠 **直觉**: batch_size 影响"梯度估计的方差"。大 batch ≈ 更接近真实梯度 → 可以用更大 lr 但容易过拟合。

### `lr = 0.1`

| 改成 | 效果 |
|---|---|
| **0.001** | 训练慢得离谱,可能跑完 100 ep 才到 80% |
| **0.01** | 还能学,但收敛慢,可能要 200+ ep |
| **0.1** | ✅ ResNet+SGD+momentum 的甜区 |
| **1.0** | **几乎必然发散** (loss → NaN) |

🧠 **直觉**: lr 是"每一步走多远"。SGD 的最优 lr 通常是 Adam 的 100~1000 倍,因为 Adam 内部自己做自适应缩放。

### `momentum = 0.9`

控制"梯度的历史影响"。`v_t = 0.9 * v_{t-1} + grad`。

| 改成 | 效果 |
|---|---|
| **0** | 退化成纯 SGD,在 ravine(峡谷地形)里慢 |
| **0.5** | 介于两者 |
| **0.9** | ✅ 标准值 |
| **0.99** | 太大,反应迟钝,可能震荡 |

### `weight_decay = 5e-4`

L2 正则强度。在每步更新里偷偷加 `param -= lr * wd * param`,逼参数往 0 缩。

| 改成 | 效果 |
|---|---|
| **0** | 无正则,容易过拟合(train acc 飙高、test acc 平) |
| **1e-4** | 弱正则 |
| **5e-4** | ✅ ResNet 标准 |
| **1e-2** | 强正则,可能 underfit |

⚠️ 用 **Adam** 时 `weight_decay` 实现有 bug → 用 **AdamW**(decoupled weight decay)。

### `EPOCHS = 100`

| 改成 | 效果 |
|---|---|
| **10** | 太少,模型还没收敛,可能 ~75% |
| **100** | ✅ 达 ~93-94% |
| **200** | 加点头,可能 95% |
| **1000** | 边际收益极低,且 cosine schedule lr 早就降到 0 了 |

⚠️ 改 `EPOCHS` 时记得同步改 `T_max`,否则 schedule 不匹配。

### `RandomCrop(32, padding=4)` 的 padding

| 改成 | 效果 |
|---|---|
| **0** (无 padding) | 退化成 identity (32→32),没augmentation 效果 |
| **2** | 弱平移扰动 |
| **4** | ✅ CIFAR 经典 |
| **8** | 太狠,可能裁掉物体主体 |

### `num_workers = 2`

数据加载并行度。和模型/loss 无关,只影响速度。

| 平台 | 建议值 |
|---|---|
| Linux + 命令行 | 2~8(看 CPU 核数) |
| Windows | 0(`num_workers>0` 经常崩) |
| Jupyter Notebook | 0(经常崩) |
| 数据本来就在 RAM | 0(workers 反而是 overhead) |

### `evaluate every 10 epochs`

**节省时间**。完整 testset 评估 1 次 ≈ 一个 epoch 训练时间的 1/5。
做最终 paper 实验时,通常每个 epoch 都测,以便画收敛曲线。

---

## 🧠 第四部分:Tensor 常用操作速查

这个脚本里出现的 tensor 操作,值得记住:

### `tensor.to(device)`
```python
x = x.to(device)              # ⚠️ 必须接住返回值!不是 in-place
```
对 **模型** in-place 也行;对 **tensor** 必须接住。

### `tensor.item()`
```python
running_loss += loss.item()
```
- 0-d tensor → Python scalar
- 顺便**脱离计算图**,不会再累积梯度引用

### `torch.max(tensor, dim)`
```python
values, indices = torch.max(outputs, 1)
# outputs: (B, 10) → values: (B,)  indices: (B,)
```
返回 `(最大值, 最大值的 index)` 的 tuple。

### `(predicted == labels).sum().item()`
- `predicted == labels` → bool tensor,shape `(B,)`
- `.sum()` → 0-d int tensor
- `.item()` → Python int

### `tensor.size(dim)` vs `tensor.shape[dim]`
完全等价,前者是方法,后者是属性。`labels.size(0)` = batch 大小。

### Tensor in-place 操作的 `_` 后缀
```python
x.add_(1)      # in-place: x = x + 1
x.zero_()      # in-place 置零
```
**梯度计算中要小心 in-place**,可能破坏计算图。

---

## ⚙️ 第五部分:这个脚本 vs 上一个(对比看进化)

| 维度 | 第一版 (LeNet) | 现在这版 (ResNet-18) |
|---|---|---|
| 模型 | 自己定义 5 层小网络 | torchvision 内置 ResNet + 改造 |
| 参数量 | ~60K | ~11M |
| BatchSize | 4 | 128 |
| 学习率 | 0.001 | 0.1 (+ momentum + wd) |
| Augmentation | 无 | RandomCrop + Flip |
| Scheduler | 无 | CosineAnnealing |
| 评估 | 训练结束才测 | 每 10 ep 测一次 |
| 保存 | 只本地 | 本地 + S3 |
| 预期准确率 | ~55% | ~94% |

→ **多出来的复杂度全都换成了准确率**。每个组件都不是装饰。

---

## ✅ 写训练脚本的自检清单(升级版)

- [ ] 数据 augmentation 只用在训练集,测试集不用
- [ ] Normalize 用的是**当前数据集**的统计量
- [ ] 预训练模型(torchvision)用在小图上时,**改 stem**
- [ ] 训练前 `net.train()`,评估前 `net.eval()` + `torch.no_grad()`
- [ ] 五步循环顺序对(`zero_grad → forward → loss → backward → step`)
- [ ] `scheduler.step()` 在 epoch 末,不是 batch 末
- [ ] `loss.item()` 别忘
- [ ] EPOCHS 改了时,`T_max` 跟着改
- [ ] `weight_decay` 别忘(Adam 优化器要用 AdamW)
- [ ] Windows / Jupyter 上 `num_workers=0`