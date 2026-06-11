"""
================================================================================
PyTorch CIFAR-10 入门教程 (官方 60min Blitz 同款)
逐行中文注释版,适合第一次写 PyTorch 训练循环的人
================================================================================
"""

# ============================================================
# Import 区
# ============================================================
import torch                          # PyTorch 主库:Tensor、autograd、device 都在这里
import torch.nn as nn                 # nn = neural network,所有 layer (Conv2d/Linear) 和 loss 的命名空间
import torch.nn.functional as F       # 函数式 API:激活函数 (relu/softmax) 等"无参数"操作
                                      # 区别:nn.ReLU() 是一个 Module 类,F.relu(x) 是直接调函数
                                      #   有参数的层 (Conv/Linear/BN) → 用 nn.Xxx
                                      #   无参数的操作 (relu/softmax/flatten) → 可以用 F.xxx
import torch.optim as optim           # 优化器:SGD, Adam, AdamW 等都在这里
import torchvision                    # PyTorch 视觉扩展库:数据集 + 预训练模型 + 图像变换
import torchvision.transforms as transforms   # 图像预处理 pipeline (ToTensor / Normalize / Resize / ...)


# ============================================================
# 1. 数据加载与预处理
# ============================================================

# transforms.Compose: 把多个预处理操作串成一个 pipeline,会按顺序应用到每张图
transform = transforms.Compose([
    # ToTensor() 做两件事:
    #   ① PIL Image (H, W, C) → Tensor (C, H, W),即 channel-first
    #   ② 像素值 [0, 255] (uint8) → [0.0, 1.0] (float32),自动除以 255
    transforms.ToTensor(),

    # Normalize(mean, std):对每个 channel 做 (x - mean) / std
    # 这里 mean=std=0.5,所以 [0,1] 被映射到 [-1, 1]
    # 为什么要 normalize? 让输入分布更接近 0 均值、单位方差,梯度下降更稳定
    # 注意:这三个数对应 RGB 三通道,所以是 (0.5, 0.5, 0.5)
    # 实战中常用 ImageNet 的统计量: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# torchvision.datasets.CIFAR10: PyTorch 内置的 CIFAR-10 Dataset 类
#   root: 数据下载/读取目录
#   train=True: 拿训练集 (50000 张); False = 测试集 (10000 张)
#   download=True: 如果 root 里没有就自动下载 (~170MB)
#   transform: 上面定义的预处理 pipeline,每次 __getitem__ 时会自动应用
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

# DataLoader: 把 Dataset 包装成"可迭代的 batch 生成器"
#   batch_size=4: 每次给 4 张图 (官方教程为了演示用的小 batch,实际训练通常 64/128/256)
#   shuffle=True: 每个 epoch 开始前打乱样本顺序 (训练集必开,测试集不开)
#   num_workers=2: 用 2 个子进程并行加载数据 (CPU 加载 + GPU 训练流水线化)
#     ⚠️ Windows 上 num_workers>0 必须把训练代码放进 if __name__ == '__main__':
#     ⚠️ Jupyter 里也容易出问题,如果报错就设成 0
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

# 测试集同理,但 shuffle=False (评估顺序不影响结果,且方便复现)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

# CIFAR-10 的 10 个类别,index 0~9 对应这个 tuple 里的字符串
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


# ============================================================
# 2. 定义网络结构
# ============================================================

# 所有自定义网络都要继承 nn.Module
# nn.Module 提供了 .parameters() / .to(device) / .train()/.eval() / .state_dict() 等方法
class Net(nn.Module):
    def __init__(self):
        # 必须调用父类 __init__,否则 nn.Module 内部的注册机制 (用来追踪 parameters) 不工作
        super().__init__()

        # Conv2d(in_channels, out_channels, kernel_size)
        # 输入 3 通道 (RGB),输出 6 个 feature map,卷积核 5×5
        # 输入 shape:  (batch, 3, 32, 32)
        # 输出 shape:  (batch, 6, 28, 28)   ← 32-5+1=28 (没 padding 默认 valid 卷积)
        self.conv1 = nn.Conv2d(3, 6, 5)

        # MaxPool2d(kernel_size, stride): 2×2 池化,步长 2,空间分辨率减半
        # 注意:同一个 pool 在 forward 里被复用了两次,这是合法的——因为它没有可学习参数
        self.pool  = nn.MaxPool2d(2, 2)

        # 第二个卷积层: 6 → 16 通道,5×5 卷积
        # 经过 conv2 后:(batch, 16, 10, 10) → pool 后 (batch, 16, 5, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Linear(in_features, out_features): 全连接层,等价于 y = xW^T + b
        # 16*5*5 = 400 是 flatten 后的维度,必须手算 (PyTorch 不会自动推断)
        # 这是新手最常报错的地方:shape mismatch
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)   # 输出 10 个 logit,对应 10 个类别

    # forward 定义"数据怎么从输入流到输出"
    # 不要自己调 net.forward(x),而是 net(x) —— 后者会触发 hook 和 autograd 注册
    def forward(self, x):
        # 输入 x: (batch, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))   # → (batch, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # → (batch, 16, 5, 5)

        # flatten(x, 1): 从第 1 维开始展平,保留 batch 维度
        # (batch, 16, 5, 5) → (batch, 400)
        # 也可以写 x.view(x.size(0), -1) 或 x.reshape(...),flatten 更直观
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))    # → (batch, 120)
        x = F.relu(self.fc2(x))    # → (batch, 84)
        return self.fc3(x)         # → (batch, 10),返回 raw logits,不加 softmax
                                   # 因为 CrossEntropyLoss 内部会自己做 log_softmax,
                                   # 这里再加一层 softmax 反而会数值不稳定 + 梯度变小

# 设备选择:有 CUDA 就用 GPU,否则 CPU
# 现代写法还可以加 'mps' (Apple Silicon) 或 'xpu' (Intel) 的判断
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# .to(device) 把模型所有参数搬到目标设备
# 注意:之后送进模型的 input 也必须在同一个 device 上,否则会报错
net = Net().to(device)


# ============================================================
# 3. 训练
# ============================================================

# CrossEntropyLoss = LogSoftmax + NLLLoss 合二为一
# 输入:logits (batch, num_classes) + 整数标签 (batch,)  注意标签不是 one-hot
criterion = nn.CrossEntropyLoss()

# SGD 优化器:net.parameters() 会递归收集模型里所有 nn.Parameter
# momentum=0.9 是经典设置,可以理解为"梯度的指数滑动平均",帮助穿越平坦区
# lr (learning rate) 是最重要的超参,这里 0.001 偏小,实战常用 0.01~0.1 + scheduler
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练 2 个 epoch (一个 epoch = 完整过一遍训练集)
for epoch in range(2):
    running_loss = 0.0

    # enumerate(trainloader) 会给出 (index, batch_data)
    # batch_data 是 (inputs, labels) 的 tuple,直接解构
    for i, (inputs, labels) in enumerate(trainloader):
        # 把这个 batch 的数据搬到 GPU/CPU,和模型保持同一设备
        inputs, labels = inputs.to(device), labels.to(device)

        # ⭐ 关键步骤①:清零梯度
        # PyTorch 的梯度默认是"累加"的 (.grad += new_grad),不清零会一直累加
        # 每个 batch 开始前都要 zero_grad
        optimizer.zero_grad()

        # ⭐ 关键步骤②:前向传播
        outputs = net(inputs)             # (batch, 10) logits

        # ⭐ 关键步骤③:计算 loss
        loss = criterion(outputs, labels) # scalar tensor,带 grad_fn

        # ⭐ 关键步骤④:反向传播
        # autograd 沿着计算图反向走,把每个 parameter 的 .grad 算出来
        loss.backward()

        # ⭐ 关键步骤⑤:参数更新
        # optimizer 根据每个参数的 .grad 和算法 (SGD/Adam/...) 更新参数值
        optimizer.step()

        # .item() 把 0-d tensor 转成 Python float,顺便把它从计算图里"剥离"
        # 不写 .item() 直接 += loss 会让计算图一直累积,显存爆炸
        running_loss += loss.item()

        # 每 2000 个 batch 打印一次平均 loss
        if i % 2000 == 1999:
            print(f'[epoch {epoch+1}, batch {i+1:5d}] loss: {running_loss/2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# 保存模型权重 (state_dict 是一个 OrderedDict: 参数名 → tensor)
# 推荐保存 state_dict 而不是整个模型对象,后者依赖文件路径和类定义
# 加载时: net.load_state_dict(torch.load('cifar10_net.pth'))
torch.save(net.state_dict(), 'cifar10_net.pth')


# ============================================================
# 4. 在测试集上评估准确率
# ============================================================
correct, total = 0, 0

# torch.no_grad(): 上下文管理器,告诉 autograd "这一段不要建计算图"
# 评估/推理时必开,可以省显存 + 加速
# 它和 net.eval() 是两件事:
#   - no_grad():    关掉 autograd
#   - net.eval():   切换 Dropout/BatchNorm 到推理模式
# 严格来讲两个都应该开。这个网络没用 Dropout/BN 所以不写 eval() 也没差,但养成习惯。
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)   # (batch, 10) logits

        # torch.max(tensor, dim): 返回 (最大值, 最大值的 index)
        # dim=1 表示沿着"类别"这个维度求 max
        # 我们只关心 argmax (predicted class),所以最大值用 _ 接住扔掉
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)                          # 累计样本数
        correct += (predicted == labels).sum().item()    # 累计预测正确数
        # (predicted == labels) → bool tensor;.sum() → 0-d tensor;.item() → Python int

print(f'Accuracy on 10,000 test images: {100 * correct / total:.1f}%')
# 这个小网络 + 2 epoch + lr=0.001,大概能到 ~55% (随机猜是 10%)
# 想冲更高:加深网络、加 BatchNorm、加 data augmentation、lr scheduler、更多 epoch