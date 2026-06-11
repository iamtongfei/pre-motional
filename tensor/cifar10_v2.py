"""
================================================================================
PyTorch CIFAR-10 + ResNet-18 训练脚本 (现代 template)
每行带:① 功能解释  ② Tensor shape  ③ 固定/Specific 标注
================================================================================

标注约定:
  [固定]    = 所有 PyTorch 训练脚本都这样写
  [Specific] = 因为是 CIFAR-10 / ResNet-18 / 这个任务才这样
  [可调]    = 超参数,值可以变
"""

# ============================================================
# Import
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models      # 内置预定义模型 (ResNet, VGG, ViT, ...)
import boto3                              # AWS SDK,这里用来上传模型到 S3 [Specific]


# ============================================================
# Config 区 —— [可调] 超参数集中管理是好习惯
# ============================================================
EPOCHS     = 100        # [可调] 训练轮数。ResNet-18 + CIFAR-10 通常 100~200 ep 达 ~94-95%
BATCH_SIZE = 128        # [可调] batch 大小。8GB GPU 跑 ResNet-18+CIFAR 上限 ~256
S3_BUCKET  = 'pnnl-s3'  # [Specific] 你的 S3 bucket 名
MODEL_PATH = 'cifar10_v2.pth'


# ============================================================
# 1. 数据预处理 + Augmentation
# ============================================================

# 训练集 transform: 加入 augmentation [Specific: CIFAR 常用组合]
transform_train = transforms.Compose([
    # RandomCrop(size, padding): 先 padding 4 像素 (32→40),再随机裁出 32x32
    # 效果: 模拟"图像平移",训练更鲁棒。padding=4 是 CIFAR 黄金值。
    transforms.RandomCrop(32, padding=4),

    # 50% 概率水平翻转。猫狗鸟都左右对称无所谓。[Specific: 数字识别就不能开!]
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),               # [固定] PIL → Tensor, [0,255] → [0,1], HWC → CHW

    # CIFAR-10 训练集的真实通道统计量 [Specific to CIFAR-10]
    # 不同数据集这个数字不同。ImageNet 是 (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 测试集 transform: ⚠️ 绝对不能有 augmentation
# augmentation 是为了让模型见过更多变体,测试时要看"真实"图片的预测
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Dataset: 单样本 shape = (3, 32, 32),返回 (image_tensor, label_int)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# DataLoader: 每次 yield (batch_images, batch_labels)
#   batch_images shape: (128, 3, 32, 32)
#   batch_labels shape: (128,)  → 整数 0~9
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ============================================================
# 2. 模型: ResNet-18 改造成 CIFAR 版
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # [固定]
print(f"Using device: {device}")

# 从 torchvision 拿一个标准 ResNet-18,把最后一层 fc 自动改成输出 10 类
# ⚠️ 这是 ImageNet 版的 ResNet-18,默认设计用于 224×224 输入,32×32 直接用会浪费分辨率!
net = models.resnet18(num_classes=10)

# ↓↓↓ 关键改造 (CIFAR-specific) ↓↓↓
# 原 conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  → 输出 16×16,信息丢一半
# 改成   : 3×3 / stride=1 / padding=1                          → 输出 32×32,保留分辨率
net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

# 原 maxpool: MaxPool2d(3, stride=2)  → 又把 32 缩到 8,太狠了
# 改成 Identity (什么都不做),保持 32×32 进 layer1
net.maxpool = nn.Identity()
# ↑↑↑ 几乎所有 CIFAR ResNet 实现都做这两步改造 ↑↑↑

net = net.to(device)   # [固定] 把所有参数搬到 GPU

# ======== Tensor shape 流向 (B=128) ========
#   输入             (128, 3,   32, 32)
#   conv1 (3×3,s=1)→ (128, 64,  32, 32)
#   bn1, relu      → (128, 64,  32, 32)
#   maxpool=Identity (128, 64,  32, 32)
#   layer1 (2×Block, 64ch, stride 1)  → (128, 64,  32, 32)
#   layer2 (2×Block, 128ch, stride 2) → (128, 128, 16, 16)
#   layer3 (2×Block, 256ch, stride 2) → (128, 256,  8,  8)
#   layer4 (2×Block, 512ch, stride 2) → (128, 512,  4,  4)
#   avgpool (Adaptive 1×1)            → (128, 512,  1,  1)
#   flatten + fc                      → (128, 10)   ← logits


# ============================================================
# 3. Loss + Optimizer + Scheduler
# ============================================================

criterion = nn.CrossEntropyLoss()
# [固定] 分类任务标配。吃 (B, num_classes) logits + (B,) int 标签

optimizer = optim.SGD(
    net.parameters(),     # [固定] 把所有参数交给 optimizer 管
    lr=0.1,               # [可调] SGD+momentum 在 ResNet+CIFAR 上的经典初值
    momentum=0.9,         # [可调] 业界标准值,极少改动
    weight_decay=5e-4     # [可调] L2 正则,经典值 1e-4 ~ 5e-4
)

# CosineAnnealingLR: lr 从 0.1 余弦式下降到 ~0,在 T_max 步结束
# T_max = EPOCHS 表示"完整 cosine 周期长度 = 总训练长度"
# [可调] 也可以用 MultiStepLR / OneCycleLR / WarmupCosine
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ============================================================
# 4. 训练循环 —— 这部分 95% 固定
# ============================================================
for epoch in range(EPOCHS):
    net.train()           # [固定] 切训练模式 (Dropout/BN 用 batch 统计)
    running_loss = 0.0

    for inputs, labels in trainloader:
        # inputs: (128, 3, 32, 32),  labels: (128,)
        inputs, labels = inputs.to(device), labels.to(device)   # [固定]

        optimizer.zero_grad()                # [固定] ① 清梯度
        loss = criterion(net(inputs), labels)# [固定] ② forward + ③ 算 loss
                                             #   net(inputs) → (128, 10)
                                             #   loss        → scalar
        loss.backward()                      # [固定] ④ backward
        optimizer.step()                     # [固定] ⑤ 更新参数

        running_loss += loss.item()          # [固定] .item() 把 0-d tensor 转 Python float

    scheduler.step()      # [固定] 每个 epoch 末调一次,推进 lr schedule

    # 每 10 个 epoch 评估一次 [可调:节省时间的做法]
    if (epoch + 1) % 10 == 0:
        net.eval()                           # [固定] 切推理模式
        correct, total = 0, 0
        with torch.no_grad():                # [固定] 关 autograd,省显存+加速
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                # net(images) → (128, 10)
                # torch.max(..., 1): 沿 dim=1 (类别维) 取 argmax
                # _ 接住 max 值不用; predicted shape = (128,) int
                _, predicted = torch.max(net(images), 1)
                total   += labels.size(0)                              # 累计样本数
                correct += (predicted == labels).sum().item()          # 累计正确数
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {running_loss/len(trainloader):.3f} | Acc: {acc:.1f}%")

print("Training complete!")


# ============================================================
# 5. 保存 + 上传 S3
# ============================================================

# [固定] 只存 state_dict (权重),不存整个模型对象
torch.save(net.state_dict(), MODEL_PATH)
print(f"Model saved locally: {MODEL_PATH}")

# [Specific] AWS S3 上传,需要本机配好 AWS credentials (~/.aws/credentials)
s3 = boto3.client('s3', region_name='us-west-2')
s3.upload_file(MODEL_PATH, S3_BUCKET, f'models/{MODEL_PATH}')
print(f"Model uploaded to s3://{S3_BUCKET}/models/{MODEL_PATH}")