# CIFAR-10 Training on AWS EC2 — Full Workflow

## Key Info
- **Instance ID:** `i-033be867bfca382d2`
- **Region:** `us-west-2`
- **Profile:** `pnnl`
- **S3 Bucket:** `YOUR-BUCKET-NAME` ← replace this once you create it

---

## Every Time You Start

### Terminal 1 — Start instance & connect

```bash
# 1. Refresh SSO login (expires every 8-12 hrs)
aws sso login --profile pnnl

# 2. Start instance
aws ec2 start-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl

# 3. Wait until running
aws ec2 wait instance-running \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl

# 4. Connect via SSM
aws ssm start-session \
  --target i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

### Inside SSM — Start Jupyter

```bash
cd ~/cifar10
python3 -m notebook --no-browser --port=8888
```

Copy the token from the output — you'll need it in the browser.

---

### Terminal 2 (new tab on Mac) — Port forward

```bash
aws ssm start-session \
  --target i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8888"],"localPortNumber":["8888"]}'
```

### Browser — Open Jupyter

```
http://localhost:8888
```

Paste the token when prompted. Your files at `~/cifar10/` are now visible and editable.

---

## First-Time Setup (do once)

### Inside SSM session

```bash
# Install packages
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install boto3 notebook

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Create working folder
mkdir -p ~/cifar10 && cd ~/cifar10
```

### Create S3 bucket (once, on your Mac)

```bash
aws s3 mb s3://YOUR-BUCKET-NAME --region us-west-2 --profile pnnl
```

---

## The Improved Training Script (v2 — targets ~93% accuracy)

Create `cifar10_v2.py` inside Jupyter or with `nano ~/cifar10/cifar10_v2.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import boto3

# --- Config ---
EPOCHS = 100
BATCH_SIZE = 128
S3_BUCKET = 'YOUR-BUCKET-NAME'
MODEL_PATH = 'cifar10_v2.pth'

# --- Data (with augmentation) ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Model (ResNet-18 adapted for 32x32 CIFAR images) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

net = models.resnet18(num_classes=10)
net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
net.maxpool = nn.Identity()
net = net.to(device)

# --- Optimizer + Scheduler ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- Training loop ---
for epoch in range(EPOCHS):
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(net(images), 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {running_loss/len(trainloader):.3f} | Acc: {acc:.1f}%")

print("Training complete!")

# --- Save locally ---
torch.save(net.state_dict(), MODEL_PATH)
print(f"Model saved locally: {MODEL_PATH}")

# --- Upload to S3 ---
s3 = boto3.client('s3', region_name='us-west-2')
s3.upload_file(MODEL_PATH, S3_BUCKET, f'models/{MODEL_PATH}')
print(f"Model uploaded to s3://{S3_BUCKET}/models/{MODEL_PATH}")
```

---

## Save Model to S3 (manual, anytime)

Inside SSM session or Jupyter terminal:

```bash
aws s3 cp ~/cifar10/cifar10_v2.pth s3://YOUR-BUCKET-NAME/models/cifar10_v2.pth \
  --region us-west-2
```

---

## Load Model from S3 (next session)

```python
import boto3, torch
import torchvision.models as models
import torch.nn as nn

# Download from S3
s3 = boto3.client('s3', region_name='us-west-2')
s3.download_file('YOUR-BUCKET-NAME', 'models/cifar10_v2.pth', 'cifar10_v2.pth')

# Rebuild model and load weights
net = models.resnet18(num_classes=10)
net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
net.maxpool = nn.Identity()
net.load_state_dict(torch.load('cifar10_v2.pth'))
net.eval()
print("Model loaded and ready")
```

---

## Close & Reopen Jupyter

### To close

1. In Terminal 1 (SSM): `Ctrl+C` to stop Jupyter
2. In Terminal 2 (port forward): `Ctrl+C` to stop tunnel
3. Your files and trained model stay on the EC2 EBS disk

### To reopen later

Just repeat the **Every Time You Start** section above. The files are still at `~/cifar10/`.

> **Important:** Always save to S3 before stopping the instance.
> EBS persists across stop/start but NOT if the instance is terminated.

---

## Stop Instance When Done

```bash
aws ec2 stop-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

---

## Quick Reference

| What | Where |
|------|-------|
| Training script | `~/cifar10/cifar10_v2.py` on EC2 |
| CIFAR-10 data | `~/cifar10/data/` on EC2 |
| Saved model (local) | `~/cifar10/cifar10_v2.pth` on EC2 |
| Saved model (cloud) | `s3://YOUR-BUCKET-NAME/models/cifar10_v2.pth` |
| Jupyter URL | `http://localhost:8888` (when tunnel is open) |
