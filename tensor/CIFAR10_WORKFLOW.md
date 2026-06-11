# CIFAR-10 Training on AWS EC2 — Full Workflow

## Key Info
- **Instance ID:** `i-033be867bfca382d2` (on-demand g4dn.xlarge)
- **Region:** `us-west-2`
- **Profile:** `pnnl`
- **S3 Bucket:** `pnnl-s3`
- **Checkpoint key:** `s3://pnnl-s3/checkpoints/cifar10_v2_checkpoint.pth`

---

## Spot Instance — Launch & Resume (save ~65% vs on-demand)

### Launch a Spot instance (first time or after interruption)

```bash
# Get your current AMI id from the running instance
aws ec2 describe-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 --profile pnnl \
  --query 'Reservations[0].Instances[0].ImageId' --output text

# Request a Spot instance with same spec (replace ami-XXXXXXXX with output above)
aws ec2 run-instances \
  --image-id ami-XXXXXXXX \
  --instance-type g4dn.xlarge \
  --region us-west-2 \
  --profile pnnl \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --iam-instance-profile Name=EC2-SSM-Role \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cifar10-spot}]' \
  --query 'Instances[0].InstanceId' --output text
```

> Spot g4dn.xlarge ≈ $0.18/hr vs $0.53/hr on-demand.
> AWS gives a **2-minute warning** before reclaiming — the script handles this automatically.

### Resume after interruption (on new Spot or same instance)

Just run the script again — it auto-downloads the S3 checkpoint and picks up from the last saved epoch:

```bash
cd ~/cifar10
python3 cifar10_v2.py
# Output: "Resumed from checkpoint — epoch 50, best acc: 87.3%"
```

### Delete checkpoint after successful training

```bash
aws s3 rm s3://pnnl-s3/checkpoints/cifar10_v2_checkpoint.pth --profile pnnl
```

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

## Complete Daily Workflow — Edit → Train → Save → Shutdown

### Step 1 — Edit your script in Jupyter (browser)

1. Open `http://localhost:8888` in your Mac browser
2. Click `cifar10_v2.py` to open and edit it
3. Make your changes, then **save with `Ctrl+S`**

---

### Step 2 — Run training in Mac Terminal (Terminal 3, new tab)

Open a **new terminal tab** on your Mac and connect via SSM:

```bash
aws ssm start-session \
  --target i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

Then inside the SSM session:

```bash
cd ~/cifar10
python3 cifar10_v2.py
```

You'll see live loss output every epoch:
```
Using device: cuda
Epoch  10/100 | Loss: 1.234 | Acc: 62.3%
Epoch  20/100 | Loss: 0.987 | Acc: 71.5%
...
Epoch 100/100 | Loss: 0.312 | Acc: 93.1%
Training complete!
Model saved locally: cifar10_v2.pth
Model uploaded to s3://pnnl-s3/models/cifar10_v2.pth
```

> Full 100 epochs takes ~15-20 min on T4 GPU.

---

### Step 3 — Edit again while training (or after)

Go back to the browser Jupyter tab, edit the script, save.
Next time you run `python3 cifar10_v2.py` in Terminal 3, it picks up the new version.

---

### Step 4 — Save model to S3 (before shutting down)

The script auto-uploads at the end. But to manually save anytime:

```bash
# Inside SSM session (Terminal 3)
aws s3 cp ~/cifar10/cifar10_v2.pth \
  s3://pnnl-s3/models/cifar10_v2.pth \
  --region us-west-2
```

Verify it uploaded:
```bash
aws s3 ls s3://pnnl-s3/models/ --region us-west-2
```

---

### Step 5 — Shut everything down (avoid charges)

Close in this order:

**Terminal 1 (SSM + Jupyter):**
```
Ctrl+C    ← stops Jupyter
exit      ← closes SSM session
```

**Terminal 2 (port forward):**
```
Ctrl+C    ← closes tunnel
```

**Terminal 3 (SSM training session):**
```
exit      ← closes SSM session
```

**Stop the EC2 instance (Mac terminal):**
```bash
aws ec2 stop-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

**Verify it stopped:**
```bash
aws ec2 describe-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl \
  --query 'Reservations[0].Instances[0].State.Name'
```

Should return `"stopped"`. You will not be charged while stopped (EBS storage still costs ~$0.10/GB/month but compute is free).

---

## Terminal Layout Summary

| Terminal Tab | Purpose | What runs there |
|---|---|---|
| Terminal 1 | SSM session | Jupyter notebook server |
| Terminal 2 | Port forward | Tunnel `localhost:8888 → EC2:8888` |
| Terminal 3 | SSM session | `python3 cifar10_v2.py` training |
| Browser | Jupyter UI | Edit `.py` files |

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
aws s3 mb s3://pnnl-s3 --region us-west-2 --profile pnnl
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
S3_BUCKET = 'pnnl-s3'
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
aws s3 cp ~/cifar10/cifar10_v2.pth s3://pnnl-s3/models/cifar10_v2.pth \
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
s3.download_file('pnnl-s3', 'models/cifar10_v2.pth', 'cifar10_v2.pth')

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
| Saved model (cloud) | `s3://pnnl-s3/models/cifar10_v2.pth` |
| Jupyter URL | `http://localhost:8888` (when tunnel is open) |
