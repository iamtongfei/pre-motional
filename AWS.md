# AWS EC2 Management Guide

## One-Time Setup

### 1. Install AWS CLI & SSM Plugin
```bash
brew install awscli
brew install --cask session-manager-plugin
```

### 2. Configure SSO
```bash
aws configure sso
```
Enter when prompted:
- SSO session name: `pnnl`
- SSO start URL: `https://pnnl.awsapps.com/start`
- SSO region: `us-west-2`
- Default region: `us-west-2`
- Default output format: `json`
- Profile name: `pnnl`
Select role: `upgrade-gpc-ne-GuestPowerUser`

---

## Daily Usage

### Refresh Login (when token expires every 8-12 hours)
```bash
aws sso login --profile pnnl
```

---

## Instance Info

### Check instance state
```bash
aws ec2 describe-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl \
  --query 'Reservations[0].Instances[0].State.Name'
```

### Check instance type & memory
```bash
aws ec2 describe-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl \
  --query 'Reservations[0].Instances[0].InstanceType'
```

---

## Start / Stop

### Start instance
```bash
aws ec2 start-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

### Stop instance
```bash
aws ec2 stop-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

### Wait until instance is running
```bash
aws ec2 wait instance-running \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

### Wait until instance is stopped
```bash
aws ec2 wait instance-stopped \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

---

## Change Instance Type (CPU / Memory)

> Instance must be STOPPED first. Memory and CPU are tied to instance type.

### Step 1 — Stop
```bash
aws ec2 stop-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

### Step 2 — Wait until stopped
```bash
aws ec2 wait instance-stopped \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

### Step 3 — Change instance type
```bash
aws ec2 modify-instance-attribute \
  --instance-id i-033be867bfca382d2 \
  --instance-type '{"Value":"g4dn.xlarge"}' \
  --region us-west-2 \
  --profile pnnl
```

### Step 4 — Start again
```bash
aws ec2 start-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

### Common Instance Types (GPU)
| Type | GPU | vCPU | Memory |
|------|-----|------|--------|
| `g4dn.xlarge` | 1x T4 | 4 | 16 GB |
| `g4dn.2xlarge` | 1x T4 | 8 | 32 GB |
| `g4dn.4xlarge` | 1x T4 | 16 | 64 GB |
| `g4dn.8xlarge` | 1x T4 | 32 | 128 GB |
| `g5.xlarge` | 1x A10G | 4 | 16 GB |
| `g5.2xlarge` | 1x A10G | 8 | 32 GB |
| `p3.2xlarge` | 1x V100 | 8 | 61 GB |
| `p3.8xlarge` | 4x V100 | 32 | 244 GB |

---

## Connect to Instance

> Instance must be RUNNING first.

### Open terminal session (Terminal 1)
```bash
aws ssm start-session \
  --target i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

### Stop instance after work (Terminal 2 — new tab)
```bash
aws ec2 stop-instances \
  --instance-ids i-033be867bfca382d2 \
  --region us-west-2 \
  --profile pnnl
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Refresh login | `aws sso login --profile pnnl` |
| Check state | `aws ec2 describe-instances ... --query '...State.Name'` |
| Start | `aws ec2 start-instances ...` |
| Stop | `aws ec2 stop-instances ...` |
| Connect | `aws ssm start-session --target i-033be867bfca382d2 ...` |
| Change type | Stop → modify-instance-attribute → Start |

---

## Key Info
- **Instance ID:** `i-033be867bfca382d2`
- **Region:** `us-west-2`
- **Profile:** `pnnl`
- **SSO URL:** `https://pnnl.awsapps.com/start`
- **Account:** `186695067702`

---

## PyTorch Practice Example

**Tutorial:** https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

### Install dependencies (inside SSM session)
```bash
pip3 install torch torchvision boto3 --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU
```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## S3 — Save & Load Model

> EBS volume persists across stop/start, but use S3 to keep model weights safe across instance rebuilds or to share between machines.

### Create a bucket (one-time)
```bash
aws s3 mb s3://YOUR-BUCKET-NAME --region us-west-2 --profile pnnl
```

### Save model to S3 (after training)
```python
import boto3
import torch

torch.save(net.state_dict(), 'cifar10_model.pth')

s3 = boto3.client('s3', region_name='us-west-2')
s3.upload_file('cifar10_model.pth', 'YOUR-BUCKET-NAME', 'models/cifar10_model.pth')
print("Model saved to S3")
```

### Load model from S3
```python
s3 = boto3.client('s3', region_name='us-west-2')
s3.download_file('YOUR-BUCKET-NAME', 'models/cifar10_model.pth', 'cifar10_model.pth')
net.load_state_dict(torch.load('cifar10_model.pth'))
net.eval()
```
