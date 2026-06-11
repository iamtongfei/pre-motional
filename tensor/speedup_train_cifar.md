# CIFAR-10 Training Speedup Guide

## Baseline (Current Setup)

| Setting | Value |
|---|---|
| Instance | g4dn.xlarge (NVIDIA T4, 16 GB VRAM, 4 vCPU, 16 GB RAM) |
| Model | ResNet-18 (CIFAR-adapted) |
| Batch size | 128 |
| DataLoader workers | 2 |
| Precision | FP32 (default) |
| Epochs | 100 |
| **Total time** | **~15–20 min** (~9–12 sec/epoch) |

---

## Speedups — Ranked by Impact

### 1. Mixed Precision (AMP) — `torch.cuda.amp`

**Expected speedup: 30–50% faster** → ~8–12 min total

The T4 has Tensor Cores that accelerate FP16 math. AMP automatically casts forward pass ops to FP16 and keeps the loss scaler in FP32 for stability.

```python
# Add at top of script
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Inside training loop — replace the inner block:
with autocast():
    loss = criterion(net(inputs), labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

**Always solution?** Yes — universally beneficial on any modern NVIDIA GPU (Volta, Turing, Ampere, Ada). The only caveat: if your model has numerical instability, the loss scaler handles it. No accuracy loss expected.

---

### 2. `cudnn.benchmark = True`

**Expected speedup: 5–15% faster on first few epochs, then sustained**

Tells cuDNN to auto-select the fastest convolution algorithm for your input shapes. Since CIFAR-10 inputs are fixed at 32×32, it benchmarks once and reuses.

```python
# Add right after device setup
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
```

**Always solution?** Yes — safe whenever input shapes are fixed throughout training. If input sizes vary (e.g., variable-length sequences), it wastes time re-benchmarking and should be disabled.

---

### 3. `pin_memory=True` + More Workers

**Expected speedup: 5–10% faster**

`pin_memory` allocates CPU tensors in pinned (page-locked) memory so GPU transfers are faster. More workers reduces CPU data loading bottleneck.

```python
# Current (cifar10_v2.py:88-89)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Change to:
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)
```

**Always solution?**
- `pin_memory=True`: always helpful when training on GPU. On CPU-only training it does nothing.
- `num_workers=4`: this-instance specific — g4dn.xlarge has exactly 4 vCPUs, so 4 workers is the sweet spot. On a 2-vCPU machine you'd stick with 2.
- `persistent_workers=True`: general; avoids re-spawning workers every epoch.

---

### 4. `zero_grad(set_to_none=True)`

**Expected speedup: 1–3% faster**

Setting grads to `None` instead of zero skips the memset operation and slightly speeds up the next backward pass.

```python
# Current (cifar10_v2.py:119)
optimizer.zero_grad()

# Change to:
optimizer.zero_grad(set_to_none=True)
```

**Always solution?** Yes — no downsides, works everywhere. Small gain but free.

---

### 5. Larger Batch Size (128 → 256 or 512)

**Expected speedup: 15–25% faster** (more GPU utilization, fewer optimizer steps)

The T4 has 16 GB VRAM — CIFAR-10 images are tiny (32×32×3), so ResNet-18 easily fits batch_size=512 with room to spare.

```python
BATCH_SIZE = 256   # safe, ~same accuracy
# or
BATCH_SIZE = 512   # faster, scale LR proportionally
```

If you push to 512, scale your learning rate linearly to maintain convergence:
```python
# Linear scaling rule: new_lr = base_lr * (new_batch / base_batch)
optimizer = optim.SGD(net.parameters(), lr=0.1 * (512/128), momentum=0.9, weight_decay=5e-4)
# = lr=0.4 with batch_size=512
```

**Always solution?** No — this-case specific in two ways:
1. Only works if the model + batch fits in VRAM (it does here).
2. Very large batches can hurt generalization (the "generalization gap" problem). For CIFAR-10 with ResNet-18, 256–512 is fine. For more complex tasks, test accuracy before committing.

---

### 6. `torch.compile` (PyTorch 2.0+)

**Expected speedup: 10–30% faster after compilation overhead**

Traces and compiles the model into optimized kernels. One-time compilation cost (~30–60 sec) on the first epoch, then every subsequent epoch is faster.

```python
# After net = net.to(device), add:
net = torch.compile(net)
```

**Always solution?** Mostly — works well on PyTorch 2.0+ with CUDA. Exceptions:
- If you're running many short training jobs, the compile overhead isn't worth it.
- Some custom ops aren't supported and fall back to eager mode silently.
- Adds ~200 MB peak memory. Fine on T4 for this model.

---

### 7. Upgrade Instance Type (Infra-level)

**Not a code change — a launch decision.**

| Instance | GPU | FP32 TFLOPS | FP16 TFLOPS | Cost (on-demand) | Cost (spot) | Relative Speed |
|---|---|---|---|---|---|---|
| g4dn.xlarge | T4 | 8.1 | 65 | $0.53/hr | ~$0.18/hr | 1× (baseline) |
| g4dn.2xlarge | T4 | 8.1 | 65 | $0.75/hr | ~$0.27/hr | ~1× (same GPU, more CPU) |
| p3.2xlarge | V100 | 14 | 112 | $3.06/hr | ~$0.90/hr | ~1.7× |
| p3.8xlarge | 4× V100 | 56 | 450 | $12.24/hr | ~$4/hr | ~6× (with DDP) |
| g5.xlarge | A10G | 31.2 | 125 | $1.01/hr | ~$0.35/hr | ~2× |

**For 100-epoch CIFAR-10 training:**
- g5.xlarge is the best value upgrade: ~2× faster, only ~2× more expensive than g4dn spot.
- p3.2xlarge gives 1.7× speed but costs 5× more on-demand.

**Always solution?** No — this is an economic tradeoff, not a code fix. Only worth it if training time is a blocker and your budget allows. For a 15-min job that runs occasionally, this is overkill.

---

### 8. Evaluate Less Frequently

**Expected speedup: 5–10% faster** (reduces eval overhead)

Current code evaluates every 10 epochs (10 times total). Evaluation on CIFAR-10 test set (~79 batches at batch_size=128) takes time that could be training time.

```python
# Current (cifar10_v2.py:127)
if (epoch + 1) % 10 == 0:

# Change to evaluate every 20 epochs (5 evals instead of 10):
if (epoch + 1) % 20 == 0:
```

Or skip eval entirely during training and evaluate once at the end — fine if you just want the final accuracy.

**Always solution?** No — this-case specific. You trade observability for speed. For debugging or hyperparameter tuning you want frequent evals. For a final training run where you just want the model, less frequent is fine.

---

## Combined Effect

Applying all code-level changes together (AMP + cudnn.benchmark + pin_memory/workers + set_to_none + compile):

| Scenario | Estimated Time | Notes |
|---|---|---|
| Baseline (current) | 15–20 min | FP32, batch=128, 2 workers |
| + AMP only | 8–12 min | Biggest single win |
| + AMP + cudnn.benchmark + pin_memory + workers | 6–9 min | All easy code changes |
| + All above + batch_size=256 | 4–7 min | Safe accuracy |
| + All above + torch.compile | 3–6 min | After first-epoch compile cost |
| Upgrade to g5.xlarge + all code changes | 2–4 min | Infra + code combined |

---

## Quick Copy-Paste: Recommended Config Block

Paste this at the top of `cifar10_v2.py` right after imports, replacing the device/model setup section:

```python
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

# Speedups
cudnn.benchmark = True
BATCH_SIZE = 256   # up from 128

# DataLoaders (update both)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)

# After net = net.to(device):
net = torch.compile(net)   # PyTorch 2.0+ only

# AMP scaler
scaler = GradScaler()

# Training inner loop — replace optimizer block:
optimizer.zero_grad(set_to_none=True)
with autocast():
    loss = criterion(net(inputs), labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Summary Table

| Approach | Speedup | Always? | Risk to Accuracy |
|---|---|---|---|
| Mixed Precision (AMP) | 30–50% | Yes | None (with scaler) |
| `cudnn.benchmark = True` | 5–15% | Yes (fixed input size) | None |
| `pin_memory=True` | 3–7% | Yes (GPU training) | None |
| More workers (2→4) | 3–7% | This instance (4 vCPU) | None |
| `persistent_workers=True` | 2–5% | Yes | None |
| `zero_grad(set_to_none=True)` | 1–3% | Yes | None |
| Larger batch (128→256) | 15–25% | No (VRAM + accuracy dep.) | Minor |
| Larger batch (128→512) | 20–30% | No | Needs LR scaling |
| `torch.compile` | 10–30% | Mostly (PyTorch 2.0+) | None |
| Less frequent eval | 5–10% | No (trades observability) | None (eval only) |
| Upgrade to g5.xlarge | ~2× | No (cost tradeoff) | None |
| Upgrade to p3.2xlarge | ~1.7× | No (expensive) | None |
