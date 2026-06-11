"""utils/debug.py — PyTorch 实验调试工具

包含 9 个函数,按用途分四组:

[打印]
    banner(title)              分隔横线 + 标题
    show(name, t, head=False)  打印单个 tensor 的 shape/dtype/device
    show_dict(d)               打印 dict 里所有 tensor

[数值检查]
    summarize(t, name)         min/max/mean/std + NaN/Inf 警告
    compare(a, b)              对比两个 tensor 是否一致

[模型检查]
    param_summary(model)       参数总数 / 可训练数 / 冻结数
    grad_summary(model)        backward 后检查每层梯度健康

[性能/复现]
    timer(name)                Context manager 计时 (自动 CUDA sync)
    seed_all(seed=42)          一键设所有随机种子

典型用法:
    from utils.debug import banner, show, summarize, timer, seed_all

    seed_all(42)
    banner("Section 1: Data")
    show("past", past)
    summarize(loss, "loss")
    with timer("forward"):
        out = model(past)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn


# ================================================================
#  1. 打印格式
# ================================================================

def banner(title: str, width: int = 68, char: str = "=") -> None:
    """分隔横线 + 标题,用来标记 section.

    示例:
        >>> banner("Section 1: Data Loading")

        ====================================================================
          Section 1: Data Loading
        ====================================================================
    """
    line = char * width
    print(f"\n{line}\n  {title}\n{line}")


def show(name: str, t: Any, *, head: bool = False, max_show: int = 30) -> None:
    """打印 tensor (或任意对象) 的信息,统一对齐格式.

    Args:
        name: 显示的名字 (会 padding 到 30 字符,自动列对齐)
        t: tensor / 其他对象
        head: True 时额外打印实际值 (仅 numel <= max_show 时)
        max_show: 允许打印数值的元素数上限,防刷屏

    示例:
        >>> show("past", past)
          past                           shape=(2, 3, 4, 2), dtype=torch.float32, device=cpu

        >>> show("best_mode", best_mode, head=True)
          best_mode                      shape=(2, 3), dtype=torch.int64, device=cpu
                                         values=[1, 0, 2, 1, 2, 0]
    """
    name_field = f"{name:30s}"
    if isinstance(t, torch.Tensor):
        info = f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"
        if t.requires_grad:
            info += ", requires_grad=True"
        if t.is_sparse:
            info += ", sparse"
        print(f"  {name_field} {info}")
        if head and t.numel() <= max_show:
            vals = t.detach().flatten().tolist()
            # 浮点数限制小数位避免刷屏
            if t.is_floating_point():
                vals = [round(v, 4) for v in vals]
            print(f"  {' '*30} values={vals}")
    elif isinstance(t, (list, tuple)):
        print(f"  {name_field} {type(t).__name__}(len={len(t)})")
    elif t is None:
        print(f"  {name_field} None")
    else:
        print(f"  {name_field} {type(t).__name__}: {t}")


def show_dict(d: dict, prefix: str = "") -> None:
    """打印 dict 里所有 tensor 的信息,适合看 batch / scene.

    示例:
        >>> show_dict(batch)
          past                           shape=(2, 3, 4, 2), ...
          future_gt                      shape=(2, 3, 6, 2), ...
          agent_mask                     shape=(2, 3), dtype=torch.bool, ...
    """
    for k, v in d.items():
        show(f"{prefix}{k}", v)


# ================================================================
#  2. 数值检查
# ================================================================

def summarize(t: torch.Tensor, name: str = "tensor") -> None:
    """打印 tensor 的统计信息 (min/max/mean/std + NaN/Inf 检查).

    用于调试 silent NaN / 数值不稳定:
        - 训练突然不收敛了 → summarize(loss, ...)
        - loss 变 NaN → summarize 每层 activation / grad
        - 输入归一化没做对 → summarize(inputs, ...)

    示例:
        >>> summarize(loss, "loss")
          loss: shape=(), dtype=torch.float32
                 min=2.4321, max=2.4321, mean=2.4321, std=nan

        >>> summarize(grad, "grad after backward")
          grad after backward: shape=(64, 8), dtype=torch.float32
                 min=-0.0234, max=0.0541, mean=0.0001, std=0.0089
                 ⚠ contains NaN
    """
    if not isinstance(t, torch.Tensor):
        print(f"  {name}: 不是 tensor (类型 {type(t).__name__})")
        return

    t_float = t.detach().float()
    has_nan = bool(torch.isnan(t_float).any().item())
    has_inf = bool(torch.isinf(t_float).any().item())

    # 排除 NaN/Inf 后再算统计,避免污染
    if has_nan or has_inf:
        finite = t_float[torch.isfinite(t_float)]
        if finite.numel() == 0:
            print(f"  {name}: shape={tuple(t.shape)}  ⚠⚠⚠ 全是 NaN/Inf!")
            return
        stats = finite
    else:
        stats = t_float

    info = (
        f"shape={tuple(t.shape)}, dtype={t.dtype}\n"
        f"  {' '*len(name)}  min={stats.min().item():.4g}, "
        f"max={stats.max().item():.4g}, "
        f"mean={stats.mean().item():.4g}, "
        f"std={stats.std().item():.4g}"
    )
    if has_nan:
        nan_count = int(torch.isnan(t_float).sum().item())
        info += f"\n  {' '*len(name)}  ⚠ contains {nan_count} NaN"
    if has_inf:
        inf_count = int(torch.isinf(t_float).sum().item())
        info += f"\n  {' '*len(name)}  ⚠ contains {inf_count} Inf"
    print(f"  {name}: {info}")


def compare(
    a: torch.Tensor,
    b: torch.Tensor,
    name_a: str = "a",
    name_b: str = "b",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """对比两个 tensor 是否一致 (shape + 数值).

    用于:
        - 加载 checkpoint 后验证模型一致
        - reproduce 实验时验证两次跑的结果一致
        - 优化后验证 fast 版本和 slow 版本结果一致

    示例:
        >>> compare(loaded_out, original_out, "loaded", "original")
          ✓ loaded vs original: max_diff=0.0, mean_diff=0.0
    """
    if a.shape != b.shape:
        print(f"  ✗ shape 不一致: {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    diff = (a.detach().float() - b.detach().float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    is_close = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    flag = "✓" if is_close else "✗"
    print(f"  {flag} {name_a} vs {name_b}: max_diff={max_diff:.4g}, mean_diff={mean_diff:.4g}")
    return is_close


# ================================================================
#  3. 模型检查
# ================================================================

def param_summary(model: nn.Module, show_layers: bool = False) -> None:
    """打印模型参数统计: 总数 / 可训练 / 冻结. 可选列出每层.

    示例:
        >>> param_summary(model)
          Total params:      11,178,058
          Trainable params:  11,178,058  (100.0%)
          Frozen params:              0

        >>> param_summary(model, show_layers=True)
          ... (同上)
          ====================================================
          ✓ encoder.conv1.weight                  (64, 3, 3, 3) (1,728)
          ✓ encoder.bn1.weight                    (64,)         (64)
          ✗ encoder.bn1.running_mean              (64,)         (64)  ← 冻结
          ...
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"  Total params:     {total:>12,}")
    pct = 100 * trainable / max(total, 1)
    print(f"  Trainable params: {trainable:>12,}  ({pct:.1f}%)")
    print(f"  Frozen params:    {frozen:>12,}")

    if show_layers:
        print(f"  {'='*68}")
        for name, p in model.named_parameters():
            flag = "✓" if p.requires_grad else "✗"
            shape_str = str(tuple(p.shape))
            print(f"  {flag} {name:<45s} {shape_str:<20s} ({p.numel():,})")


def grad_summary(model: nn.Module, top_k: int | None = None) -> None:
    """backward() 之后检查每层梯度健康 (norm / NaN / 是否被冻结).

    重要场景:
        - 训练 loss 不下降 → 看哪些层 grad 是 0 (可能 detach 错了)
        - 梯度爆炸 → 看哪些层 norm 异常大
        - 出现 NaN → 立刻定位罪魁

    Args:
        model: 必须已经跑过 backward()
        top_k: 只打印 grad norm 最大的 k 层 (None=全部)

    示例:
        >>> loss.backward()
        >>> grad_summary(model, top_k=5)
          Layer                                    Grad Norm    NaN?
          ------------------------------------------------------------
          head.weight                                  0.4231
          encoder.lin2.weight                          0.1832
          encoder.lin1.weight                          0.0521
          ...
    """
    rows = []
    any_nan = False
    for name, p in model.named_parameters():
        if not p.requires_grad:
            rows.append((name, None, False, "(frozen)"))
            continue
        if p.grad is None:
            rows.append((name, None, False, "(no grad)"))
            continue
        norm = p.grad.norm().item()
        has_nan = bool(torch.isnan(p.grad).any().item())
        any_nan |= has_nan
        rows.append((name, norm, has_nan, ""))

    # 按 grad norm 排序 (None 排最后)
    rows.sort(key=lambda r: (-r[1] if r[1] is not None else float("inf")))
    if top_k is not None:
        rows = rows[:top_k]

    print(f"  {'Layer':<45s} {'Grad Norm':>12s}  Flags")
    print(f"  {'-'*68}")
    for name, norm, has_nan, status in rows:
        if norm is None:
            print(f"  {name:<45s} {status:>12s}")
        else:
            flag = "⚠ NaN" if has_nan else ""
            print(f"  {name:<45s} {norm:>12.4g}  {flag}")

    if any_nan:
        print("\n  ⚠⚠⚠ 检测到 NaN 梯度! 排查: lr 过大 / 输入有 NaN / loss 不稳定 / 没做 grad clipping")


# ================================================================
#  4. 性能 / 复现
# ================================================================

@contextmanager
def timer(name: str = "block", sync_cuda: bool = True) -> Iterator[None]:
    """Context manager 计时. GPU 上自动 sync 保证准确.

    ⚠️ GPU 上不 sync 的话,记的是 kernel launch 时间,不是真正执行时间!

    示例:
        >>> with timer("forward"):
        ...     out = model(past)
          [forward] 12.34 ms

        >>> with timer("eval epoch", sync_cuda=False):  # CPU only 时可以关
        ...     evaluate(model, loader)
    """
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        # ms 还是 s 自动选
        if elapsed < 1.0:
            print(f"  [{name}] {elapsed*1000:.2f} ms")
        else:
            print(f"  [{name}] {elapsed:.3f} s")


def seed_all(seed: int = 42, deterministic: bool = False) -> None:
    """一键设所有随机种子. 实验复现的标准操作.

    Args:
        seed: 种子值
        deterministic: True 时强制 cuDNN 确定性 (速度变慢,但完全复现)

    示例:
        >>> seed_all(42)              # 普通复现
        >>> seed_all(42, deterministic=True)  # paper 级严格复现
    """
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  Seed set to {seed}{' (deterministic)' if deterministic else ''}")
