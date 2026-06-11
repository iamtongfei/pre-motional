"""utils — 个人 PyTorch 实验工具包

用法:
    from utils import banner, show, summarize, timer
    或者:
    from utils.debug import banner, show, ...
"""

from .debug import (
    # 打印格式
    banner,
    show,
    show_dict,
    # 数值检查
    summarize,
    compare,
    # 模型检查
    param_summary,
    grad_summary,
    # 性能
    timer,
    # 复现
    seed_all,
)

__all__ = [
    "banner",
    "show",
    "show_dict",
    "summarize",
    "compare",
    "param_summary",
    "grad_summary",
    "timer",
    "seed_all",
]
