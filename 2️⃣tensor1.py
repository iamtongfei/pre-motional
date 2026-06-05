import torch
import numpy as np

# Initialization
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

    # from numpy -> tensor
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}") # >>> t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
print(f"n: {n}") # >>> n: [2. 2. 2. 2. 2.]

# Devices check
# We move our tensor to the GPU if available
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 版本是 2.4.0，但这个 conda 环境的 torch 构建里没有包含 torch.accelerator 模块。用兼容性更好的写法替换一下
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")
# 第一次打印 cpu — tensor 刚创建时默认在 CPU 上
# 第二次打印 mps:0 — 成功移到了 Mac 的 GPU（Apple Silicon 的 Metal Performance Shaders）