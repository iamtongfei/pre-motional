Core concepts to practice:

- Tensor creation: torch.zeros, torch.tensor, torch.from_numpy
- Device management: .to(device), torch.cuda.is_available()
- Autograd basics: requires_grad_(), .backward(), .grad
- No-grad context: @torch.no_grad() decorator and with torch.no_grad():
- Model loading: torch.load(), model.load_state_dict(), model.eval()
- Forward hooks: module.register_forward_hook(fn) — you'll use this for encoder
ablation
- Tensor operations: torch.cat, torch.stack, broadcasting, einsum
- Shapes and views: .view(), .reshape(), .unsqueeze(), .squeeze()
- Masking: torch.where, boolean masks, masked_fill
- Dataclasses with tensors: our codebase uses @dataclass classes that hold tensors

Recommended resources:

- PyTorch official 60-min blitz: [https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- "What is torch.nn really?" tutorial: [https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)
- Forward hooks tutorial: [https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)

# 1 Tensor creation: torch.zeros, torch.tensor, torch.from_numpy

```python
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

Random Tensor:
 tensor([[0.5615, 0.8933, 0.1537],
        [0.1362, 0.2967, 0.6852]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

NumPy array to Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

Changes in the NumPy array reflects in the tensor.

```python
np.add(n, 1, out=n)
print(f"t: {t}") # >>> t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
print(f"n: {n}") # >>> n: [2. 2. 2. 2. 2.]
```

# 2 Device management: .to(device), torch.cuda.is_available()

Each of them can be run on the GPU (at typically higher speeds than on a CPU). If you’re using Colab, allocate a GPU by going to Edit > Notebook Settings.

```python
# We move our tensor to the GPU if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")
```

参考2️⃣[tensor1.py](http://tensor1.py)

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 版本是 2.4.0，但这个 conda 环境的 torch 构建里没有包含 torch.accelerator 模块。用兼容性更好的写法替换一下
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")
```

**MPS 就是 GPU，只是苹果的叫法。**


|            | CUDA                                | MPS                             |
| ---------- | ----------------------------------- | ------------------------------- |
| 全称         | Compute Unified Device Architecture | Metal Performance Shaders       |
| 厂商         | NVIDIA                              | Apple                           |
| 适用硬件       | NVIDIA 独立显卡                         | Apple Silicon（M1/M2/M3）芯片内置 GPU |
| PyTorch 使用 | `torch.cuda`                        | `torch.backends.mps`            |
| 设备名        | `cuda:0`                            | `mps:0`                         |


**关键区别是架构：**

- **NVIDIA GPU（CUDA）**：独立显卡，有专属显存（VRAM），CPU 内存和 GPU 显存是分开的，数据需要通过 PCIe 总线来回传输
- **Apple Silicon GPU（MPS）**：集成在芯片里，和 CPU **共享同一块内存**（Unified Memory），没有独立显存，所以 `.to('mps')` 几乎不涉及数据拷贝

你的输出 `mps:0` 就是说 tensor 现在在 M 系列芯片的 GPU 核心上跑，用的是 Metal 框架加速计算。

# 3 Autograd basics: requires_grad_(), .backward(), .grad

[https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

#### Background

Neural networks (NNs) are a collection of nested functions that are executed on some input data. These functions are defined by parameters (consisting of weights and biases), which in PyTorch are stored in tensors.

Training a NN happens in two steps:

1. Forward Propagation: In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.
2. Backward Propagation: In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent.

