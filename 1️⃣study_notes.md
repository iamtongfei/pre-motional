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
- PyTorch official 60-min blitz: https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- "What is torch.nn really?" tutorial: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
- Forward hooks tutorial: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
