'''
A typical training procedure for a neural network is as follows:

Define the neural network that has some learnable parameters (or weights)
Iterate over a dataset of inputs
Process input through the network
Compute the loss (how far is the output from being correct)
Propagate gradients back into the network’s parameters
Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
教神经网络识别手写数字（看图猜 0-9）。这就是经典的 LeNet，深度学习的 "Hello World"，1998 年 Yann LeCun 提出，原本用来读支票上的数字。

任务设定：

输入：一张 32×32 的灰度小图（比如某人手写的数字 "7"）
输出：模型给出 10 个分数 —— 分别对应 0, 1, 2, ..., 9，分数最高的那个就是预测结果
数据从哪来：tutorial 后面会教你用 torchvision.datasets.MNIST 下载，是 6 万张 28×28 的手写数字图片（LeNet 论文里 pad 到 32×32）
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        