import torch
'''
torch: 2.4.0
Name: torch
Version: 2.4.0
Name: torchvision
Version: 0.19.0.dev20240528
'''

########################## 2 START ##########################
# to fix the bug:
# conda run -n torch-cav-py38 pip install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
# conda run -n torch-cav-py38 python -c "import torchvision; print(torchvision.__version__)"
# >>> It should print 0.19.0 (no .dev), and the import error will be gone.

from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
'''
How did I know torch.rand(1, 3, 64, 64) for the input?

That comes from what ResNet18 expects — it's a CNN for images:

Dimension	Value	Meaning
1	batch size	1 image at a time
3	channels	RGB (red, green, blue)
64	height	64 pixels tall
64	width	64 pixels wide
The batch and spatial dims (1, 64, 64) are flexible — you could use torch.rand(4, 3, 224, 224) for 4 images at 224×224 (the standard ImageNet size). The 3 channels is fixed by the model's first conv layer.
'''
labels = torch.rand(1, 1000)
'''
Why torch.rand(1, 1000) for labels? Where does 1000 come from?

ResNet18 was pretrained on ImageNet, which has exactly 1000 classes (cats, dogs, cars, etc.). So the model always outputs 1000 scores — one per class. The labels must match that shape so the loss (prediction - labels) can subtract element-wise:


prediction: [1, 1000]
labels:     [1, 1000]  ← must match
loss:       scalar     ← .sum() collapses everything
The 1 is batch size (one image). The 1000 is forced by the model architecture.
'''

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step() #gradient descent


'''
In a NN, parameters that don’t compute gradients are usually called frozen parameters. 
It is useful to “freeze” part of your model if you know in advance that you won’t need the gradients of those parameters (this offers some performance benefits by reducing autograd computations).

In finetuning, we freeze most of the model and typically only modify the classifier layers to make predictions on new labels. 
Let’s walk through a small example to demonstrate this. As before, we load a pretrained resnet18 model, and freeze all the parameters.
'''

from torch import nn, optim

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
    
'''
Let’s say we want to finetune the model on a new dataset with 10 labels. 
In resnet, the classifier is the last linear layer model.fc. We can simply replace it with a new linear layer (unfrozen by default) that acts as our classifier.
'''
model.fc = nn.Linear(512, 10)

'''
Now all parameters in the model, except the parameters of model.fc, are frozen. The only parameters that compute gradients are the weights and bias of model.fc.
'''
# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)