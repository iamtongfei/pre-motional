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