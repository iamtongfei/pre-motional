import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import boto3
import os
import threading
import time
import urllib.request
import urllib.error

# --- Config ---
EPOCHS = 100
BATCH_SIZE = 128
S3_BUCKET = 'pnnl-s3'
MODEL_PATH = 'cifar10_v2.pth'
CHECKPOINT_LOCAL = 'checkpoint.pth'
CHECKPOINT_S3_KEY = 'checkpoints/cifar10_v2_checkpoint.pth'
CHECKPOINT_EVERY = 10   # save to S3 every N epochs

# --- Spot interruption monitor ---
_spot_interrupted = threading.Event()

def _poll_spot_termination():
    """AWS gives a 2-min warning via metadata before reclaiming a Spot instance."""
    while not _spot_interrupted.is_set():
        try:
            urllib.request.urlopen(
                'http://169.254.169.254/latest/meta-data/spot/termination-time',
                timeout=1
            )
            # 200 response means termination is imminent
            print("\n[SPOT] 2-min termination notice received — will checkpoint at end of epoch.")
            _spot_interrupted.set()
            return
        except urllib.error.HTTPError as e:
            if e.code != 404:   # 404 = normal (not being terminated)
                pass
        except Exception:
            pass
        time.sleep(5)

# --- S3 helpers ---
s3 = boto3.client('s3', region_name='us-west-2')

def save_checkpoint(net, optimizer, scheduler, epoch, best_acc, reason="scheduled"):
    state = {
        'epoch': epoch,
        'best_acc': best_acc,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, CHECKPOINT_LOCAL)
    s3.upload_file(CHECKPOINT_LOCAL, S3_BUCKET, CHECKPOINT_S3_KEY)
    print(f"  [{reason}] Checkpoint saved → s3://{S3_BUCKET}/{CHECKPOINT_S3_KEY}  (epoch {epoch + 1})")

def load_checkpoint(net, optimizer, scheduler):
    try:
        s3.download_file(S3_BUCKET, CHECKPOINT_S3_KEY, CHECKPOINT_LOCAL)
        state = torch.load(CHECKPOINT_LOCAL, map_location=device)
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print(f"Resumed from checkpoint — epoch {state['epoch'] + 1}, best acc: {state['best_acc']:.1f}%")
        return state['epoch'] + 1, state['best_acc']
    except Exception:
        print("No checkpoint found — starting fresh")
        return 0, 0.0

# --- Data ---
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

# --- Model ---
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

# --- Resume from S3 checkpoint if one exists ---
start_epoch, best_acc = load_checkpoint(net, optimizer, scheduler)

# --- Start Spot monitor (harmless on on-demand instances) ---
monitor_thread = threading.Thread(target=_poll_spot_termination, daemon=True)
monitor_thread.start()

# --- Training loop ---
for epoch in range(start_epoch, EPOCHS):
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
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {running_loss/len(trainloader):.3f} | Acc: {acc:.1f}% | Best: {best_acc:.1f}%")

    # Checkpoint every N epochs
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        save_checkpoint(net, optimizer, scheduler, epoch, best_acc)

    # Spot termination — save immediately and exit cleanly
    if _spot_interrupted.is_set():
        save_checkpoint(net, optimizer, scheduler, epoch, best_acc, reason="SPOT INTERRUPT")
        print("Exiting cleanly. Re-launch to resume from checkpoint.")
        _spot_interrupted.set()  # stop monitor thread
        exit(0)

print("Training complete!")
_spot_interrupted.set()  # stop monitor thread

# --- Save final model locally + upload ---
torch.save(net.state_dict(), MODEL_PATH)
print(f"Model saved locally: {MODEL_PATH}")

s3.upload_file(MODEL_PATH, S3_BUCKET, f'models/{MODEL_PATH}')
print(f"Model uploaded to s3://{S3_BUCKET}/models/{MODEL_PATH}")
