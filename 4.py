# main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ================================
# 1. Config
# ================================
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {DEVICE}")

# ================================
# 2. Data
# ================================
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ================================
# 3. Model
# ================================
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 512),   nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.net(self.flatten(x))

model = NeuralNetwork().to(DEVICE)
print(model)

# ================================
# 4. Loss & Optimizer
# ================================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# ================================
# 5. Train / Test Loops
# ================================
def train_one_epoch():
    model.train()
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
            print(f"  loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

def test():
    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = 100 * correct / size
    print(f"\nTest Accuracy: {accuracy:>6.2f}%, Avg loss: {test_loss:>8f}\n")
    return accuracy

# ================================
# 6. Training
# ================================
print("\nStarting training...\n" + "="*50)
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}\n{'-'*30}")
    train_one_epoch()
    test()

print(f"Done! Total time: {time.time() - start_time:.1f}s")

# ================================
# 7. Save Model
# ================================
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/fashion_mnist_mlp.pth")
print("Model saved to model/fashion_mnist_mlp.pth")

# ================================
# 8. Inference Demo
# ================================
model.eval()
idx = np.random.randint(len(test_data))
x, true = test_data[idx]
x_batch = x.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(x_batch)
    pred_label = pred.argmax(1).item()

plt.figure(figsize=(4,4))
plt.imshow(x.squeeze(), cmap='gray')
plt.title(f"Pred: {classes[pred_label]}\nTrue: {classes[true]}", fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.show()