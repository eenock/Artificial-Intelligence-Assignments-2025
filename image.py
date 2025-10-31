# =========================================================
# Project 4 - Image Recognition (PyTorch)
# Safe version for Windows (uses __main__ guard)
# =========================================================

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ---------------------------
# Helper Functions
# ---------------------------
def imshow(img, title=None, save_as=None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    if save_as:
        plt.savefig(save_as, dpi=300)
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # ---------------------------
    # 1. Data Loading & Preprocessing
    # ---------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # ---------------------------
    # 2. Visualize Training Images
    # ---------------------------
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images[:8]),
           title='Sample CIFAR-10 Training Images',
           save_as='training_samples.png')

    print('Labels:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))

    # ---------------------------
    # 3. Initialize Model, Loss, Optimizer
    # ---------------------------
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # ---------------------------
    # 4. Train Network
    # ---------------------------
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('✅ Finished Training')

    # ---------------------------
    # 5. Save Model
    # ---------------------------
    PATH = './cnn_cifar10.pth'
    torch.save(net.state_dict(), PATH)

    # ---------------------------
    # 6. Evaluate Model
    # ---------------------------
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:8]),
           title='Test Images',
           save_as='test_images.png')

    print('GroundTruth:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted:', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(8)))

    imshow(torchvision.utils.make_grid(images[:8]),
           title='Predicted: ' + ' '.join(f'{classes[predicted[j]]}' for j in range(8)),
           save_as='predicted_samples.png')

    # ---------------------------
    # 7. Compute Test Accuracy
    # ---------------------------
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * correct / total
    print(f'🎯 Accuracy on 10000 test images: {accuracy:.2f}%')

    # ---------------------------
    # 8. Confusion Matrix
    # ---------------------------
    cf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cf_matrix, annot=False, cmap='Blues')
    plt.title('Confusion Matrix - CIFAR-10')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
