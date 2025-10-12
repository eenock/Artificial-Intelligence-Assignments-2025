import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

print("PyTorch CIFAR-10 Classification Project")
print("=" * 50)

# Set device to CPU (optimized for CPU training)
device = torch.device("cpu")
print(f"Using device: {device}")
print("Note: Training optimized for CPU - reduced complexity for faster convergence")

# Data preprocessing and augmentation
print("\n1. Setting up data transforms and loaders...")
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

# Create data loaders (CPU optimized)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)  # Smaller batch size for CPU
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)  # num_workers=0 for CPU

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(f"Dataset loaded: {len(trainset)} training samples, {len(testset)} test samples")
print(f"Classes: {classes}")

# Define CNN Architecture (CPU-optimized smaller model)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers (reduced channels for CPU)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (smaller sizes for CPU)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third convolutional block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 4 * 4)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Initialize model, loss function, and optimizer (CPU optimized)
print("\n2. Initializing model, loss function, and optimizer...")
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)  # Slightly higher LR for CPU
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)  # More aggressive scheduling

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Training function
def train_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=15):
    print(f"\n3. Training model for {num_epochs} epochs (CPU optimized)...")
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 200 == 199:  # Less frequent printing for CPU training
                print(f'[Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}] Loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        # Calculate training accuracy
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(testloader)
        
        # Store metrics
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}')
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return train_losses, train_accuracies, test_accuracies

# Train the model (reduced epochs for CPU)
train_losses, train_accuracies, test_accuracies = train_model(
    model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=15
)

# Final evaluation
def evaluate_model(model, testloader, classes):
    print("\n4. Final model evaluation...")
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for confusion matrix
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    overall_accuracy = 100 * correct / total
    print(f'Final Test Accuracy: {overall_accuracy:.2f}%')
    
    print("\nPer-class accuracies:")
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{classes[i]}: {class_acc:.2f}%')
    
    return overall_accuracy, all_predicted, all_labels

# Evaluate the model
final_accuracy, predictions, true_labels = evaluate_model(model, testloader, classes)

# Visualization function
def create_visualizations(train_losses, train_accuracies, test_accuracies, 
                            predictions, true_labels, classes):
    print("\n5. Creating visualizations...")
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training and Test Accuracy
    epochs = range(1, len(train_accuracies) + 1)
    ax1.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    ax2.plot(epochs, train_losses, 'g-', linewidth=2)
    ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, 
                ax=ax3, cmap='Blues')
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # Plot 4: Per-class Accuracy
    class_accuracies = []
    for i in range(10):
        class_mask = np.array(true_labels) == i
        class_predictions = np.array(predictions)[class_mask]
        class_true = np.array(true_labels)[class_mask]
        if len(class_true) > 0:
            acc = (class_predictions == class_true).mean() * 100
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    bars = ax4.bar(classes, class_accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    ax4.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xlabel('Classes')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pytorch_cifar10_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return class_accuracies

# Generate all visualizations
class_accuracies = create_visualizations(train_losses, train_accuracies, test_accuracies, 
                                        predictions, true_labels, classes)

# Print final summary
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"Final Test Accuracy: {final_accuracy:.2f}%")
print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Total Model Parameters: {total_params:,}")
print(f"Device Used: {device}")

print(f"\nTop 3 performing classes:")
class_acc_pairs = list(zip(classes, class_accuracies))
class_acc_pairs.sort(key=lambda x: x[1], reverse=True)
for i, (class_name, acc) in enumerate(class_acc_pairs[:3]):
    print(f"{i+1}. {class_name}: {acc:.2f}%")

print(f"\nBottom 3 performing classes:")
for i, (class_name, acc) in enumerate(class_acc_pairs[-3:]):
    print(f"{i+1}. {class_name}: {acc:.2f}%")

print("\nModel training completed successfully!")
print("Results saved as 'pytorch_cifar10_results.png'")