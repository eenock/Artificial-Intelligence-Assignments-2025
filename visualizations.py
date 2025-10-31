# visualizations.py
"""
Generate academic figures for Fashion-MNIST report
Run this AFTER training your model (after running 4.py)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ================================
# Setup (same as your 4.py)
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Load your trained model
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
model.load_state_dict(torch.load("model/fashion_mnist_mlp.pth"))
model.eval()

print("Model loaded successfully!")

# ================================
# Figure 1: Training Curves
# ================================
def plot_training_curves():
    """Plot accuracy and loss curves over epochs"""
    # Your training data from the output
    epochs = list(range(1, 11))
    accuracy = [70.74, 77.46, 79.90, 81.63, 82.02, 82.35, 83.39, 83.80, 83.56, 84.49]
    loss = [0.7999, 0.6348, 0.5620, 0.5210, 0.4966, 0.4873, 0.4652, 0.4602, 0.4584, 0.4358]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(epochs, accuracy, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Over Training', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Loss plot
    ax2.plot(epochs, loss, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Model Loss Over Training', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig('figure1_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1 saved: figure1_training_curves.png")
    plt.close()

# ================================
# Figure 2: Sample Predictions Grid
# ================================
def plot_prediction_grid():
    """Show 4x5 grid of predictions with correct/incorrect highlighted"""
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle('Sample Model Predictions on Test Set', fontsize=16, fontweight='bold', y=0.995)
    
    np.random.seed(42)  # Reproducible
    
    for idx, ax in enumerate(axes.flat):
        test_idx = np.random.randint(len(test_data))
        x, true = test_data[test_idx]
        x_batch = x.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = model(x_batch)
            pred_label = pred.argmax(1).item()
            confidence = torch.softmax(pred, dim=1)[0][pred_label].item()
        
        ax.imshow(x.squeeze(), cmap='gray')
        
        # Green border for correct, red for incorrect
        if pred_label == true:
            color = 'green'
            status = '✓'
        else:
            color = 'red'
            status = '✗'
        
        ax.set_title(f"{status} Pred: {classes[pred_label]}\nTrue: {classes[true]}\nConf: {confidence:.2f}", 
                     fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
        
        # Add colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('figure2_prediction_grid.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2 saved: figure2_prediction_grid.png")
    plt.close()

# ================================
# Figure 3: Confusion Matrix
# ================================
def plot_confusion_matrix():
    """Generate confusion matrix showing where model makes mistakes"""
    print("Computing confusion matrix (this may take a moment)...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            pred = model(X)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Confusion Matrix - Classification Performance by Class', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('figure3_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3 saved: figure3_confusion_matrix.png")
    plt.close()

# ================================
# Figure 4: Per-Class Accuracy Bar Chart
# ================================
def plot_class_accuracy():
    """Bar chart showing accuracy for each clothing category"""
    print("Computing per-class accuracy...")
    
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            pred = model(X)
            predictions = pred.argmax(1).cpu()
            
            for label, prediction in zip(y, predictions):
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1
    
    accuracies = [100 * class_correct[i] / class_total[i] for i in range(10)]
    
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if acc > 85 else '#f39c12' if acc > 75 else '#e74c3c' for acc in accuracies]
    bars = plt.bar(range(10), accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Clothing Category', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Classification Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(10), classes, rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    # Add average line
    avg_acc = np.mean(accuracies)
    plt.axhline(y=avg_acc, color='black', linestyle='--', linewidth=2, 
                label=f'Average: {avg_acc:.2f}%')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('figure4_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 4 saved: figure4_class_accuracy.png")
    plt.close()

# ================================
# Figure 5: Confidence Distribution
# ================================
def plot_confidence_distribution():
    """Show distribution of model confidence for correct vs incorrect predictions"""
    print("Analyzing prediction confidence...")
    
    correct_conf = []
    incorrect_conf = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            pred = model(X)
            probs = torch.softmax(pred, dim=1)
            predictions = pred.argmax(1).cpu()
            confidences = probs.max(dim=1).values.cpu().numpy()
            
            for label, prediction, conf in zip(y, predictions, confidences):
                if label == prediction:
                    correct_conf.append(conf)
                else:
                    incorrect_conf.append(conf)
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_conf, bins=50, alpha=0.7, label='Correct Predictions', 
             color='green', edgecolor='black')
    plt.hist(incorrect_conf, bins=50, alpha=0.7, label='Incorrect Predictions', 
             color='red', edgecolor='black')
    
    plt.xlabel('Model Confidence', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    plt.title('Distribution of Model Confidence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    avg_correct = np.mean(correct_conf)
    avg_incorrect = np.mean(incorrect_conf)
    plt.axvline(x=avg_correct, color='green', linestyle='--', linewidth=2, 
                label=f'Avg Correct: {avg_correct:.3f}')
    plt.axvline(x=avg_incorrect, color='red', linestyle='--', linewidth=2,
                label=f'Avg Incorrect: {avg_incorrect:.3f}')
    
    plt.tight_layout()
    plt.savefig('figure5_confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 5 saved: figure5_confidence_distribution.png")
    plt.close()

# ================================
# Figure 6: Failure Cases
# ================================
def plot_failure_cases():
    """Show examples where the model made mistakes"""
    print("Finding misclassified examples...")
    
    mistakes = []
    with torch.no_grad():
        for X, y in test_loader:
            X_device = X.to(DEVICE)
            pred = model(X_device)
            probs = torch.softmax(pred, dim=1)
            predictions = pred.argmax(1).cpu()
            
            for i, (label, prediction) in enumerate(zip(y, predictions)):
                if label != prediction:
                    conf = probs[i][prediction].item()
                    mistakes.append((X[i], label.item(), prediction.item(), conf))
                    if len(mistakes) >= 12:
                        break
            if len(mistakes) >= 12:
                break
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('Failure Cases - Misclassified Examples', fontsize=16, fontweight='bold', y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(mistakes):
            img, true_label, pred_label, conf = mistakes[idx]
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(f"True: {classes[true_label]}\nPredicted: {classes[pred_label]}\nConf: {conf:.2f}", 
                        fontsize=9, color='red', fontweight='bold')
            ax.axis('off')
            
            # Red border for mistakes
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('figure6_failure_cases.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 6 saved: figure6_failure_cases.png")
    plt.close()

# ================================
# Figure 7: Model Architecture Diagram
# ================================
def plot_architecture():
    """Visualize the neural network architecture"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define layers
    layers = [
        ("Input\n28×28", 784),
        ("Hidden 1\n512 neurons\nReLU", 512),
        ("Hidden 2\n512 neurons\nReLU", 512),
        ("Output\n10 classes", 10)
    ]
    
    layer_positions = [0.15, 0.4, 0.65, 0.9]
    
    # Draw layers
    for i, (label, size) in enumerate(layers):
        x = layer_positions[i]
        
        # Draw rectangle for layer
        height = 0.15 + (size / 1000) * 0.3  # Scale by size
        rect = plt.Rectangle((x - 0.05, 0.5 - height/2), 0.1, height, 
                            fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, 0.3, label, ha='center', va='top', fontsize=11, fontweight='bold')
        
        # Add parameter count
        if i > 0:
            prev_size = layers[i-1][1]
            params = size * prev_size + size  # weights + biases
            ax.text(x, 0.73, f"{params:,}\nparams", ha='center', va='bottom', 
                   fontsize=9, style='italic', color='darkblue')
        
        # Draw arrows
        if i < len(layers) - 1:
            ax.arrow(x + 0.05, 0.5, 0.15, 0, head_width=0.03, head_length=0.02, 
                    fc='black', ec='black', linewidth=2)
    
    # Add title and info
    ax.text(0.5, 0.95, 'Neural Network Architecture', ha='center', 
           fontsize=16, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.05, 'Total Parameters: 669,706', ha='center', 
           fontsize=12, fontweight='bold', transform=ax.transAxes, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figure7_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 7 saved: figure7_architecture.png")
    plt.close()

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating Academic Figures for Fashion-MNIST Report")
    print("="*60 + "\n")
    
    plot_training_curves()
    plot_prediction_grid()
    plot_confusion_matrix()
    plot_class_accuracy()
    plot_confidence_distribution()
    plot_failure_cases()
    plot_architecture()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  • figure1_training_curves.png - Accuracy and loss over epochs")
    print("  • figure2_prediction_grid.png - Sample predictions with confidence")
    print("  • figure3_confusion_matrix.png - Detailed confusion matrix")
    print("  • figure4_class_accuracy.png - Per-class accuracy bar chart")
    print("  • figure5_confidence_distribution.png - Confidence analysis")
    print("  • figure6_failure_cases.png - Misclassified examples")
    print("  • figure7_architecture.png - Network architecture diagram")
    print("\nAdd these to your report for a professional, academic look!")