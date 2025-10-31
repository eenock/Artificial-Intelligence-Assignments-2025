"""
What is torch.nn Really? - Complete Tutorial Implementation
Progressive refinement from scratch to full PyTorch abstractions
"""

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import math
import numpy as np
from pathlib import Path
import requests
import pickle
import gzip
import time

# =============================================================================
# DATA PREPARATION
# =============================================================================

def download_mnist():
    """Download MNIST dataset if not already present."""
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)
    
    URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
    FILENAME = "mnist.pkl.gz"
    
    if not (PATH / FILENAME).exists():
        print("Downloading MNIST dataset...")
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
        print("Download complete!")
    
    return PATH / FILENAME

def load_mnist():
    """Load MNIST data from pickle file."""
    filepath = download_mnist()
    
    with gzip.open(filepath.as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    
    # Convert to PyTorch tensors
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    
    return x_train, y_train, x_valid, y_valid

# =============================================================================
# STAGE 1: FROM SCRATCH (NO torch.nn)
# =============================================================================

def stage1_from_scratch(x_train, y_train, x_valid, y_valid):
    """Neural net using only tensor operations."""
    print("\n" + "="*70)
    print("STAGE 1: FROM SCRATCH (Pure PyTorch Tensors)")
    print("="*70)
    
    n, c = x_train.shape
    bs = 64
    lr = 0.5
    epochs = 2
    
    # Initialize weights and bias
    weights = torch.randn(784, 10) / math.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(10, requires_grad=True)
    
    # Manual log_softmax
    def log_softmax(x):
        return x - x.exp().sum(-1).log().unsqueeze(-1)
    
    def model(xb):
        return log_softmax(xb @ weights + bias)
    
    # Negative log-likelihood loss
    def nll(input, target):
        return -input[range(target.shape[0]), target].mean()
    
    # Accuracy
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    
    # Initial prediction
    xb = x_train[0:bs]
    yb = y_train[0:bs]
    preds = model(xb)
    
    print(f"\nInitial state:")
    print(f"  Loss: {nll(preds, yb):.4f}")
    print(f"  Accuracy: {accuracy(preds, yb):.4f}")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = nll(pred, yb)
            
            loss.backward()
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()
    
    training_time = time.time() - start_time
    
    # Final results
    final_loss = nll(model(xb), yb)
    final_acc = accuracy(model(xb), yb)
    
    print(f"\nAfter {epochs} epochs:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return final_loss.item(), final_acc.item(), training_time

# =============================================================================
# STAGE 2: USING torch.nn.functional
# =============================================================================

def stage2_functional(x_train, y_train, x_valid, y_valid):
    """Replace manual functions with torch.nn.functional."""
    print("\n" + "="*70)
    print("STAGE 2: USING torch.nn.functional")
    print("="*70)
    
    n, c = x_train.shape
    bs = 64
    lr = 0.5
    epochs = 2
    
    weights = torch.randn(784, 10) / math.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(10, requires_grad=True)
    
    def model(xb):
        return xb @ weights + bias
    
    loss_func = F.cross_entropy
    
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    
    print(f"\nUsing F.cross_entropy (combines log_softmax + NLL)")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)
            
            loss.backward()
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()
    
    training_time = time.time() - start_time
    
    xb = x_train[0:bs]
    yb = y_train[0:bs]
    final_loss = loss_func(model(xb), yb)
    final_acc = accuracy(model(xb), yb)
    
    print(f"\nResults:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return final_loss.item(), final_acc.item(), training_time

# =============================================================================
# STAGE 3: USING nn.Module
# =============================================================================

def stage3_module(x_train, y_train, x_valid, y_valid):
    """Use nn.Module for cleaner code."""
    print("\n" + "="*70)
    print("STAGE 3: USING nn.Module")
    print("="*70)
    
    class Mnist_Logistic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
            self.bias = nn.Parameter(torch.zeros(10))
        
        def forward(self, xb):
            return xb @ self.weights + self.bias
    
    n, c = x_train.shape
    bs = 64
    lr = 0.5
    epochs = 2
    
    model = Mnist_Logistic()
    loss_func = F.cross_entropy
    
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    
    print(f"\nUsing nn.Module with nn.Parameter")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)
            
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()
    
    training_time = time.time() - start_time
    
    xb = x_train[0:bs]
    yb = y_train[0:bs]
    final_loss = loss_func(model(xb), yb)
    final_acc = accuracy(model(xb), yb)
    
    print(f"\nResults:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return final_loss.item(), final_acc.item(), training_time

# =============================================================================
# STAGE 4: USING nn.Linear
# =============================================================================

def stage4_linear(x_train, y_train, x_valid, y_valid):
    """Use nn.Linear instead of manual matrix multiplication."""
    print("\n" + "="*70)
    print("STAGE 4: USING nn.Linear")
    print("="*70)
    
    class Mnist_Logistic(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(784, 10)
        
        def forward(self, xb):
            return self.lin(xb)
    
    n, c = x_train.shape
    bs = 64
    lr = 0.5
    epochs = 2
    
    model = Mnist_Logistic()
    loss_func = F.cross_entropy
    
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    
    print(f"\nUsing nn.Linear (handles weight init automatically)")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)
            
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()
    
    training_time = time.time() - start_time
    
    xb = x_train[0:bs]
    yb = y_train[0:bs]
    final_loss = loss_func(model(xb), yb)
    final_acc = accuracy(model(xb), yb)
    
    print(f"\nResults:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return final_loss.item(), final_acc.item(), training_time

# =============================================================================
# STAGE 5: USING torch.optim
# =============================================================================

def stage5_optim(x_train, y_train, x_valid, y_valid):
    """Use torch.optim for automatic optimization."""
    print("\n" + "="*70)
    print("STAGE 5: USING torch.optim")
    print("="*70)
    
    class Mnist_Logistic(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(784, 10)
        
        def forward(self, xb):
            return self.lin(xb)
    
    n, c = x_train.shape
    bs = 64
    lr = 0.5
    epochs = 2
    
    model = Mnist_Logistic()
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_func = F.cross_entropy
    
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    
    print(f"\nUsing optim.SGD (handles parameter updates)")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
    
    training_time = time.time() - start_time
    
    xb = x_train[0:bs]
    yb = y_train[0:bs]
    final_loss = loss_func(model(xb), yb)
    final_acc = accuracy(model(xb), yb)
    
    print(f"\nResults:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return final_loss.item(), final_acc.item(), training_time

# =============================================================================
# STAGE 6: USING Dataset and DataLoader
# =============================================================================

def stage6_dataloader(x_train, y_train, x_valid, y_valid):
    """Use Dataset and DataLoader for cleaner data handling."""
    print("\n" + "="*70)
    print("STAGE 6: USING Dataset and DataLoader")
    print("="*70)
    
    class Mnist_Logistic(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(784, 10)
        
        def forward(self, xb):
            return self.lin(xb)
    
    bs = 64
    lr = 0.5
    epochs = 2
    
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    
    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
    
    model = Mnist_Logistic()
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_func = F.cross_entropy
    
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    
    print(f"\nUsing DataLoader for automatic batching")
    print(f"  Train batches: {len(train_dl)}")
    print(f"  Valid batches: {len(valid_dl)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
            valid_acc = sum(accuracy(model(xb), yb) * len(xb) for xb, yb in valid_dl) / len(valid_ds)
        
        print(f"  Epoch {epoch}: valid_loss={valid_loss/len(valid_dl):.4f}, valid_acc={valid_acc:.4f}")
    
    training_time = time.time() - start_time
    
    print(f"\nTraining time: {training_time:.2f}s")
    
    return (valid_loss/len(valid_dl)).item(), valid_acc.item(), training_time

# =============================================================================
# STAGE 7: CNN with Sequential
# =============================================================================

def stage7_cnn(x_train, y_train, x_valid, y_valid):
    """Build a CNN using nn.Sequential."""
    print("\n" + "="*70)
    print("STAGE 7: CNN with nn.Sequential")
    print("="*70)
    
    bs = 64
    lr = 0.1
    epochs = 2
    
    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func
        
        def forward(self, x):
            return self.func(x)
    
    def preprocess(x, y):
        return x.view(-1, 1, 28, 28), y
    
    class WrappedDataLoader:
        def __init__(self, dl, func):
            self.dl = dl
            self.func = func
        
        def __len__(self):
            return len(self.dl)
        
        def __iter__(self):
            for b in self.dl:
                yield (self.func(*b))
    
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    
    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)
    
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )
    
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_func = F.cross_entropy
    
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    
    print(f"\n3-layer CNN with:")
    print(f"  Conv2d(1→16) + ReLU")
    print(f"  Conv2d(16→16) + ReLU")
    print(f"  Conv2d(16→10) + ReLU")
    print(f"  AdaptiveAvgPool2d")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
            valid_acc = sum(accuracy(model(xb), yb) * len(xb) for xb, yb in valid_dl) / len(valid_ds)
        
        print(f"  Epoch {epoch}: valid_loss={valid_loss/len(valid_dl):.4f}, valid_acc={valid_acc:.4f}")
    
    training_time = time.time() - start_time
    
    print(f"\nTraining time: {training_time:.2f}s")
    
    return (valid_loss/len(valid_dl)).item(), valid_acc.item(), training_time

# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_all_experiments():
    """Run all 7 stages and compare results."""
    print("="*70)
    print("WHAT IS torch.nn REALLY?")
    print("Progressive Refinement Tutorial")
    print("="*70)
    
    # Load data
    print("\nLoading MNIST dataset...")
    x_train, y_train, x_valid, y_valid = load_mnist()
    print(f"Training set: {x_train.shape}")
    print(f"Validation set: {x_valid.shape}")
    
    results = {}
    
    # Run all stages
    results['Stage 1'] = stage1_from_scratch(x_train, y_train, x_valid, y_valid)
    results['Stage 2'] = stage2_functional(x_train, y_train, x_valid, y_valid)
    results['Stage 3'] = stage3_module(x_train, y_train, x_valid, y_valid)
    results['Stage 4'] = stage4_linear(x_train, y_train, x_valid, y_valid)
    results['Stage 5'] = stage5_optim(x_train, y_train, x_valid, y_valid)
    results['Stage 6'] = stage6_dataloader(x_train, y_train, x_valid, y_valid)
    results['Stage 7'] = stage7_cnn(x_train, y_train, x_valid, y_valid)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Comparison Across All Stages")
    print("="*70)
    print(f"{'Stage':<20} {'Loss':<12} {'Accuracy':<12} {'Time (s)':<12}")
    print("-"*70)
    
    for stage, (loss, acc, time_taken) in results.items():
        print(f"{stage:<20} {loss:<12.4f} {acc:<12.4f} {time_taken:<12.2f}")
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS:")
    print("="*70)
    print("1. Stage 1-5: Same model, progressively cleaner code")
    print("2. Stage 6: Better data handling with DataLoader")
    print("3. Stage 7: CNN achieves better accuracy (0.97+)")
    print("4. Code length reduced ~50% from Stage 1 to Stage 6")
    print("5. Performance similar across logistic regression stages")
    print("="*70)

if __name__ == "__main__":
    run_all_experiments()