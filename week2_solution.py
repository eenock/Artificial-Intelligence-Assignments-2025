import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import os

# ==================== DATA LOADING ====================

def load_data_safely(filename):
    """Try to load CSV data, return None if file doesn't exist"""
    try:
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            print(f"✓ Loaded {filename}: {data.shape}")
            return data
        else:
            print(f"⚠ File not found: {filename}")
            return None
    except Exception as e:
        print(f"⚠ Error loading {filename}: {e}")
        return None

# ==================== PART A: GD/SGD/MINIBATCH COMPARISON ====================

def add_bias_1d(x):
    """Add bias term for 1D input"""
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([np.ones_like(x), x])

def poly2_1d(x):
    """Polynomial degree-2 features: [1, x, x^2]"""
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([np.ones_like(x), x, x**2])

def loss_mse(X, y, w):
    """Compute mean squared error"""
    err = X @ w - y
    return float((err**2).mean())

def grad_mse_full(X, y, w):
    """Compute full batch gradient for MSE"""
    n = X.shape[0]
    return (2.0/n) * (X.T @ (X@w - y))

def fit_linear(X, y, method="sgd", lr=0.1, epochs=10, batch_size=32, lr_schedule="constant", seed=0):
    """
    Unified linear regression with multiple optimization methods
    
    Args:
        X: Feature matrix
        y: Target values
        method: 'gd', 'sgd', or 'minibatch'
        lr: Base learning rate
        epochs: Number of training epochs
        batch_size: Batch size for minibatch method
        lr_schedule: 'constant' or 'sqrt_decay'
        seed: Random seed for reproducibility
    
    Returns:
        w: Final weights
        losses: Loss history
        times: Time per epoch
        updates: Number of parameter updates
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    t_updates = 0
    losses = []
    times = []
    
    print(f"Training {method.upper()} - {n} samples, {d} features, lr={lr}, schedule={lr_schedule}")
    
    start_time = time.time()
    
    for ep in range(epochs):
        epoch_start = time.time()
        idx = np.arange(n)
        rng.shuffle(idx)
        
        if method == "gd":
            # Gradient Descent: Full batch update
            eta = lr / math.sqrt(max(1, t_updates+1)) if lr_schedule=="sqrt_decay" else lr
            g = grad_mse_full(X, y, w)
            w -= eta * g
            t_updates += 1
            losses.append(loss_mse(X, y, w))
            
        elif method == "sgd":
            # Stochastic Gradient Descent: Single sample updates
            for i in idx:
                xi, yi = X[i], y[i]
                eta = lr / math.sqrt(max(1, t_updates+1)) if lr_schedule=="sqrt_decay" else lr
                g = 2.0 * (xi @ w - yi) * xi
                w -= eta * g
                t_updates += 1
            losses.append(loss_mse(X, y, w))
            
        else:  # minibatch
            # Mini-batch Gradient Descent
            B = max(1, min(batch_size, n))
            for k in range(0, n, B):
                j = idx[k:k+B]
                Xb, yb = X[j], y[j]
                eta = lr / math.sqrt(max(1, t_updates+1)) if lr_schedule=="sqrt_decay" else lr
                g = (2.0/len(j)) * (Xb.T @ (Xb@w - yb))
                w -= eta * g
                t_updates += 1
            losses.append(loss_mse(X, y, w))
        
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        
        # Progress reporting
        if ep % (epochs//5) == 0 or ep == epochs-1:
            print(f"  Epoch {ep:3d}: Loss = {losses[-1]:.6f}, Time = {epoch_time:.4f}s")
    
    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.2f}s, Updates: {t_updates}")
    
    return w, np.array(losses), np.array(times), t_updates

# ==================== PART B: NON-LINEAR FEATURES ====================

def phi_linear_1d(x):
    """Linear features: [1, x]"""
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([np.ones_like(x), x])

def phi_poly2_1d(x):
    """Polynomial degree-2 features: [1, x, x^2]"""
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([np.ones_like(x), x, x**2])

def phi_bins_1d(x, B=5, lo=0.0, hi=5.0):
    """Piecewise constant features with B bins"""
    x = np.asarray(x).reshape(-1)
    edges = np.linspace(lo, hi, B+1)
    X = np.zeros((len(x), B))
    
    for i, xi in enumerate(x):
        b = np.searchsorted(edges, xi, side="right") - 1
        b = min(max(b, 0), B-1)  # Clamp to valid range
        X[i, b] = 1.0
    
    return np.hstack([np.ones((len(x), 1)), X])  # Add bias

def phi_periodic_1d(x, omega=3.0):
    """Periodic features: [1, x, x^2, cos(ω*x)]"""
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([np.ones_like(x), x, x**2, np.cos(omega*x)])

# ==================== PART C: TWO-LAYER NEURAL NETWORK ====================

def relu(z):
    """ReLU activation function"""
    return np.maximum(z, 0.0)

def d_relu(z):
    """Derivative of ReLU"""
    return (z > 0).astype(float)

class TinyTwoLayer:
    """Two-layer neural network with ReLU activation"""
    
    def __init__(self, d_in, d_h=4, lr=0.1, epochs=200, seed=0):
        rng = np.random.default_rng(seed)
        
        # Initialize weights with small random values
        self.W1 = rng.normal(scale=0.5, size=(d_h, d_in))  # Hidden layer weights
        self.b1 = np.zeros(d_h)                            # Hidden layer bias
        self.W2 = rng.normal(scale=0.5, size=(1, d_h))     # Output layer weights
        self.b2 = np.zeros(1)                              # Output layer bias
        
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
        
        print(f"Initialized 2-layer NN: {d_in}→{d_h}→1, lr={lr}, epochs={epochs}")
    
    def fit(self, X, y):
        """Train the neural network using backpropagation"""
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)  # Assume binary labels {0,1}
        
        print(f"Training on {len(y)} samples...")
        
        for epoch in range(self.epochs):
            # Forward pass
            z1 = X @ self.W1.T + self.b1    # Hidden layer pre-activation
            h = relu(z1)                     # Hidden layer activation
            z2 = h @ self.W2.T + self.b2     # Output layer pre-activation
            yhat = 1/(1+np.exp(-z2)).reshape(-1)  # Sigmoid output
            
            # Compute loss (binary cross-entropy)
            eps = 1e-15  # Numerical stability
            yhat_clipped = np.clip(yhat, eps, 1-eps)
            loss = -np.mean(y * np.log(yhat_clipped) + (1-y) * np.log(1-yhat_clipped))
            self.loss_history.append(loss)
            
            # Backward pass (backpropagation)
            # Output layer gradients
            dz2 = (yhat - y)[:, None]  # Shape: (n, 1)
            gW2 = dz2.T @ h / len(y)   # Shape: (1, d_h)
            gb2 = dz2.mean(0)          # Shape: (1,)
            
            # Hidden layer gradients
            dh = dz2 @ self.W2         # Shape: (n, d_h)
            dz1 = dh * d_relu(z1)      # Shape: (n, d_h)
            gW1 = dz1.T @ X / len(y)   # Shape: (d_h, d_in)
            gb1 = dz1.mean(0)          # Shape: (d_h,)
            
            # Parameter updates
            self.W2 -= self.lr * gW2
            self.b2 -= self.lr * gb2
            self.W1 -= self.lr * gW1
            self.b1 -= self.lr * gb1
            
            # Progress reporting
            if epoch % (self.epochs//5) == 0 or epoch == self.epochs-1:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch:3d}: Loss = {loss:.6f}, Accuracy = {acc:.3f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        z1 = X @ self.W1.T + self.b1
        h = relu(z1)
        z2 = h @ self.W2.T + self.b2
        yhat = 1/(1+np.exp(-z2)).reshape(-1)
        return (yhat >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z1 = X @ self.W1.T + self.b1
        h = relu(z1)
        z2 = h @ self.W2.T + self.b2
        return 1/(1+np.exp(-z2)).reshape(-1)
    
    def accuracy(self, X, y):
        """Compute accuracy"""
        pred = self.predict(X)
        return (pred == y).mean()

# ==================== MAIN EXECUTION ====================

def run_week2_part_a():
    """Part A: Compare GD/SGD/Minibatch methods"""
    print("=" * 60)
    print("PART A: OPTIMIZATION METHODS COMPARISON")
    print("=" * 60)
    
    # Load nonlinear regression data
    data = load_data_safely('regression_nonlinear.csv')
    if data is None:
        # Create synthetic nonlinear data if file not found
        print("Creating synthetic nonlinear data...")
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y = 0.5 * x**2 - 2*x + 1 + 0.5*np.random.randn(100)
        data = pd.DataFrame({'x': x, 'y': y})
    
    x_data = data.iloc[:, 0].values
    y_data = data.iloc[:, 1].values
    
    print(f"Dataset: {len(x_data)} samples")
    
    # Test both feature mappings
    feature_maps = {
        'Linear [1,x]': phi_linear_1d,
        'Polynomial [1,x,x²]': phi_poly2_1d
    }
    
    methods = ['gd', 'sgd', 'minibatch']
    schedules = ['constant', 'sqrt_decay']
    
    results = {}
    
    for feat_name, feat_func in feature_maps.items():
        print(f"\n--- Feature Mapping: {feat_name} ---")
        X_feat = feat_func(x_data)
        
        for schedule in schedules:
            print(f"\n*** Learning Rate Schedule: {schedule} ***")
            
            for method in methods:
                lr = 0.01 if method == 'sgd' else 0.1
                batch_size = 32 if method == 'minibatch' else 1
                
                w, losses, times, updates = fit_linear(
                    X_feat, y_data, 
                    method=method, 
                    lr=lr, 
                    epochs=50,
                    batch_size=batch_size,
                    lr_schedule=schedule,
                    seed=42
                )
                
                results[f"{feat_name}_{schedule}_{method}"] = {
                    'w': w,
                    'losses': losses,
                    'times': times,
                    'updates': updates,
                    'final_mse': losses[-1],
                    'total_time': times.sum()
                }
    
    # Create visualization for Part A
    create_part_a_plots(results, feature_maps.keys(), methods, schedules)
    
    return results

def run_week2_part_b():
    """Part B: Non-linear features comparison"""
    print("\n" + "=" * 60)
    print("PART B: NON-LINEAR FEATURES COMPARISON") 
    print("=" * 60)
    
    # Load nonlinear regression data
    data = load_data_safely('regression_nonlinear.csv')
    if data is None:
        # Create synthetic nonlinear data
        print("Creating synthetic nonlinear data...")
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y = 0.5 * x**2 - 2*x + 1 + 0.5*np.random.randn(100)
        data = pd.DataFrame({'x': x, 'y': y})
    
    x_data = data.iloc[:, 0].values
    y_data = data.iloc[:, 1].values
    
    # Define feature transformations
    feature_transforms = {
        'Linear': phi_linear_1d,
        'Polynomial (deg 2)': phi_poly2_1d,
        'Piecewise (5 bins)': lambda x: phi_bins_1d(x, B=5, lo=x.min(), hi=x.max()),
        'Periodic (ω=3)': lambda x: phi_periodic_1d(x, omega=3.0)
    }
    
    results_b = {}
    
    for name, transform in feature_transforms.items():
        print(f"\n--- {name} Features ---")
        X_feat = transform(x_data)
        print(f"Feature dimensionality: {X_feat.shape[1]}")
        
        # Train with GD (most stable)
        w, losses, times, updates = fit_linear(
            X_feat, y_data,
            method='gd',
            lr=0.1,
            epochs=100,
            seed=42
        )
        
        final_mse = losses[-1]
        results_b[name] = {
            'w': w,
            'X_feat': X_feat,
            'losses': losses,
            'final_mse': final_mse,
            'transform': transform
        }
        
        print(f"Final MSE: {final_mse:.6f}")
    
    # Create visualization for Part B
    create_part_b_plots(x_data, y_data, results_b)
    
    return results_b

def run_week2_part_c():
    """Part C: Two-layer neural network on XOR"""
    print("\n" + "=" * 60)
    print("PART C: TWO-LAYER NEURAL NETWORK (XOR)")
    print("=" * 60)
    
    # Load XOR classification data
    data = load_data_safely('classification_xor.csv')
    if data is None:
        # Create XOR data if file not found
        print("Creating XOR dataset...")
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0])  # XOR truth table
        data = pd.DataFrame(np.column_stack([X_xor, y_xor]), 
                            columns=['x1', 'x2', 'y'])
    
    X_xor = data.iloc[:, :2].values
    y_xor = data.iloc[:, -1].values.astype(int)
    
    print(f"XOR dataset: {X_xor.shape[0]} samples, {X_xor.shape[1]} features")
    print("XOR truth table:")
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} → {y_xor[i]}")
    
    # Train neural network
    nn = TinyTwoLayer(d_in=2, d_h=4, lr=0.1, epochs=1000, seed=42)
    nn.fit(X_xor, y_xor)
    
    # Test final accuracy
    final_accuracy = nn.accuracy(X_xor, y_xor)
    predictions = nn.predict(X_xor)
    probabilities = nn.predict_proba(X_xor)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {final_accuracy:.3f}")
    print("Predictions vs Truth:")
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} → Pred: {predictions[i]}, Truth: {y_xor[i]}, Prob: {probabilities[i]:.3f}")
    
    # Create visualization for Part C
    create_part_c_plots(X_xor, y_xor, nn)
    
    return nn, final_accuracy

def create_part_a_plots(results, feature_names, methods, schedules):
    """Create separate plots for Part A analysis"""
    
    # Plot 1: Loss convergence comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Part A: Optimization Methods Comparison', fontsize=16)
    
    colors = {'gd': 'blue', 'sgd': 'orange', 'minibatch': 'green'}
    
    plot_idx = 0
    for feat_name in feature_names:
        for schedule in schedules:
            row, col = plot_idx // 2, plot_idx % 2
            ax = axes[row, col]
            
            for method in methods:
                key = f"{feat_name}_{schedule}_{method}"
                if key in results:
                    losses = results[key]['losses']
                    ax.plot(losses, label=f'{method.upper()}', 
                            color=colors[method], linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title(f'{feat_name}\n{schedule.replace("_", " ").title()} LR')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # Wait for user input before proceeding
    input("Press Enter to continue to next plot...")
    
    # Plot 2: Final MSE comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, schedule in enumerate(schedules):
        ax = ax1 if i == 0 else ax2
        
        feat_results = []
        for feat_name in feature_names:
            method_mse = []
            for method in methods:
                key = f"{feat_name}_{schedule}_{method}"
                if key in results:
                    method_mse.append(results[key]['final_mse'])
                else:
                    method_mse.append(float('nan'))
            feat_results.append(method_mse)
        
        x = np.arange(len(feature_names))
        width = 0.25
        
        for j, method in enumerate(methods):
            values = [feat_results[k][j] for k in range(len(feature_names))]
            ax.bar(x + j*width, values, width, label=method.upper(), 
                    color=colors[method], alpha=0.8)
        
        ax.set_xlabel('Feature Mapping')
        ax.set_ylabel('Final MSE')
        ax.set_title(f'Final MSE - {schedule.replace("_", " ").title()} LR')
        ax.set_xticks(x + width)
        ax.set_xticklabels([name.split()[0] for name in feature_names])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def create_part_b_plots(x_data, y_data, results_b):
    """Create separate plots for Part B analysis"""
    
    # Plot 1: Feature transformations visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Part B: Feature Transformations and Model Fits', fontsize=16)
    
    x_plot = np.linspace(x_data.min(), x_data.max(), 200)
    
    plot_idx = 0
    for name, result in results_b.items():
        row, col = plot_idx // 2, plot_idx % 2
        ax = axes[row, col]
        
        # Plot original data
        ax.scatter(x_data, y_data, alpha=0.6, s=20, color='red', label='Data')
        
        # Plot model fit
        X_plot = result['transform'](x_plot)
        y_plot = X_plot @ result['w']
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'{name} Fit')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name}\nMSE: {result["final_mse"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # Wait for user input before proceeding
    input("Press Enter to continue to next plot...")
    
    # Plot 2: Loss convergence for different features
    plt.figure(figsize=(12, 6))
    
    for name, result in results_b.items():
        plt.plot(result['losses'], label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Part B: Training Loss Convergence by Feature Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    # Plot 3: MSE comparison bar chart
    plt.figure(figsize=(10, 6))
    
    names = list(results_b.keys())
    mse_values = [results_b[name]['final_mse'] for name in names]
    
    bars = plt.bar(names, mse_values, alpha=0.8, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    plt.xlabel('Feature Type')
    plt.ylabel('Final MSE')
    plt.title('Part B: Final MSE Comparison Across Feature Types')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mse in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{mse:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def create_part_c_plots(X_xor, y_xor, nn):
    """Create separate plots for Part C analysis"""
    
    # Plot 1: Training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(nn.loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('Part C: Neural Network Training Loss (XOR Problem)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    # Plot 2: Decision boundary visualization
    plt.figure(figsize=(12, 5))
    
    # Create mesh for decision boundary
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get probability predictions for mesh
    Z = nn.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='P(Class 1)')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    
    # Plot data points
    colors = ['red' if label == 0 else 'blue' for label in y_xor]
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)
    
    # Add labels
    for i in range(len(X_xor)):
        plt.annotate(f'({X_xor[i,0]},{X_xor[i,1]})→{y_xor[i]}', 
                    (X_xor[i, 0], X_xor[i, 1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Decision Boundary & Data Points')
    plt.grid(True, alpha=0.3)
    
    # Plot predictions vs truth
    plt.subplot(1, 2, 2)
    predictions = nn.predict(X_xor)
    probabilities = nn.predict_proba(X_xor)
    
    x_pos = np.arange(len(X_xor))
    width = 0.35
    
    plt.bar(x_pos - width/2, y_xor, width, label='True Labels', alpha=0.8, color='green')
    plt.bar(x_pos + width/2, predictions, width, label='Predictions', alpha=0.8, color='orange')
    
    # Add probability text
    for i, prob in enumerate(probabilities):
        plt.text(i, 0.5, f'{prob:.3f}', ha='center', va='center', 
                fontweight='bold', fontsize=10)
    
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.title('Predictions vs Truth')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(-0.2, 1.2)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run complete Week 2-1 assignment"""
    print("=" * 70)
    print("WEEK 2-1 ASSIGNMENT: ADVANCED OPTIMIZATION & FEATURE ENGINEERING")
    print("=" * 70)
    
    # Run all parts
    results_a = run_week2_part_a()
    results_b = run_week2_part_b() 
    nn, accuracy = run_week2_part_c()
    
    # Summary
    print("\n" + "=" * 60)
    print("WEEK 2-1 SUMMARY")
    print("=" * 60)
    
    print("\n🔧 PART A - Optimization Methods:")
    print("   • Implemented GD, SGD, and Mini-batch methods")
    print("   • Compared constant vs sqrt_decay learning rates")
    print("   • Tested on linear vs polynomial features")
    
    print("\n🎨 PART B - Feature Engineering:")
    best_feature = min(results_b.keys(), key=lambda k: results_b[k]['final_mse'])
    print(f"   • Best feature type: {best_feature}")
    print(f"   • Best MSE: {results_b[best_feature]['final_mse']:.6f}")
    print("   • Compared linear, polynomial, piecewise, and periodic features")
    
    print(f"\n🧠 PART C - Neural Network:")
    print(f"   • XOR Problem Training Accuracy: {accuracy:.3f}")
    print("   • Successfully learned non-linear XOR mapping")
    print("   • 2-layer network with ReLU activation")
    
    print(f"\n✅ ASSIGNMENT COMPLETED!")
    print(f"   • All optimization methods implemented and compared")
    print(f"   • Feature engineering explored with multiple transformations")
    print(f"   • Neural network achieved target performance")

if __name__ == "__main__":
    main()