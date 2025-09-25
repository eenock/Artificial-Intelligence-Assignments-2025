import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

# ==================== PART A: TWO-LAYER BACKPROP (SQUARED LOSS) ====================

def sigmoid(z):
    """Sigmoid activation function"""
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1/(1+np.exp(-z))

def forward(phi, V, w, y):
    """
    Forward pass for two-layer network with squared loss
    h = σ(Vφ(x)), s = w·h, L = (s-y)²
    """
    z1 = V @ phi              # Pre-activation of hidden layer
    h = sigmoid(z1)           # Hidden layer activations  
    s = float(w @ h)          # Output (scalar)
    residual = s - float(y)   # Residual
    loss = residual**2        # Squared loss
    
    return (phi, V, w, y, z1, h, s, residual, loss)

def backward(cache):
    """
    Backward pass computing gradients analytically
    """
    phi, V, w, y, z1, h, s, residual, loss = cache
    
    # Gradient w.r.t. output weights: ∂L/∂w = 2(s-y)h
    grad_w = 2.0 * residual * h
    
    # Gradient w.r.t. hidden weights: ∂L/∂V = 2(s-y)w·h·(1-h)·φᵀ
    # Chain rule: ∂L/∂V = ∂L/∂s · ∂s/∂h · ∂h/∂z1 · ∂z1/∂V
    g = 2.0 * residual * (w * h * (1-h))  # Gradient flowing to hidden layer
    grad_V = np.outer(g, phi)             # Outer product for weight matrix
    
    return grad_w, grad_V, loss

def finite_difference_check(phi, V, w, y, eps=1e-5):
    """
    Check analytical gradients against finite differences
    """
    print("=== Finite Difference Gradient Check ===")
    
    # Forward pass
    cache = forward(phi, V, w, y)
    grad_w_analytical, grad_V_analytical, loss = backward(cache)
    
    # Finite difference for w
    grad_w_fd = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy(); w_plus[i] += eps
        w_minus = w.copy(); w_minus[i] -= eps
        
        _, _, _, _, _, _, _, _, loss_plus = forward(phi, V, w_plus, y)
        _, _, _, _, _, _, _, _, loss_minus = forward(phi, V, w_minus, y)
        
        grad_w_fd[i] = (loss_plus - loss_minus) / (2 * eps)
    
    # Finite difference for V
    grad_V_fd = np.zeros_like(V)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V_plus = V.copy(); V_plus[i,j] += eps
            V_minus = V.copy(); V_minus[i,j] -= eps
            
            _, _, _, _, _, _, _, _, loss_plus = forward(phi, V_plus, w, y)
            _, _, _, _, _, _, _, _, loss_minus = forward(phi, V_minus, w, y)
            
            grad_V_fd[i,j] = (loss_plus - loss_minus) / (2 * eps)
    
    # Compare gradients
    w_diff = np.abs(grad_w_analytical - grad_w_fd)
    V_diff = np.abs(grad_V_analytical - grad_V_fd)
    
    print(f"Gradient w:")
    print(f"  Analytical: {grad_w_analytical}")
    print(f"  Finite Diff: {grad_w_fd}")
    print(f"  Max difference: {w_diff.max():.8f}")
    
    print(f"Gradient V:")
    print(f"  Analytical:\n{grad_V_analytical}")
    print(f"  Finite Diff:\n{grad_V_fd}")
    print(f"  Max difference: {V_diff.max():.8f}")
    
    # Check if gradients match within tolerance
    w_correct = w_diff.max() < 1e-5
    V_correct = V_diff.max() < 1e-5
    
    print(f"\nGradient Check Results:")
    print(f"  w gradient correct: {w_correct}")
    print(f"  V gradient correct: {V_correct}")
    print(f"  Overall: {'PASS' if w_correct and V_correct else 'FAIL'}")
    
    return w_correct and V_correct

# ==================== PART B: K-MEANS WITH RESTARTS & K++ ====================

def kmeans_pp_init(X, K, rng):
    """
    K-means++ initialization for better cluster initialization
    """
    n = X.shape[0]
    centers = np.empty((K, X.shape[1]))
    
    # Choose first center randomly
    i0 = rng.integers(n)
    centers[0] = X[i0]
    
    # Initialize squared distances
    d2 = np.full(n, np.inf)
    
    # Choose remaining centers
    for k in range(1, K):
        # Update distances to nearest center
        d2 = np.minimum(d2, ((X - centers[k-1])**2).sum(1))
        
        # Choose next center with probability proportional to squared distance
        probs = d2 / d2.sum()
        i = rng.choice(n, p=probs)
        centers[k] = X[i]
    
    return centers

def kmeans_single_run(X, K, init_centers, max_iter=100):
    """
    Single run of K-means algorithm
    """
    centers = init_centers.copy()
    n, d = X.shape
    
    for iteration in range(max_iter):
        # Assign points to nearest centers
        distances = ((X[:,None,:] - centers[None,:,:])**2).sum(-1)  # (n, K)
        assignments = distances.argmin(1)  # (n,)
        
        # Update centers
        new_centers = centers.copy()
        for k in range(K):
            idx = np.where(assignments == k)[0]
            if len(idx) > 0:
                new_centers[k] = X[idx].mean(0)
        
        # Check convergence
        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        
        centers = new_centers
    
    # Compute final loss (sum of squared distances)
    final_distances = ((X[:,None,:] - centers[None,:,:])**2).sum(-1)
    final_assignments = final_distances.argmin(1)
    loss = ((X - centers[final_assignments])**2).sum()
    
    return final_assignments, centers, float(loss), iteration + 1

def kmeans(X, K, init='random', restarts=10, max_iter=100, seed=0):
    """
    K-means with multiple restarts and different initialization methods
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    n, d = X.shape
    
    best_result = None
    all_losses = []
    
    print(f"Running K-means: K={K}, init={init}, restarts={restarts}")
    
    for restart in range(restarts):
        # Initialize centers
        if init == 'k++':
            init_centers = kmeans_pp_init(X, K, rng)
        else:  # random initialization
            x_min, x_max = X.min(0), X.max(0)
            init_centers = rng.uniform(x_min, x_max, size=(K, d))
        
        # Run K-means
        assignments, centers, loss, iterations = kmeans_single_run(X, K, init_centers, max_iter)
        all_losses.append(loss)
        
        # Keep best result
        if best_result is None or loss < best_result[2]:
            best_result = (assignments, centers, loss, iterations)
        
        if restart % max(1, restarts//5) == 0 or restart == restarts-1:
            print(f"  Restart {restart:2d}: Loss = {loss:.4f}, Iterations = {iterations}")
    
    print(f"Best loss: {best_result[2]:.4f} (converged in {best_result[3]} iterations)")
    
    return best_result, all_losses

# ==================== PART C: VALIDATION & REGULARIZATION ====================

def add_bias_1d(x):
    """Add bias term for 1D regression"""
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([np.ones_like(x), x])

def mse_loss(X, y, w):
    """Compute MSE loss"""
    err = X @ w - y
    return float((err**2).mean())

def fit_ridge(X, y, lam=0.0, lr=0.1, epochs=300, early_stop=False, patience=20, X_val=None, y_val=None, verbose=False):
    """
    Ridge regression with optional early stopping
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    
    # Track best validation performance
    best = (np.inf, w.copy())
    wait = 0
    
    train_losses = []
    val_losses = []
    
    for ep in range(epochs):
        # Compute gradient with L2 regularization
        grad = (2.0/n) * (X.T @ (X@w - y)) + lam * w
        w -= lr * grad
        
        # Track training loss
        train_loss = mse_loss(X, y, w)
        train_losses.append(train_loss)
        
        # Validation and early stopping
        if X_val is not None and y_val is not None:
            val_loss = mse_loss(X_val, y_val, w)
            val_losses.append(val_loss)
            
            # Check for improvement
            if val_loss < best[0] - 1e-10:
                best = (val_loss, w.copy())
                wait = 0
            else:
                wait += 1
                if early_stop and wait >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {ep}, patience exceeded")
                    break
        
        if verbose and ep % (epochs//5) == 0:
            val_str = f", Val: {val_losses[-1]:.6f}" if X_val is not None else ""
            print(f"    Epoch {ep:3d}: Train: {train_loss:.6f}{val_str}")
    
    final_w = best[1] if early_stop and X_val is not None else w
    return final_w, train_losses, val_losses

def grid_search_ridge(X_train, y_train, X_val, y_val, lambda_values, early_stop=True, verbose=True):
    """
    Grid search for optimal lambda with cross-validation
    """
    results = {}
    
    print("=== Ridge Regression Grid Search ===")
    
    for lam in lambda_values:
        print(f"\nTesting λ = {lam}")
        
        w, train_losses, val_losses = fit_ridge(
            X_train, y_train, 
            lam=lam, 
            lr=0.01,  # Conservative learning rate
            epochs=500,
            early_stop=early_stop,
            patience=20,
            X_val=X_val,
            y_val=y_val,
            verbose=verbose
        )
        
        final_train_mse = mse_loss(X_train, y_train, w)
        final_val_mse = mse_loss(X_val, y_val, w)
        
        results[lam] = {
            'weights': w,
            'train_mse': final_train_mse,
            'val_mse': final_val_mse,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"  Final - Train MSE: {final_train_mse:.6f}, Val MSE: {final_val_mse:.6f}")
    
    # Find best lambda
    best_lambda = min(results.keys(), key=lambda l: results[l]['val_mse'])
    best_val_mse = results[best_lambda]['val_mse']
    
    print(f"\n*** Best λ = {best_lambda}, Validation MSE = {best_val_mse:.6f} ***")
    
    return results, best_lambda

# ==================== VISUALIZATION FUNCTIONS ====================

def plot_backprop_demo():
    """Demonstrate backpropagation with visualization"""
    print("\n=== PART A: Two-Layer Backprop Demo ===")
    
    # Create simple example
    np.random.seed(42)
    phi = np.array([1.0, 2.0])  # Input features [bias, x]
    V = np.random.randn(3, 2) * 0.5  # Hidden layer weights (3 hidden units)
    w = np.random.randn(3) * 0.5     # Output layer weights
    y = 1.5  # Target value
    
    print(f"Input φ(x): {phi}")
    print(f"Hidden weights V:\n{V}")
    print(f"Output weights w: {w}")
    print(f"Target y: {y}")
    
    # Forward pass
    cache = forward(phi, V, w, y)
    phi, V, w, y, z1, h, s, residual, loss = cache
    
    print(f"\nForward pass:")
    print(f"  Hidden pre-activation z1: {z1}")
    print(f"  Hidden activation h: {h}")
    print(f"  Output s: {s:.4f}")
    print(f"  Loss: {loss:.4f}")
    
    # Backward pass
    grad_w, grad_V, _ = backward(cache)
    print(f"\nBackward pass:")
    print(f"  ∂L/∂w: {grad_w}")
    print(f"  ∂L/∂V:\n{grad_V}")
    
    # Gradient check
    gradient_correct = finite_difference_check(phi, V, w, y)
    
    return gradient_correct

def plot_kmeans_demo():
    """Demonstrate K-means with different initialization methods"""
    print("\n=== PART B: K-means Demo ===")
    
    # Generate synthetic 2D data with clear clusters
    np.random.seed(42)
    
    # Create 3 clusters
    cluster1 = np.random.randn(30, 2) + [2, 2]
    cluster2 = np.random.randn(30, 2) + [-2, -2] 
    cluster3 = np.random.randn(30, 2) + [2, -2]
    
    X = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.hstack([np.zeros(30), np.ones(30), np.full(30, 2)])
    
    print(f"Generated dataset: {X.shape[0]} points, 3 true clusters")
    
    # Test both initialization methods
    methods = ['random', 'k++']
    results = {}
    
    for method in methods:
        print(f"\n--- {method.upper()} Initialization ---")
        
        (best_assignments, best_centers, best_loss, best_iters), all_losses = kmeans(
            X, K=3, init=method, restarts=10, max_iter=100, seed=42
        )
        
        results[method] = {
            'assignments': best_assignments,
            'centers': best_centers,
            'loss': best_loss,
            'iterations': best_iters,
            'all_losses': all_losses
        }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('K-means Clustering Comparison', fontsize=16)
    
    # Plot original data
    ax = axes[0, 0]
    colors = ['red', 'blue', 'green']
    for i in range(3):
        mask = true_labels == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, label=f'True Cluster {i}')
    ax.set_title('True Clusters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot results for each method
    for idx, method in enumerate(methods):
        ax = axes[0, 1+idx]
        result = results[method]
        
        # Plot points colored by assignment
        for i in range(3):
            mask = result['assignments'] == i
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6)
        
        # Plot centers
        ax.scatter(result['centers'][:, 0], result['centers'][:, 1], 
                    c='black', marker='x', s=200, linewidths=3, label='Centers')
        ax.set_title(f'{method.upper()} Init (Loss: {result["loss"]:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot loss comparison across restarts
    ax = axes[1, 0]
    for method in methods:
        ax.plot(results[method]['all_losses'], 'o-', label=f'{method.upper()}', linewidth=2)
    ax.set_xlabel('Restart')
    ax.set_ylabel('Final Loss')
    ax.set_title('Loss Across Restarts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "K-means Results Summary:\n\n"
    for method in methods:
        result = results[method]
        min_loss = min(result['all_losses'])
        max_loss = max(result['all_losses'])
        std_loss = np.std(result['all_losses'])
        summary_text += f"{method.upper()} Initialization:\n"
        summary_text += f"  Best Loss: {min_loss:.4f}\n"
        summary_text += f"  Worst Loss: {max_loss:.4f}\n"
        summary_text += f"  Std Dev: {std_loss:.4f}\n"
        summary_text += f"  Avg Iterations: {result['iterations']:.1f}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    input("Press Enter to continue to next plot...")
    
    return results

def plot_regularization_demo():
    """Demonstrate ridge regression with validation"""
    print("\n=== PART C: Ridge Regression with Validation ===")
    
    # Load or generate data
    try:
        data = pd.read_csv('regression_nonlinear.csv')
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        print(f"Loaded regression data: {len(x)} samples")
    except:
        print("Generating synthetic polynomial data...")
        np.random.seed(42)
        x = np.linspace(0, 1, 80)
        y = 0.5 + 1.5*x - 2*x**2 + 0.5*np.random.randn(len(x))
    
    # Add polynomial features
    X = add_bias_1d(x)
    X = np.hstack([X, x.reshape(-1, 1)**2])  # [1, x, x²]
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training set: {len(y_train)} samples")
    print(f"Validation set: {len(y_val)} samples")
    print(f"Feature dimensionality: {X.shape[1]} (including bias)")
    
    # Grid search over lambda values
    lambda_values = [0, 1e-3, 1e-2, 1e-1]
    
    results, best_lambda = grid_search_ridge(
        X_train, y_train, X_val, y_val, 
        lambda_values, early_stop=True, verbose=True
    )
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Ridge Regression with Validation', fontsize=16)
    
    # Plot training/validation curves for each lambda
    ax = axes[0, 0]
    for lam in lambda_values:
        result = results[lam]
        epochs = range(len(result['train_losses']))
        ax.plot(epochs, result['train_losses'], '--', alpha=0.7, label=f'Train λ={lam}')
        if result['val_losses']:
            ax.plot(epochs, result['val_losses'], '-', linewidth=2, label=f'Val λ={lam}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training/Validation Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot final MSE comparison
    ax = axes[0, 1]
    train_mses = [results[lam]['train_mse'] for lam in lambda_values]
    val_mses = [results[lam]['val_mse'] for lam in lambda_values]
    
    x_pos = np.arange(len(lambda_values))
    width = 0.35
    
    ax.bar(x_pos - width/2, train_mses, width, label='Training MSE', alpha=0.8)
    ax.bar(x_pos + width/2, val_mses, width, label='Validation MSE', alpha=0.8)
    
    ax.set_xlabel('λ (Regularization)')
    ax.set_ylabel('Final MSE')
    ax.set_title('Final MSE by Regularization Strength')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{lam}' for lam in lambda_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Plot model fits
    ax = axes[1, 0]
    x_plot = np.linspace(x.min(), x.max(), 200)
    X_plot = add_bias_1d(x_plot)
    X_plot = np.hstack([X_plot, x_plot.reshape(-1, 1)**2])
    
    # Plot original data
    ax.scatter(x, y, alpha=0.6, s=20, color='gray', label='Data')
    
    # Plot fits for different lambdas
    colors = ['blue', 'orange', 'green', 'red']
    for i, lam in enumerate(lambda_values):
        result = results[lam]
        y_plot = X_plot @ result['weights']
        label = f'λ={lam}' + (' (Best)' if lam == best_lambda else '')
        linewidth = 3 if lam == best_lambda else 2
        ax.plot(x_plot, y_plot, color=colors[i], linewidth=linewidth, label=label)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Model Fits for Different λ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "Ridge Regression Results:\n\n"
    summary_text += f"Best λ: {best_lambda}\n"
    summary_text += f"Best Val MSE: {results[best_lambda]['val_mse']:.6f}\n\n"
    
    summary_text += "All Results:\n"
    for lam in lambda_values:
        result = results[lam]
        summary_text += f"λ={lam:6.3f}: "
        summary_text += f"Train={result['train_mse']:.4f}, "
        summary_text += f"Val={result['val_mse']:.4f}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    input("Press Enter to continue to next plot...")
    
    return results, best_lambda

# ==================== MAIN EXECUTION ====================

def main():
    """Run complete Week 2-2 assignment"""
    print("=" * 70)
    print("WEEK 2-2 ASSIGNMENT: BACKPROP, K-MEANS, VALIDATION")
    print("=" * 70)
    
    # Part A: Backpropagation
    backprop_success = plot_backprop_demo()
    
    # Part B: K-means clustering  
    kmeans_results = plot_kmeans_demo()
    
    # Part C: Ridge regression with validation
    ridge_results, best_lambda = plot_regularization_demo()
    
    # Final summary
    print("\n" + "=" * 60)
    print("WEEK 2-2 SUMMARY")
    print("=" * 60)
    
    print(f"\n🔧 PART A - Backpropagation:")
    print(f"   • Gradient check: {'PASSED' if backprop_success else 'FAILED'}")
    print(f"   • Implemented analytical gradients for two-layer network")
    print(f"   • Verified with finite difference approximation")
    
    print(f"\n🎯 PART B - K-means Clustering:")
    random_loss = kmeans_results['random']['loss']
    kpp_loss = kmeans_results['k++']['loss']
    improvement = (random_loss - kpp_loss) / random_loss * 100
    print(f"   • Random init final loss: {random_loss:.4f}")
    print(f"   • K++ init final loss: {kpp_loss:.4f}")
    print(f"   • K++ improvement: {improvement:.1f}%")
    
    print(f"\n📊 PART C - Ridge Regression:")
    print(f"   • Best λ: {best_lambda}")
    print(f"   • Best validation MSE: {ridge_results[best_lambda]['val_mse']:.6f}")
    print(f"   • Early stopping and grid search implemented")
    
    print(f"\n✅ ASSIGNMENT COMPLETED!")
    print(f"   • Backpropagation with gradient verification")
    print(f"   • K-means with multiple initialization strategies") 
    print(f"   • Ridge regression with validation and regularization")

if __name__ == "__main__":
    main()