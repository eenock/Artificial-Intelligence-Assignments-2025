import numpy as np
import matplotlib.pyplot as plt

# ==================== PART R1: LINEAR REGRESSION ====================

def add_bias(x):
    """Add bias term (column of 1s) to input features"""
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([np.ones_like(x), x])

def fit_linear_gd(X, y, lr=0.1, epochs=200):
    """
    Fit linear regression using Gradient Descent
    
    Args:
        X: Feature matrix (n_samples x n_features)
        y: Target values (n_samples,)
        lr: Learning rate
        epochs: Number of training epochs
    
    Returns:
        w: Learned weights
        hist: Loss history over epochs
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights to zero
    hist = []
    
    for epoch in range(epochs):
        # Forward pass: compute predictions
        pred = X @ w
        
        # Compute residual and loss
        err = pred - y
        loss = (err**2).mean()  # Mean Squared Error
        
        # Compute gradient: ∇w = (2/n) * X^T * (X*w - y)
        grad = (2.0/n) * (X.T @ err)
        
        # Update weights: w := w - η * ∇w
        w -= lr * grad
        
        # Store loss for plotting
        hist.append(loss)
        
        # Print progress every 50 epochs
        if epoch % 50 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    return w, np.array(hist)

def fit_linear_sgd(X, y, lr=0.1, epochs=200):
    """
    Fit linear regression using Stochastic Gradient Descent (SGD)
    
    Args:
        X: Feature matrix (n_samples x n_features)  
        y: Target values (n_samples,)
        lr: Learning rate
        epochs: Number of training epochs
        
    Returns:
        w: Learned weights
        hist: Loss history over epochs
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    hist = []
    
    for epoch in range(epochs):
        # Shuffle data for each epoch
        indices = np.random.permutation(n)
        
        epoch_loss = 0
        for i in indices:
            # Single sample gradient
            xi, yi = X[i], y[i]
            pred_i = xi @ w
            err_i = pred_i - yi
            
            # SGD update using single sample
            grad_i = 2 * err_i * xi
            w -= lr * grad_i
            
            epoch_loss += err_i**2
        
        # Average loss over epoch
        avg_loss = epoch_loss / n
        hist.append(avg_loss)
        
        if epoch % 50 == 0 or epoch == epochs-1:
            print(f"SGD Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    return w, np.array(hist)

# ==================== PART C1: LINEAR CLASSIFICATION ====================

def fit_hinge_gd(X, y, lr=0.1, epochs=200, l2=0.0):
    """
    Fit linear classifier using Hinge Loss with subgradient descent
    
    Args:
        X: Feature matrix (n_samples x n_features)
        y: Binary labels in {-1, +1}
        lr: Learning rate
        epochs: Number of training epochs
        l2: L2 regularization parameter
        
    Returns:
        w: Learned weights
        hist: Loss history over epochs
        acc: Final 0-1 accuracy
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    hist = []
    
    for epoch in range(epochs):
        # Compute margins: y_i * (w^T * x_i)
        margins = (X @ w) * y
        
        # Subgradient computation
        # For hinge loss: max(0, 1 - margin)
        # Subgradient is -y_i * x_i when margin < 1, else 0
        mask = margins < 1.0
        
        if np.any(mask):
            # Average subgradient over violated constraints
            grad = -(X[mask].T @ y[mask]) / n + l2 * w
        else:
            grad = l2 * w
        
        # Compute hinge loss value
        hinge_losses = np.maximum(1 - margins, 0)
        loss = hinge_losses.mean() + 0.5 * l2 * np.dot(w, w)
        
        # Update weights
        w -= lr * grad
        hist.append(loss)
        
        if epoch % 50 == 0 or epoch == epochs-1:
            print(f"Hinge Epoch {epoch}: Loss = {loss:.6f}")
    
    # Compute final 0-1 accuracy
    predictions = np.sign(X @ w)
    acc = (predictions == y).mean()
    
    return w, np.array(hist), acc

# ==================== DEMO AND TESTING ====================

def demo_regression():
    """Demo linear regression on toy dataset from slides"""
    print("=== LINEAR REGRESSION DEMO ===")
    
    # Toy dataset from slides
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([1.0, 3.0, 3.0])
    X = add_bias(x)
    
    print(f"Input features X:\n{X}")
    print(f"Target values y: {y}")
    
    # Fit using Gradient Descent
    print("\n--- Gradient Descent ---")
    w_gd, hist_gd = fit_linear_gd(X, y, lr=0.1, epochs=200)
    print(f"Final weights w*: {w_gd}")
    print(f"Final loss: {hist_gd[-1]:.6f}")
    
    # Fit using SGD for comparison
    print("\n--- Stochastic Gradient Descent ---")
    w_sgd, hist_sgd = fit_linear_sgd(X, y, lr=0.05, epochs=200)
    print(f"Final weights w*: {w_sgd}")  
    print(f"Final loss: {hist_sgd[-1]:.6f}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist_gd, label='Gradient Descent', linewidth=2)
    plt.plot(hist_sgd, label='SGD', linewidth=2, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot fit
    plt.subplot(1, 2, 2)
    x_plot = np.linspace(0.5, 4.5, 100)
    X_plot = add_bias(x_plot)
    y_pred_gd = X_plot @ w_gd
    y_pred_sgd = X_plot @ w_sgd
    
    plt.scatter(x, y, color='red', s=100, label='Data', zorder=5)
    plt.plot(x_plot, y_pred_gd, 'b-', label=f'GD fit: y = {w_gd[1]:.2f}x + {w_gd[0]:.2f}')
    plt.plot(x_plot, y_pred_sgd, 'g--', label=f'SGD fit: y = {w_sgd[1]:.2f}x + {w_sgd[0]:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_classification():
    """Demo linear classification on toy dataset from slides"""
    print("\n\n=== LINEAR CLASSIFICATION DEMO ===")
    
    # Toy points from slides
    X = np.array([[0.0, 2.0],
                    [-2.0, 0.0], 
                    [1.0, -1.0]])
    y = np.array([+1, +1, -1])
    
    print(f"Input features X:\n{X}")
    print(f"Binary labels y: {y}")
    
    # Fit using Hinge Loss
    print("\n--- Hinge Loss Classification ---")
    w_hinge, hist_hinge, acc = fit_hinge_gd(X, y, lr=0.1, epochs=50)
    print(f"Final weights w*: {w_hinge}")
    print(f"Final accuracy: {acc:.3f}")
    print(f"Final hinge loss: {hist_hinge[-1]:.6f}")
    
    # With L2 regularization
    print("\n--- With L2 Regularization (λ=0.01) ---")
    w_reg, hist_reg, acc_reg = fit_hinge_gd(X, y, lr=0.1, epochs=50, l2=0.01)
    print(f"Regularized weights w*: {w_reg}")
    print(f"Regularized accuracy: {acc_reg:.3f}")
    print(f"Final regularized loss: {hist_reg[-1]:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(hist_hinge, label='No regularization', linewidth=2)
    plt.plot(hist_reg, label='L2 reg (λ=0.01)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Hinge Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Decision boundary without regularization
    plt.subplot(1, 3, 2)
    plot_decision_boundary(X, y, w_hinge, title='No Regularization')
    
    # Decision boundary with regularization  
    plt.subplot(1, 3, 3)
    plot_decision_boundary(X, y, w_reg, title='L2 Regularization')
    
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X, y, w, title=""):
    """Plot 2D data points and decision boundary"""
    # Create a mesh for plotting decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    
    # Compute decision function values
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mesh_points @ w
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary (where w^T x = 0)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', linewidths=2)
    
    # Plot data points
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolors='black')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂') 
    plt.title(title)
    plt.grid(True, alpha=0.3)

if __name__ == "__main__":
    # Run demonstrations
    demo_regression()
    demo_classification()
    
    print("\n=== ASSIGNMENT COMPLETED ===")
    print("✓ Implemented fit_linear_gd() with squared loss")
    print("✓ Implemented fit_linear_sgd() for comparison")
    print("✓ Implemented fit_hinge_gd() with subgradient descent")
    print("✓ Added L2 regularization support")
    print("✓ Generated loss plots and visualizations")
    print("\nNext steps:")
    print("1. Test on data/regression_toy.csv if available")
    print("2. Test on data/classification_toy.csv if available")
    print("3. Write your report following the template structure")