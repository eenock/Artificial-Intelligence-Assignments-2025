import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================== UTILITY FUNCTIONS ====================

def add_bias(x):
    """Add bias term (column of 1s) to input features"""
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([np.ones_like(x), x])

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

# ==================== PART R1: LINEAR REGRESSION ====================

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
    
    print(f"Training linear regression: {n} samples, {d} features")
    print(f"Learning rate: {lr}, Epochs: {epochs}")
    
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
        
        # Print progress
        if epoch % 40 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch:3d}: Loss = {loss:.6f}")
    
    return w, np.array(hist)

def fit_linear_sgd(X, y, lr=0.1, epochs=200, batch_size=1):
    """
    Fit linear regression using Stochastic Gradient Descent
    
    Args:
        X: Feature matrix (n_samples x n_features)  
        y: Target values (n_samples,)
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size (1 for pure SGD, >1 for mini-batch)
        
    Returns:
        w: Learned weights
        hist: Loss history over epochs
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    hist = []
    
    method_name = "SGD" if batch_size == 1 else f"Mini-batch GD (batch_size={batch_size})"
    print(f"Training {method_name}: {n} samples, {d} features")
    
    for epoch in range(epochs):
        # Shuffle data for each epoch
        indices = np.random.permutation(n)
        
        epoch_loss = 0
        for i in range(0, n, batch_size):
            # Get batch indices
            batch_indices = indices[i:i+batch_size]
            
            # Batch data
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Compute batch gradient
            pred_batch = X_batch @ w
            err_batch = pred_batch - y_batch
            grad_batch = (2.0/len(batch_indices)) * (X_batch.T @ err_batch)
            
            # Update weights
            w -= lr * grad_batch
            
            epoch_loss += (err_batch**2).sum()
        
        # Average loss over epoch
        avg_loss = epoch_loss / n
        hist.append(avg_loss)
        
        if epoch % 40 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
    
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
    
    reg_str = f", L2={l2}" if l2 > 0 else ""
    print(f"Training hinge loss classifier: {n} samples, {d} features{reg_str}")
    
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
        
        if epoch % 40 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch:3d}: Loss = {loss:.6f}")
    
    # Compute final 0-1 accuracy
    predictions = np.sign(X @ w)
    # Handle case where prediction is exactly 0
    predictions[predictions == 0] = 1
    acc = (predictions == y).mean()
    
    return w, np.array(hist), acc

# ==================== MAIN EXECUTION ====================

def run_week1_assignment():
    """Run the complete Week 1 assignment"""
    print("=" * 60)
    print("WEEK 1 ASSIGNMENT: LINEAR REGRESSION & CLASSIFICATION")
    print("=" * 60)
    
    # ========== PART R1: LINEAR REGRESSION ==========
    print("\n" + "="*50)
    print("PART R1: LINEAR REGRESSION")
    print("="*50)
    
    # Try to load regression data
    regression_data = load_data_safely('regression_hand.csv')
    if regression_data is None:
        regression_data = load_data_safely('data/regression_toy.csv')
    
    if regression_data is not None:
        # Use loaded data
        if 'x' in regression_data.columns and 'y' in regression_data.columns:
            x_data = regression_data['x'].values
            y_data = regression_data['y'].values
        else:
            # Assume first column is x, second is y
            x_data = regression_data.iloc[:, 0].values
            y_data = regression_data.iloc[:, 1].values
        
        print(f"Using dataset: {len(x_data)} samples")
        X_reg = add_bias(x_data)
    else:
        # Use toy dataset from assignment
        print("Using toy dataset from assignment slides")
        x_data = np.array([1.0, 2.0, 4.0])
        y_data = np.array([1.0, 3.0, 3.0])
        X_reg = add_bias(x_data)
    
    print(f"Features shape: {X_reg.shape}")
    print(f"Targets shape: {y_data.shape}")
    
    # Gradient Descent
    print("\n--- Gradient Descent ---")
    w_gd, hist_gd = fit_linear_gd(X_reg, y_data, lr=0.1, epochs=200)
    print(f"Final weights w* = {w_gd}")
    print(f"Final loss = {hist_gd[-1]:.6f}")
    
    # SGD Comparison
    print("\n--- Stochastic Gradient Descent ---")
    w_sgd, hist_sgd = fit_linear_sgd(X_reg, y_data, lr=0.05, epochs=200)
    print(f"SGD weights w* = {w_sgd}")
    print(f"SGD final loss = {hist_sgd[-1]:.6f}")
    
    # Mini-batch GD
    if len(y_data) >= 4:
        print("\n--- Mini-batch Gradient Descent ---")
        w_mb, hist_mb = fit_linear_sgd(X_reg, y_data, lr=0.08, epochs=200, batch_size=min(4, len(y_data)//2))
        print(f"Mini-batch weights w* = {w_mb}")
        print(f"Mini-batch final loss = {hist_mb[-1]:.6f}")
    else:
        hist_mb = hist_sgd  # For plotting consistency
    
    # ========== PART C1: LINEAR CLASSIFICATION ==========
    print("\n" + "="*50)
    print("PART C1: LINEAR CLASSIFICATION")
    print("="*50)
    
    # Try to load classification data
    classification_data = load_data_safely('data/classification_toy.csv')
    
    if classification_data is not None:
        # Use loaded data
        if 'x1' in classification_data.columns:
            X_clf = classification_data[['x1', 'x2']].values
            y_clf = classification_data['y'].values
        else:
            # Assume first two columns are features, last is label
            X_clf = classification_data.iloc[:, :2].values
            y_clf = classification_data.iloc[:, -1].values
        
        # Convert labels to {-1, +1} if needed
        if set(np.unique(y_clf)) == {0, 1}:
            y_clf = 2 * y_clf - 1
        
        print(f"Using dataset: {len(y_clf)} samples")
    else:
        # Use toy dataset from assignment
        print("Using toy dataset from assignment slides")
        X_clf = np.array([[0.0, 2.0],
                            [-2.0, 0.0], 
                            [1.0, -1.0]])
        y_clf = np.array([+1, +1, -1])
    
    print(f"Features shape: {X_clf.shape}")
    print(f"Labels shape: {y_clf.shape}")
    print(f"Unique labels: {np.unique(y_clf)}")
    
    # Hinge Loss without regularization
    print("\n--- Hinge Loss (No Regularization) ---")
    w_hinge, hist_hinge, acc = fit_hinge_gd(X_clf, y_clf, lr=0.1, epochs=100)
    print(f"Final weights w* = {w_hinge}")
    print(f"Final accuracy = {acc:.3f}")
    print(f"Final hinge loss = {hist_hinge[-1]:.6f}")
    
    # Hinge Loss with L2 regularization
    print("\n--- Hinge Loss (L2 Regularization λ=0.01) ---")
    w_reg, hist_reg, acc_reg = fit_hinge_gd(X_clf, y_clf, lr=0.1, epochs=100, l2=0.01)
    print(f"Regularized weights w* = {w_reg}")
    print(f"Regularized accuracy = {acc_reg:.3f}")
    print(f"Final regularized loss = {hist_reg[-1]:.6f}")
    
    # ========== PLOTTING RESULTS ==========
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(15, 10))
    
    # Regression loss curves
    plt.subplot(2, 3, 1)
    plt.plot(hist_gd, label='Gradient Descent', linewidth=2)
    plt.plot(hist_sgd, label='SGD', linewidth=2, alpha=0.7)
    if len(y_data) >= 4:
        plt.plot(hist_mb, label='Mini-batch GD', linewidth=2, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Regression: Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Regression fit
    plt.subplot(2, 3, 2)
    x_plot = np.linspace(min(x_data)-0.5, max(x_data)+0.5, 100)
    X_plot = add_bias(x_plot)
    y_pred_gd = X_plot @ w_gd
    
    plt.scatter(x_data, y_data, color='red', s=100, label='Data', zorder=5, edgecolors='black')
    plt.plot(x_plot, y_pred_gd, 'b-', linewidth=2, 
                label=f'GD: y = {w_gd[1]:.2f}x + {w_gd[0]:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Classification loss curves
    plt.subplot(2, 3, 4)
    plt.plot(hist_hinge, label='No regularization', linewidth=2)
    plt.plot(hist_reg, label='L2 reg (λ=0.01)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Hinge Loss')
    plt.title('Classification: Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Classification decision boundaries
    if X_clf.shape[1] == 2:  # Only plot for 2D data
        # No regularization
        plt.subplot(2, 3, 5)
        plot_decision_boundary(X_clf, y_clf, w_hinge, title='No Regularization')
        
        # With regularization
        plt.subplot(2, 3, 6)
        plot_decision_boundary(X_clf, y_clf, w_reg, title='L2 Regularization')
    
    plt.tight_layout()
    plt.show()
    
    # ========== SUMMARY RESULTS ==========
    print("\n" + "="*50)
    print("ASSIGNMENT SUMMARY")
    print("="*50)
    
    print("\n📊 REGRESSION RESULTS:")
    print(f"   • GD final loss:        {hist_gd[-1]:.6f}")
    print(f"   • SGD final loss:       {hist_sgd[-1]:.6f}")
    if len(y_data) >= 4:
        print(f"   • Mini-batch final loss: {hist_mb[-1]:.6f}")
    print(f"   • GD weights:           {w_gd}")
    
    print(f"\n🎯 CLASSIFICATION RESULTS:")
    print(f"   • Hinge accuracy:       {acc:.3f}")
    print(f"   • Regularized accuracy: {acc_reg:.3f}")
    print(f"   • Hinge loss (no reg):  {hist_hinge[-1]:.6f}")
    print(f"   • Hinge loss (L2 reg):  {hist_reg[-1]:.6f}")
    
    print(f"\n✅ ASSIGNMENT COMPLETED!")
    print(f"   • Base requirements: ✓ Linear regression, ✓ Hinge loss")
    print(f"   • Challenge features: ✓ SGD comparison, ✓ L2 regularization")
    
    return {
        'regression': {'w_gd': w_gd, 'w_sgd': w_sgd, 'loss_gd': hist_gd[-1], 'loss_sgd': hist_sgd[-1]},
        'classification': {'w_hinge': w_hinge, 'w_reg': w_reg, 'acc': acc, 'acc_reg': acc_reg}
    }

def plot_decision_boundary(X, y, w, title=""):
    """Plot 2D data points and decision boundary"""
    if X.shape[1] != 2:
        print("Can only plot decision boundary for 2D data")
        return
        
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
    
    # Add labels
    for i in range(len(X)):
        plt.annotate(f'{y[i]:+d}', (X[i, 0], X[i, 1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, color='white', fontweight='bold')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂') 
    plt.title(title)
    plt.grid(True, alpha=0.3)

if __name__ == "__main__":
    # Run the complete Week 1 assignment
    results = run_week1_assignment()
    
    print(f"\n📝 NEXT STEPS:")
    print(f"   1. Copy results into your report template")
    print(f"   2. Add mathematical derivations (gradient formulas)")
    print(f"   3. Include the generated plots")
    print(f"   4. Write analysis and insights section")
    print(f"   5. Submit paper version with curves/tables")