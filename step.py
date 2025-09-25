import torch
import matplotlib.pyplot as plt

# Set seed for reproducible results
torch.manual_seed(42)

# Create dataset
X = torch.randn(100, 1) * 2
true_w, true_b = 3.5, -2.0
noise = torch.randn(100, 1) * 0.5
y = true_w * X + true_b + noise

# Initialize parameters
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(f"Initial: w = {w.item():.4f}, b = {b.item():.4f}")

# Training
learning_rate = 0.01
costs = []

for epoch in range(1000):
    y_pred = w * X + b
    cost = torch.mean((y_pred - y) ** 2)
    costs.append(cost.item())
    
    cost.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

print(f"Final: w = {w.item():.6f}, b = {b.item():.6f}")
print(f"True:  w = {true_w}, b = {true_b}")
print(f"Final cost: {costs[-1]:.8f}")

# Create visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X.numpy(), y.numpy(), alpha=0.6)
x_line = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = w * x_line + b
plt.plot(x_line.numpy(), y_line.detach().numpy(), 'r-')
plt.title('Fitted Line')

plt.subplot(1, 2, 2)
plt.plot(costs)
plt.title('Cost Function')
plt.yscale('log')
plt.show()