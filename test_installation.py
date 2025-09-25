import torch
import torchvision
import numpy as np

print("=== PyTorch Installation Verification ===")
print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

# Test basic tensor operations
print("\n=== Testing Basic Operations ===")
x = torch.rand(3, 3)
y = torch.rand(3, 3)
z = x + y
print("✓ Tensor creation and addition successful")
print(f"Sample tensor shape: {x.shape}")

# Test gradients
print("\n=== Testing Autograd ===")
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print("✓ Gradient computation successful")
print(f"Gradients: {x.grad}")

print("\n=== Installation Verification Complete ===")