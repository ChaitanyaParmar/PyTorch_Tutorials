import torch

x = torch.randn(3, requires_grad=True)
print(x)

## How to prevent gradient from tracking gradients

# 1. x.requires_grad_(False)

x. requires_grad_(False) 
print(x)


# 2. x.detach()

y = x.detach()
print(y)

# 3. with torch.no_grad()

with torch.no_grad():
    y = x + 2
    print(y)