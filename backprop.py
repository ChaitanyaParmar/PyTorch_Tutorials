# 1. Forward Pass : Compute Loss
# 2. Compute Local Gradients
# 3. Backward Pass : Compute dLoss/dWeights using the Chain Rule

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# Forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss)

# backward pass
loss.backward()
print(w.grad)