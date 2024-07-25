import torch

x = torch.randn(3, requires_grad=True) # the requires_grad attribute is used to indicate whether a tensor should track operations and compute gradients with respect to it during the backpropagation process. This is essential for training neural networks, as gradients are used to update the weights.
# print(x)

y = x + 2
# print(y)

z = y * 2
# print(z)
         
# z = z.mean()
# print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32)
z.backward(v) # starts the backward pass , dz/dx
print(x.grad) # prints all the values in backward pass