import torch

# x = torch.ones(2, 3, dtype = torch.float16)
# x = torch.tensor([2.5, 0.1])
# print(x)

a = torch.rand(2,2)
b = torch.rand(2,2)
# c = a + b
# c = torch.add(a,b) # sub, mul, div
b.add_(a) # sub_, mul_,
# print(b)

x = torch.rand(5,3)
# print(x[1,1].item()) # Prints a single item


x = torch.rand(4,4)
print(x)
# y = x.view(16) # no of elements must be same
y = x.view(-1,2) # dynamic way
print(y)