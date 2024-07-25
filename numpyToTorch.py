import torch
import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))

a.add_(1)
print(a)
print(b) # by changing original tensor it affects both

