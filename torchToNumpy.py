import torch
import numpy as np

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 2
print(a)
print(b) # same result as numpyToTorch

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu")