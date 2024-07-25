import torch

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output  = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_() # empty the grads before next iteration

## Does the same thing as above

# optimizer = torch.optim.SGD(weights, lr = 0.01)
# optimizer.step()
# optimizer.zero_grad()