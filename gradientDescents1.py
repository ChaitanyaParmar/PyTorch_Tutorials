## Prediction : Manually
## Gradients Computation : Manually
## Loss Computation : Manually
## Paramenter Updates : Manually

import numpy as np

# f = w * x
# f = 2 * x

x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

## model prediction
def forward(x):
    return w * x

## loss
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

## gradient
# Mean Squared Error = 1/N * (w*x-y)**2
# dJ/dW = 1/N 2x (w*x-y)

def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

## Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(x)

    # loss
    l = loss(y, y_pred)

    # gradients
    dw = gradient(x,y,y_pred)

    # update weights
    w -= learning_rate*dw

    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
