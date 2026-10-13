# Lab 2: PyTorch Introduction

## Neural Net Workflow

1. Prepare Data
2. Define Model
3. Select Hyperparameters
4. Train Model
5. Evaluate

## Example: Learn $y = 2x + 1$

```python
import torch

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(1, 1)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()                     # clear old gradients
    loss = loss_func(model(inputs), targets)  # forward pass
    loss.backward()                           # back-propagation
    optimizer.step()                          # update weights

with torch.no_grad():
    predicted = model(inputs)
```

## Tensors

Tensors are NumPy-like arrays that run on CPU or GPU and support autograd.

```python
t = torch.from_numpy(arr)   # NumPy -> Tensor
arr = t.numpy()             # Tensor -> NumPy
t = t.cuda()                # move to GPU
```

## Assignment: XOR from Scratch

XOR is not linearly separable, so it needs a hidden layer. Implement a one-hidden-layer network with NumPy only.

$$z = \sigma(W_1 x + b_1), \quad y = \sigma(W_2 z + b_2), \quad J = \sum (y - t)^2$$

$$\frac{\partial J}{\partial W_2} = 2(y-t)\,y(1-y)\,z, \qquad \frac{\partial J}{\partial W_1} = 2(y-t)\,y(1-y)\,W_2\,z(1-z)\,x$$

$$W \leftarrow W - \alpha \frac{\partial J}{\partial W}$$
