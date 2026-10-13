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

# ---------- 1. Prepare Data ----------
# PyTorch works on tensors: NumPy-like arrays that can run on GPU
# and track gradients automatically.
inputs = torch.arange(11, dtype=torch.float32).unsqueeze(1)  # shape (11, 1): x = 0..10
targets = 2 * inputs + 1                                     # y = 2x + 1 (the rule to learn)

# ---------- 2. Define Model ----------
# Every PyTorch model subclasses nn.Module. You only describe the layers
# and the forward pass; PyTorch handles parameters and gradients.
class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()  # set up nn.Module internals (parameter tracking, etc.)
        # nn.Linear holds learnable weight w and bias b and computes y = w·x + b.
        # Both start random and are registered as parameters automatically.
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Forward pass: how data flows through the network.
        # PyTorch records every operation here to build the graph used for back-prop.
        return self.linear(x)

model = LinearRegression(1, 1)  # 1 input -> 1 output, no hidden layer

# ---------- 3. Select Hyperparameters ----------
loss_func = torch.nn.MSELoss()  # loss J = mean((ŷ - y)²)
# The optimizer gets all learnable parameters (w, b) and applies
# gradient descent: param ← param - lr * grad
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ---------- 4. Train ----------
for epoch in range(100):              # 1 epoch = one pass over the training data
    optimizer.zero_grad()             # gradients accumulate by default, so clear last epoch's
    outputs = model(inputs)           # ① Forward: x -> ŷ  (calls forward())
    loss = loss_func(outputs, targets)  #   ... -> J
    loss.backward()                   # ② Backward: autograd computes ∂J/∂w, ∂J/∂b via chain rule
                                      #    and stores them in w.grad and b.grad
    optimizer.step()                  # ③ Update: w, b ← w, b - lr * grad

# ---------- 5. Evaluate ----------
# no_grad(): inference only, so skip graph building (faster, less memory)
with torch.no_grad():
    predicted = model(inputs)

print(model.linear.weight.item(), model.linear.bias.item())  # ≈ 2 and ≈ 1 once trained
```

::: tip What PyTorch does for you
You write only the **model** and the **forward pass**. PyTorch then:
- **stores parameters** (`nn.Linear` creates $w$, $b$)
- **computes gradients automatically** (`loss.backward()`, so no hand-derived back-prop as in the XOR assignment)
- **updates parameters** (`optimizer.step()`)
:::

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
