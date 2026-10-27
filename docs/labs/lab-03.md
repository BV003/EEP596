# Lab 3: MLP in PyTorch

## Shallow vs. Deep

| Shallow (1 hidden layer) | Deep (> 1 hidden layer) |
|---|---|
| can approximate any function, given enough neurons | upper layers reuse features from lower layers |
| easier to train | harder to train |
| simple problems | complex problems |

## Classification Basics

**Softmax** turns raw outputs (logits) into probabilities that sum to 1:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad (5.0,\ 2.5) \to (0.92,\ 0.08)$$

**One-hot encoding** turns a class label into a vector: class 2 of 5 → `[0, 0, 1, 0, 0]` (`torch.nn.functional.one_hot`).

::: tip CrossEntropyLoss does both for you
`nn.CrossEntropyLoss()` applies softmax internally and accepts **integer labels** (`int64`) directly. So the model returns **raw logits** and you do not one-hot encode the targets.
:::

| Task | Loss |
|---|---|
| Regression | `nn.MSELoss()` |
| Classification | `nn.CrossEntropyLoss()` |

## Data Split

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = StandardScaler().fit_transform(X)                       # zero mean, unit variance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
# final: train 60%, val 20%, test 20%
```

## Three Ways to Loop

Same training step, different amounts of data per step:

```python
for epoch in range(epochs):
    for xb, yb in batches:          # SGD: batch = 1 sample, Batch GD: one batch = all data
        optimizer.zero_grad()
        loss = loss_func(model(xb), yb)
        loss.backward()
        optimizer.step()
```

| | Iterations per epoch |
|---|---|
| SGD | $N$ (number of samples) |
| Mini-batch | $N / \text{batch size}$ |
| Batch GD | 1 |

## Building Blocks

| What | PyTorch |
|---|---|
| Activations | `F.relu` `F.leaky_relu` `torch.tanh` `torch.sigmoid` |
| Optimizers | `optim.SGD` `optim.Adam` `optim.RMSprop` `optim.Adadelta` |
| L2 regularization | `optim.Adam(..., weight_decay=λ)` |
| L1 regularization | add it to the loss yourself (below) |
| Dropout | `nn.Dropout(p)`, $p$ = probability a neuron is zeroed |
| Batch norm | `nn.BatchNorm1d(num_features)`, applied **before** the activation |
| Init | `nn.init.xavier_uniform_` (tanh), `nn.init.kaiming_uniform_` (ReLU) |

```python
# L1: add λ·Σ|w| to the loss
l1 = sum(p.abs().sum() for p in model.parameters())
loss = loss_func(outputs, targets) + 1e-4 * l1
```

::: info Default init is fine
PyTorch already initializes layers well. Manual initialization is mostly for experiments.
:::

## Example: Iris Classification

150 flowers, 4 features (sepal/petal length and width), 3 species.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden1_dim=30, hidden2_dim=10, output_dim=3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden1_dim)
        self.layer2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.layer3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)        # raw logits, CrossEntropyLoss adds softmax

model = MLP()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train).long()   # class labels must be int64
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val).long()

train_loss, val_acc = [], []
for epoch in range(25):
    optimizer.zero_grad()
    loss = loss_func(model(X_train_t), y_train_t)
    train_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = model(X_val_t).argmax(dim=1)          # class with highest score
        val_acc.append((pred == y_val_t).float().mean().item())
```

Result: about **93%** test accuracy.

## Assignment: MNIST with an MLP

Handwritten digits 0–9, $28\times28$ grayscale (1000 train, 100 test).

1. Flatten each image to 784 values, scale with `StandardScaler`, split train/val/test.
2. Design your own MLP (layers, activation, optimizer, ...).
3. Goal: **> 90% test accuracy within 50 epochs**.
4. Plot training loss and validation accuracy, print test accuracy.
5. Show the misclassified test images with the model's guesses, and explain why they failed and how to fix it.

**Tips**

- First decide: layers, neurons, activation, batch size, learning rate, optimizer, epochs.
- If it overfits (train good, val/test bad): add dropout, add regularization, stop early, or make the network smaller.
