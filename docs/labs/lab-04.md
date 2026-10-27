# Lab 4: CNNs in PyTorch

## Layers

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AvgPool2d(kernel_size=2, stride=2)
```

PyTorch images use shape **(N, C, H, W)**: batch size, channels, height, width.

| Input | Conv2d | Weights | Output |
|---|---|---|---|
| (1, 1, 5, 5) | `in=1, out=3, k=2` | (3, 1, 2, 2) | (1, 3, 4, 4) |
| (1, 3, 5, 5) | `in=3, out=5, k=2` | (5, 3, 2, 2) | (1, 5, 4, 4) |

`out_channels` = number of filters. Each filter spans all input channels.

## What CNNs Are Used For

| Task | Output | Examples |
|---|---|---|
| **Visual recognition** | one class for the whole image | medical diagnosis, galaxy images, game-playing AI |
| **Image segmentation** | one class for **every pixel** | self-driving cars, face parsing, satellite land maps |

Since AlexNet (2012), CNNs have won the ImageNet competition every year.

## Example: MNIST with a CNN

**Data**: scale with `StandardScaler` (flatten to 2D first, then reshape back to $28\times28$). First 100 training samples → validation, remaining 900 → training. Reshape to `(N, 1, 28, 28)`.

**Model**

```
(1, 28, 28) → conv 5×5, 16, p=2 → ReLU → maxpool 2 → (16, 14, 14)
            → conv 5×5, 32, p=2 → ReLU → maxpool 2 → (32, 7, 7)
            → flatten 1568 → FC → 10
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)   # "same": 28 → 28
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)                        # 28 → 14
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)  # 14 → 14
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)                        # 14 → 7
        self.fc1 = nn.Linear(32 * 7 * 7, 10)                               # 1568 → 10

    def forward(self, x):
        x = self.maxpool1(F.relu(self.cnn1(x)))
        x = self.maxpool2(F.relu(self.cnn2(x)))
        x = x.flatten(start_dim=1)       # (N, 32, 7, 7) → (N, 1568)
        return self.fc1(x)               # raw logits

torch.manual_seed(0)                     # reproducible results
model = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
```

**Mini-batch training** with `torch.split`:

```python
x_batches = torch.split(X_train, 100)    # mini-batch size 100
y_batches = torch.split(y_train, 100)

for epoch in range(15):
    for xb, yb in zip(x_batches, y_batches):
        optimizer.zero_grad()
        loss = loss_func(model(xb), yb)
        loss.backward()
        optimizer.step()
    # compute validation accuracy once per epoch
```

With mini-batches you can record the training loss **per iteration** instead of per epoch.

Result: about **94%** test accuracy.

## Assignment: Fashion-MNIST, Beat Humans

10 clothing classes (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot), $28\times28$ grayscale, 10000 train / 1000 test. Harder than digits.

1. Normalize and split into train/val/test.
2. Design your own CNN (conv/pool layers + MLP head).
3. Goal: **≥ 90% test accuracy within 30 epochs** (humans: 83.5%).
4. Plot training loss and validation accuracy, print test accuracy.
5. Print **per-class accuracy**. Which class is worst, and how would you improve it?

**Tips**

- Start with a **small baseline** network.
- Tune mini-batch size and learning rate **before** changing the architecture.
- Add one component at a time (an extra conv/pool layer, an FC layer).
- Baseline optimizer: Adam with `lr = 3e-4`.
- Stop early when validation accuracy stops improving.
- Borrow ideas from LeNet-5 and AlexNet.

## Popular Image Datasets

| Dataset | Size | Used for |
|---|---|---|
| ImageNet | 14M+ images, 21K+ categories | classification benchmark (ILSVRC) |
| Google Open Images | 9M+ images, 600 box classes | detection, segmentation, relationships |
| COCO | 330K images, 1.5M objects | detection, segmentation, captioning |
