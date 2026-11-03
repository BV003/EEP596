# Lab 5: ResNet in PyTorch

Theory (degradation, residual blocks, bottlenecks, inception) is in [Lecture 5](/lectures/lecture-05). This lab uses torchvision's built-in ResNet-18.

## Example: MNIST with ResNet-18

### Prepare Data

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),                        # PIL image → (C, H, W) float in [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean and std
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
```

Unlike Lab 4, the data comes from **torchvision** and normalization is a transform applied to each image as it is loaded.

### Define Model

```python
import torch
from torchvision.models import resnet18

class MNIST_ResNet_Classifier(torch.nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.resnet = resnet18()          # untrained ResNet-18 (expects 3-channel input, 1000 classes)

        # Replace the first conv: MNIST has 1 input channel, not 3
        self.resnet.conv1 = torch.nn.Conv2d(
            1, out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            bias=False,                   # followed by batch norm, so no bias needed
        )
        # Replace the last layer: 10 classes, not 1000
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 10)   # in_features = 512

    def forward(self, x):
        return self.resnet(x)             # raw logits
```

::: tip Editing a pre-built model
Every part of `resnet18()` is an attribute you can replace: `conv1`, `bn1`, `maxpool`, `layer1`–`layer4`, `avgpool`, `fc`. Print the model to see them all.
:::

### Hyperparameters

```python
model = MNIST_ResNet_Classifier(out_channels=64, kernel_size=7, stride=2, padding=3)  # same as original conv1

learning_rate = 0.001
epochs = 10
batchsize = 64

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.cuda()                              # move to GPU

train_loss_list = []                      # track training loss for plotting
```

### Train

`DataLoader` handles shuffling and mini-batching (replacing `torch.split` from Lab 4):

```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1000,      shuffle=False)

for epoch in range(epochs):
    model.train()                         # batch norm uses batch statistics
    for batch_idx, (train_input, train_target) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(train_input.cuda())  # inputs and targets must be on the same device as the model
        loss = loss_func(pred, train_target.cuda())
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
    print(f"Train Epoch: {epoch} \tLoss: {loss.item():.6f}")
```

### Evaluate

```python
model.eval()                              # batch norm uses running statistics
test_loss, correct = 0, 0

with torch.no_grad():                     # no gradients needed → faster, less memory
    for test_input, test_target in test_loader:
        output = model(test_input.cuda())
        test_loss += loss_func(output, test_target.cuda()).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(test_target.view_as(pred).cuda()).sum().item()

test_loss /= len(test_loader.dataset)
print(f"Test Accuracy: {correct}/{len(test_loader.dataset)} "
      f"({100.0 * correct / len(test_loader.dataset):.0f}%)")
```

Result: **99%** test accuracy (9921/10000) within 10 epochs, compared with ~94% for the small CNN in Lab 4.

::: warning `train()` vs `eval()`
ResNet has batch norm (and dropout in other models), which behave differently in training and evaluation. Call `model.eval()` before validating/testing and `model.train()` before training again. Forgetting to switch back after a per-epoch validation is a common bug.
:::

## Assignment: CIFAR-10 with ResNet-18

**CIFAR-10**: 60000 colour images, $32\times32\times3$, 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), 6000 per class. 50000 train / 10000 test.

1. **Normalize** the dataset and **split** into train / validation / test.
2. Optionally **augment** the training data (e.g. random flips) for better performance.
3. Build your own **ResNet-18 variant**: change channels, kernel size, stride, padding, etc.
4. Goal: **≥ 90% test accuracy within 15 epochs**.
5. Plot training loss and validation accuracy, and print test accuracy.
6. Print **per-class accuracy** (as in Lab 4). Which class is best and worst, and what might cause the low-performing classes?

### Tips

**Augmentation** applies only to the training set:

```python
mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)   # CIFAR-10 per-channel stats

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     # random shift
    transforms.RandomHorizontalFlip(),        # random mirror
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

**Adapt the stem for small images.** The original ResNet stem (7×7 conv stride 2 + max pool stride 2) shrinks the input 4×, so a $32\times32$ image becomes $8\times8$ before the residual blocks even start. For CIFAR, a common change is:

```python
self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
self.resnet.maxpool = torch.nn.Identity()     # remove the first down-sampling
```

**Validation split** with `torch.utils.data.random_split`:

```python
train_set, val_set = torch.utils.data.random_split(full_train, [45000, 5000])
```

Both subsets share `full_train`'s transform. To keep augmentation out of validation, load the training data twice (once with `train_transform`, once with `test_transform`) and split both with the same indices.

Other ideas: tune learning rate and batch size first, try a learning-rate scheduler, and stop early when validation accuracy stops improving.
