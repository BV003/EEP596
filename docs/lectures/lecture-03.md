# Lecture 3: Optimization in Deep Learning

## GD vs. SGD

The three variants differ only in **how many samples one update uses**.

| | SGD | Mini-batch GD | Batch GD |
|---|---|---|---|
| Samples per iteration | 1 | $N/m$ (a mini-batch) | all $N$ |
| Path to minimum | noisy, zig-zag | mostly smooth | smooth |
| Parallel on GPU | hard | yes | maximum |
| Main drawback | oscillates near the minimum | — | slow per step, may not fit in memory |

::: tip In practice
**Mini-batch GD** is the default. It is fast (parallel) and stable enough. "SGD" in libraries usually means mini-batch GD.
:::

With SGD the loss curve is **noisy**, but it still converges, and the noise can even help it find a shortcut.

## Learning Rate

- Too small → very slow training.
- Too large → loss jumps around or diverges.
- **Variable learning rate**: start large, then shrink it as training gets close to the minimum.

## Optimizers

| Optimizer | Idea in one line |
|---|---|
| SGD | fixed step $\alpha$ along the gradient |
| Momentum | keep a running average of gradients, like a ball rolling downhill → faster, less zig-zag |
| AdaGrad | a separate learning rate for each weight, shrinks for weights that got big gradients |
| RMSProp | AdaGrad with a moving average, so the learning rate does not shrink to zero |
| **Adam** | Momentum + RMSProp. **Standard choice** in deep learning |

Momentum and Adam updates (for reference):

$$v \leftarrow \beta v + (1-\beta)\nabla w, \qquad w \leftarrow w - \alpha v$$

$$\text{Adam: } w \leftarrow w - \alpha \frac{\hat v}{\sqrt{\hat s} + \epsilon}, \quad v = \text{avg of } \nabla w,\; s = \text{avg of } (\nabla w)^2$$

::: info Optimizer vs. optimization techniques
The **optimizer** is just the update rule. **Optimization techniques** are everything else that helps training: data splitting, regularization, normalization, batch norm, initialization, hyperparameter tuning.
:::

## Train / Validation / Test

| Set | Used for | When |
|---|---|---|
| Train | computing the loss and updating weights | during training |
| Validation | checking progress, choosing hyperparameters | during training |
| Test | final, honest score | only once, after training |

Common ratios: 70/15/15, 80/10/10, 60/20/20.

## Underfitting vs. Overfitting

| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| Train accuracy | bad | good | great |
| Test accuracy | bad | good | bad |
| Problem | **high bias** (model too simple) | — | **high variance** (memorized the training set) |

| Fix underfitting | Fix overfitting |
|---|---|
| more layers / neurons | more training data, data augmentation |
| train longer | L1 / L2 regularization |
| better architecture | dropout |
| tune hyperparameters | early stopping |

## Regularization

Add a penalty on large weights to the loss:

$$J_{\text{reg}} = J + \lambda \sum |w| \;\;(\text{L1}), \qquad J_{\text{reg}} = J + \lambda \sum w^2 \;\;(\text{L2})$$

For a multi-layer network L2 sums over all layers: $\lambda \sum_l \lVert W^{[l]} \rVert_F^2$ (Frobenius norm).

| L1 | L2 |
|---|---|
| pushes many weights to exactly 0 → **sparse** model | keeps all weights small → **dense** model |
| robust to outliers | sensitive to outliers |
| weaker on complex patterns | learns complex patterns |

**Dropout**: in each training step, randomly switch off neurons with probability $p$. No neuron can rely on one specific other neuron, so the network learns more robust features. Effect is similar to L2. **Turn dropout off at test time.**

**Data augmentation**: make more training data from what you have: add noise, rotate, flip, resize, distort.

**Early stopping**: watch the validation loss. When it stops going down (while the training loss keeps going down), stop training.

## Vanishing / Exploding Gradients

In a deep network the gradient is a product of many per-layer factors. If they are $< 1$ the gradient **vanishes** (early layers stop learning); if $> 1$ it **explodes** (training diverges).

Three remedies:

1. **Normalize the inputs**: zero mean, unit variance, $x \leftarrow (x - \mu)/\sigma$. Features on the same scale give a rounder loss surface and more stable steps. Use the training-set $\mu, \sigma$ for **all** sets (train, val, test).
2. **Batch normalization**: do the same normalization *inside* the network, on each layer's output, using the statistics of the current mini-batch (mean ≈ 0, std ≈ 1).
3. **Good weight initialization**:

| Init | Variance of $w$ | Use with |
|---|---|---|
| all zeros | — | ✗ every neuron learns the same thing |
| $\mathcal N(0, 1)$ | 1 | ✗ too large in deep nets |
| Xavier | $1/n_{\text{in}}$ | tanh |
| He (Kaiming) | $2/n_{\text{in}}$ | ReLU |

## Hyperparameter Tuning

Hyperparameters: learning rate, number of layers, neurons per layer, activation, batch size, optimizer, number of epochs.

- **Grid search**: try every combination on a fixed grid.
- **Random search**: try random combinations. Usually better, because it tests more distinct values of the parameters that matter most (often the learning rate).
