# Lecture 5: Advanced CNN Architectures

## Why Go Deeper?

More layers let a network learn more complex patterns. Modern image models often have **more than 100 layers**. But simply stacking more conv layers (as in VGG) stops working after a point.

## The Degradation Problem

When a plain CNN gets very deep, **both training and test accuracy saturate and then get worse** than a shallower network.

::: warning Not overfitting
Overfitting means low training error but high test error. In degradation the **training** error also goes up, so the deep network is failing to optimize, not memorizing.
:::

Root cause identified by He et al. (2015):

- A deeper network has a **more complex gradient landscape**, so it is harder to optimize.
- **Identity mapping deficit**: it is hard for a stack of nonlinear layers to learn even "do nothing", $f(x) = x$. In theory a deep network could copy a shallow one and pass the rest through unchanged, but in practice it does not find that solution.

Two tools for training very deep networks:

| Tool | Idea |
|---|---|
| **Skip connection** (residual block) | let the input bypass layers |
| **Data compression** (bottleneck layer) | shrink the representation before expensive operations |

## Residual Block

A normal block must learn the whole output: $y = F(x)$.

A residual block adds a **skip connection** that carries $x$ around the layers, so the layers only need to learn the **change**:

$$y = x + F(x), \qquad F(x) = \Delta x$$

```
x ──┬──> [layers F] ──> (+) ──> y
    └───────────────────┘
        skip connection
```

If the best thing to do is nothing, the block just learns $F(x) = 0$, which is easy (push weights toward zero). That fixes the identity-mapping problem.

**Benefits**
- Enables training **much deeper** networks.
- Prevents **vanishing gradients**: the gradient through the block is $\frac{\partial y}{\partial x} = 1 + \frac{\partial F}{\partial x}$, so there is always a direct path of "1" back to earlier layers.
- Reduces overfitting.

::: tip Shapes must match
$x + F(x)$ is element-wise, so $x$ and $F(x)$ need the same shape. When $F$ changes the number of channels or the spatial size, the skip path uses a **$1\times1$ conv** (with a stride) to reshape $x$.
:::

## Bottleneck Layer

Squeeze the data through a **smaller hidden layer** in the middle:

```
Input (wide) → Compression (narrow) → Output (wide)
```

- Forces a **low-dimensional representation** that keeps only the important information.
- **More computationally efficient** (fewer parameters).
- Reduces overfitting.

### $1\times1$ Convolution

A $1\times1$ conv looks at one pixel at a time but across **all channels**. Each output channel is a **linear combination of the input channels**.

- Like pooling, it shrinks the data, but in the **channel** direction instead of height/width.
- $H\times W\times C_\text{in}$ with $C_\text{out}$ filters of $1\times1\times C_\text{in}$ → $H\times W\times C_\text{out}$.

### ResNet Bottleneck Block

```
512 ch ─ 1×1 conv → 128 ch ─ 3×3 conv → 128 ch ─ 1×1 conv → 512 ch ─(+)→ 512 ch
   └──────────────────────────────────────────────────────────────┘
```

Compress by 4× ($D \to D/4$), do the expensive $3\times3$ conv on the small tensor, expand back ($D/4 \to D$), and add the skip.

| | Weights |
|---|---|
| Two $3\times3$ convs, 512 → 512 | $2 \times 3\cdot3\cdot512\cdot512 \approx 4.7\text{M}$ |
| Bottleneck 512 → 128 → 128 → 512 | $512\cdot128 + 3\cdot3\cdot128\cdot128 + 128\cdot512 \approx 0.28\text{M}$ |

About **17× fewer** weights (biases ignored).

## Inception Module

Instead of choosing one filter size, run **several convolutions in parallel** and stack the results along the channel axis (Szegedy et al., 2014).

Example: input $28\times28\times192$

| Branch | Operations | Output |
|---|---|---|
| 1 | $1\times1$ conv, 64 | $28\times28\times64$ |
| 2 | $1\times1$ conv, 96 → $3\times3$ conv, 128 | $28\times28\times128$ |
| 3 | $1\times1$ conv, 16 → $5\times5$ conv, 32 | $28\times28\times32$ |
| 4 | $3\times3$ max pool ($s=1$, same) → $1\times1$ conv, 32 | $28\times28\times32$ |
| **Concat** | | $28\times28\times256$ |

- All branches use "same" padding, so height and width match and can be concatenated.
- The $1\times1$ convs before the $3\times3$ and $5\times5$ are **bottlenecks** that keep the cost down.

::: details Why the 1×1 bottleneck matters (branch 3)
Multiplications for the $5\times5$ conv producing $28\times28\times32$:

- Directly from 192 channels: $28\cdot28\cdot32 \times 5\cdot5\cdot192 \approx 120\text{M}$
- Through a 16-channel bottleneck: $28\cdot28\cdot16 \times 192 + 28\cdot28\cdot32 \times 5\cdot5\cdot16 \approx 2.4\text{M} + 10\text{M} = 12.4\text{M}$

About 10× cheaper.
:::

**Benefits**
- Parallel convolutions capture features at **different scales** at once.
- A richer variety of features in a single output.
- Works as a **standardized module** that can be stacked (e.g. Inception V3).

## ResNet-18

```
7×7 conv, 64, s=2 → BN → 3×3 max pool, s=2
→ stage 1: 2 residual blocks, 64 ch
→ stage 2: 2 residual blocks, 128 ch   ┐
→ stage 3: 2 residual blocks, 256 ch   ├ first block halves H×W, uses 1×1 conv on skip
→ stage 4: 2 residual blocks, 512 ch   ┘
→ global average pool → FC
```

**Basic residual block**

```
x ─ 3×3 conv → BN → ReLU → 3×3 conv → BN ─(+)→ ReLU
└────────────────────────────────────────┘
        (1×1 conv here when down-sampling)
```

**Counting to 18**: 1 ($7\times7$ conv) + 4 stages × 2 blocks × 2 convs = 16, + 1 FC = **18 weighted layers**. The $1\times1$ skip convs are not counted.

**Global average pooling** averages each channel's whole feature map to one number ($7\times7\times512 \to 512$), so the FC layer stays small.

## GoogLeNet

A deep network built by **stacking Inception modules**, with occasional max pooling between them to shrink height and width. It won ImageNet 2014 with far fewer parameters than VGG.

## CNNs for 1-D Data: WaveNet

Convolutions also work on sequences such as audio (a **Temporal Convolutional Network**, TCN). **WaveNet** (DeepMind) generates raw audio waveforms.

**Autoregressive generation**: predict the next audio sample from all previous samples, append it to the input, and repeat.

$$p(x) = \prod_t p(x_t \mid x_1, \dots, x_{t-1})$$

**Key components**

| Component | Purpose |
|---|---|
| **Causal conv** | output at time $t$ only uses inputs at $\le t$, never the future |
| **Dilated conv** | skip inputs with gaps of 1, 2, 4, 8, … so the receptive field grows **exponentially** with depth |
| **Gated activation** | $\tanh(W_f * x) \odot \sigma(W_g * x)$ |
| **Residual + skip connections** | train a deep stack of $k$ layers; skip outputs from every layer are summed |
| **Output head** | sum → ReLU → $1\times1$ → ReLU → $1\times1$ → softmax over possible sample values |

::: tip Filter and gate
- $\tanh \in (-1, 1)$ is the **filter**: what signal to produce.
- $\sigma \in (0, 1)$ is the **gate**: how much of it to let through.

The same idea appears again in LSTM and GRU.
:::

## Summary

| Problem | Fix |
|---|---|
| Deep plain networks degrade | residual blocks (skip connections) |
| Too many parameters / too slow | $1\times1$ conv bottlenecks |
| Which filter size to use? | inception: use several in parallel |
| Sequences / audio | causal + dilated 1-D convolutions (WaveNet) |
