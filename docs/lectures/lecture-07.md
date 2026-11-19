# Lecture 7: Advanced Recurrent Neural Networks

[Lecture 6](/lectures/lecture-06) introduced the vanilla RNN and ended with its main weakness: gradients vanish or explode over long sequences. This lecture introduces **gated RNNs** (LSTM and GRU), which are built to carry information across many time steps.

## Why We Need Gated RNNs

### Recap: Backpropagation Through Time

The forward pass runs left to right through the unfolded network. The backward pass runs **backward in time**: the gradient of a loss at output $o_3$ reaches $h_0$ only by passing through every step in between.

```
output   o0      o1      o2      o3      o4
          ↑       ↑       ↑      ↑↓       ↑
hidden   h0 ⇄    h1 ⇄    h2 ⇄    h3 ──→  h4
        ∂h1/∂h0  ∂h2/∂h1  ∂h3/∂h2  ∂o3/∂h3
          ↑↓      ↑↓      ↑↓      ↑↓
input    x0      x1      x2      x3      x4
```

$$\frac{\partial o_3}{\partial h_0} = \frac{\partial o_3}{\partial h_3}\cdot\frac{\partial h_3}{\partial h_2}\cdot\frac{\partial h_2}{\partial h_1}\cdot\frac{\partial h_1}{\partial h_0}$$

### Vanishing and Exploding Gradients

Simplify the vanilla RNN to a scalar: $h_t = \sigma(w\, h_{t-1})$. Then

$$\frac{\partial h_{t'}}{\partial h_t} = \prod_{k=1}^{t'-t} w\,\sigma'(w\,h_{t'-k}) = \underbrace{w^{\,t'-t}}_{!!!}\ \prod_{k=1}^{t'-t} \sigma'(w\,h_{t'-k})$$

The weight is raised to the power of the distance in time. If $w \neq 1$ the gradient **decays or grows exponentially**.

**Longer input sequence → higher risk of vanishing/exploding gradients.** The gradient for early inputs ($\nabla w$ from $x_0, x_1$) becomes tiny, so the network cannot learn long-range dependencies.

We need an RNN architecture capable of processing **longer sequences**.

## Gated RNNs

| | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| State passed between steps | $h_t$ | $h_t$ **and** cell state $C_t$ | $h_t$ |
| Gates | none | forget, input, output (3) | reset, update (2) |
| Weight matrices (per layer) | 1 set | 4 sets | 3 sets |
| Year | — | 1997 | 2014 |

Diagram legend used in the slides: yellow box = neural network layer (linear + activation), pink circle = pointwise operation, arrows = vector transfer, merging lines = concatenate, splitting lines = copy.

### Vanilla RNN

$$
\begin{aligned}
a^{(t)} &= b + W h^{(t-1)} + U x^{(t)} \\
h^{(t)} &= \tanh\big(a^{(t)}\big) \\
o^{(t)} &= c + V h^{(t)} \\
\hat y^{(t)} &= \text{softmax}\big(o^{(t)}\big)
\end{aligned}
$$

A single tanh layer on the concatenation of $h_{t-1}$ and $x_t$. Every step overwrites the whole hidden state.

## LSTM (Long Short-Term Memory)

Hochreiter & Schmidhuber, *Long short-term memory*, Neural Computation 9.8 (1997).

$$
\begin{aligned}
f_t &= \sigma_g(W_f x_t + U_f h_{t-1} + b_f) && \text{forget gate} \\
i_t &= \sigma_g(W_i x_t + U_i h_{t-1} + b_i) && \text{input gate} \\
o_t &= \sigma_g(W_o x_t + U_o h_{t-1} + b_o) && \text{output gate} \\
\tilde C_t &= \sigma_c(W_c x_t + U_c h_{t-1} + b_c) && \text{candidate} \\
c_t &= f_t \circ c_{t-1} + i_t \circ \tilde C_t && \text{cell state update} \\
h_t &= o_t \circ \sigma_h(c_t) && \text{new hidden state}
\end{aligned}
$$

- $\sigma_g$ = sigmoid (output in $[0, 1]$, acts as a soft on/off switch); $\sigma_c, \sigma_h$ = tanh.
- $\circ$ = element-wise (Hadamard) product.

### Step by Step

**1. Cell state $C_t$**: the horizontal line across the top of the cell.
- Unique to LSTM.
- The **long-term memory** of the model. It is changed only by one multiplication ($\times f_t$) and one addition ($+\, i_t \circ \tilde C_t$), so information can flow along it almost unchanged.

**2. Forget gate $f_t$**: what information to **forget** from $C_{t-1}$.
$f_t \approx 0$ erases that component; $f_t \approx 1$ keeps it.

**3. Input gate $i_t$ and candidate $\tilde C_t$**: what new information to **keep**.
$\tilde C_t$ (tanh, in $[-1, 1]$) proposes new content; $i_t$ decides how much of it is written.

**4. Update the cell state**: forget + add new info, modulated by $i_t$:
$c_t = f_t \circ c_{t-1} + i_t \circ \tilde C_t$.

**5. Output gate $o_t$**: form the new hidden state from the cell state, modulated by $o_t$:
$h_t = o_t \circ \tanh(c_t)$.

| Gate | Role |
|---|---|
| Forget $f_t$ | decides what is relevant to keep from previous steps |
| Input $i_t$ | decides what information is relevant to add from the current step |
| Output $o_t$ | combined with the cell state to determine the next hidden state |

::: tip Key idea
LSTM divides the original hidden state into
1. **long-term memory** (cell state $C_t$) and
2. **short-term context** (LSTM hidden state $h_t$).
:::

### Why LSTM Fixes the Gradient Problem

Along the cell state, the gradient between two time steps is (ignoring the paths through the gates)

$$\frac{\partial c_{t'}}{\partial c_t} = \prod_{k=1}^{t'-t} \sigma(v_{t+k}) = \prod_{k=1}^{t'-t} f_{t+k}$$

| | Gradient across $t'-t$ steps | |
|---|---|---|
| Vanilla RNN | $w^{\,t'-t} \prod \sigma'(\cdot)$ | the **same** weight multiplied again and again → exponential decay/growth if $w \neq 1$ |
| LSTM | $\prod f_{t+k}$ | **no exponential weight term**; each factor is a forget gate value the network learns to set near 1 when it needs to remember |

The cell state acts like the skip connection in ResNet ([Lecture 5](/lectures/lecture-05)): an additive path that gradients can travel along without being repeatedly squashed.

## GRU (Gated Recurrent Unit)

Chung et al., *Empirical evaluation of gated recurrent neural networks on sequence modeling*, arXiv:1412.3555 (2014).

A simplified LSTM: no separate cell state, two gates instead of three.

$$
\begin{aligned}
z_t &= \sigma\big(W_z \cdot [h_{t-1}, x_t]\big) && \in [0, 1] && \text{update gate} \\
r_t &= \sigma\big(W_r \cdot [h_{t-1}, x_t]\big) && \in [0, 1] && \text{reset gate} \\
\tilde h_t &= \tanh\big(W \cdot [r_t * h_{t-1},\ x_t]\big) && \in [-1, 1] && \text{candidate} \\
h_t &= (1 - z_t) * h_{t-1} + z_t * \tilde h_t
\end{aligned}
$$

$[h_{t-1}, x_t]$ is concatenation; $*$ is element-wise multiplication.

### Information Flow in GRU

**Reset gate $r_t$**: how much of the past information $h_{t-1}$ should be retained, with respect to the new input $x_t$, to form the new candidate $\tilde h_t$. $r_t \approx 0$ lets the candidate ignore the past ("start fresh").

**Update gate $z_t$**:
- $(1 - z_t)$: how much of the past information $h_{t-1}$ should be retained;
- $z_t$: how much of the new information $\tilde h_t$ makes it into the final $h_t$.

| $z_t$ | Mode |
|---|---|
| $z = 1$ | pure **overwrite** mode: $h_t = \tilde h_t$ |
| $z = 0$ | pure **memory** mode: $h_t = h_{t-1}$ |

GRU keeps the original RNN input–output structure $(x_t, h_t)$ by letting the single gate $z_t$ handle both data **retention** and **attrition** (LSTM uses separate $f_t$ and $i_t$ for this).

### Gradient in GRU

$$\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=k+1}^{T} \Big[\text{diag}(1 - z_t) + \text{nonlinear terms}\Big]$$

No weight matrix $W$ multiplies the $(1 - z_t)$ term, unlike the vanilla RNN. When $z_t \approx 0$ the factor is close to the identity, so the gradient passes through unchanged.

### LSTM vs GRU

| | LSTM | GRU |
|---|---|---|
| States | $h_t$, $C_t$ | $h_t$ |
| Gates | 3 | 2 |
| Parameters | more (4 weight sets) | ~25% fewer (3 weight sets) |
| Speed | slower | faster to train |
| Performance | similar in practice; LSTM sometimes better on very long / complex sequences | similar; often better on smaller datasets |

## RNN Configurations and Extensions

These apply equally to vanilla RNN, LSTM and GRU (see [Lecture 6](/lectures/lecture-06#rnn-problem-types) for details).

| Configuration | Example |
|---|---|
| One to one | image classification |
| One to many | image captioning |
| Many to one | sentiment analysis |
| Many to many (unaligned) | machine translation |
| Many to many (aligned) | per-frame labelling, signal denoising ([Lab 7](/labs/lab-07)) |

| Extension | Idea |
|---|---|
| **Deep RNN** | stack recurrent layers (`num_layers`) |
| **Bi-directional RNN** | one pass left → right, one right → left; concatenate the hidden states (`bidirectional=True`) |

## Next: Encoder–Decoder

```
ENCODER                                   DECODER
[ ]→[ ]→[ ]→[ ]→[ ] → ENCODED STATE → [ ]→[ ]→[ ]→[ ]
 ↑   ↑   ↑   ↑   ↑                      ↑   ↑   ↑   ↑
 encoder input sequence               START  (previous output fed back)
```

The encoder compresses the whole input sequence into one state; the decoder starts from that state and a START token and generates the output sequence one step at a time, feeding each output back as the next input. This handles unaligned many-to-many problems such as translation.

## Summary

| Concept | Key point |
|---|---|
| Vanishing/exploding | vanilla RNN gradient contains $w^{t'-t}$ |
| LSTM | cell state (long-term) + hidden state (short-term); forget / input / output gates |
| LSTM cell update | $c_t = f_t \circ c_{t-1} + i_t \circ \tilde C_t$, additive path, no repeated weight |
| GRU | reset + update gates; $h_t = (1-z_t) h_{t-1} + z_t \tilde h_t$ |
| GRU update gate | $z=0$ memory mode, $z=1$ overwrite mode |
| Extensions | deep (stacked) and bi-directional work for LSTM/GRU too |
