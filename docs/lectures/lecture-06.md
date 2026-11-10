# Lecture 6: Recurrent Neural Networks

## Why Do We Need RNNs?

A feed-forward network (MLP or CNN) maps one fixed-size input to one output. Its neurons have **no memory of past inputs**: each input is processed as if it were the first.

Language does not work like that. When translating

> 안녕하세요, 제 이름은 지민이에요 → Hello, my name is Jimin

each word depends on the words before it, and the next sentence may be longer or shorter.

We need an architecture that handles:

| Requirement | Example |
|---|---|
| **Data order** | "dog bites man" ≠ "man bites dog" |
| **Temporal dependencies** | the meaning of a word depends on earlier words |
| **Variable input sizes** | sentences, audio clips and videos have different lengths |

A **Recurrent Neural Network** adds a loop: the hidden layer feeds its own output back into itself at the next time step.

## RNN Architecture

### The Recurrent Cell

At every time step $t$ the cell combines the new input $x_t$ with the previous hidden state $h_{t-1}$:

$$h_t = \tanh\left(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h\right)$$

$$o_t = W_{ho}\, h_t + b_o$$

- $h_t$ is the network's **memory**: a summary of everything seen so far.
- $h_0$ is usually all zeros.
- The **same weights** $W_{xh}, W_{hh}, W_{ho}$ are used at every time step (weight sharing across time, like a conv filter shared across space).

### Unfolding in Time

Drawing one copy of the cell per time step turns the loop into a long feed-forward chain:

```
output   o0      o1      o2      o3      o4
          ↑       ↑       ↑       ↑       ↑
hidden   h0 ──→  h1 ──→  h2 ──→  h3 ──→  h4
          ↑       ↑       ↑       ↑       ↑
input    x0      x1      x2      x3      x4
       "Hello"  "my"   "name"   "is"  "Jimin"
```

Because the chain can be any length, the same network handles sequences of any size.

::: tip The brain is recurrent too
Biological neurons have their own voltage dynamics over time, and different brain regions send information both forward and backward. Recurrence is a natural way to give a network memory.
:::

### Sequential Data

Anything ordered in time or position: text, speech/audio, music, video frames, sensor and stock time series, DNA.

## Embedding and Decoder

An RNN works on vectors, but words (or characters) are symbols. Two extra layers translate between them:

```
"Hello" → number 4 → [Embedding] → vector → [RNN] → h_t → [Decoder] → softmax → number → "안녕하세요"
```

### Embedding

1. Give every token in the vocabulary an integer ID ("Hello" → 4, "my" → 2).
2. The **embedding matrix** has one row per token: shape (number of embeddings, embedding dim).
3. The embedding of token $i$ is simply **row $i$** of the matrix.

| ID | Embedding (dim = 4) |
|---|---|
| 0 | 8 2 1 9 |
| 1 | 6 5 4 0 |
| 2 ("my") | **7 1 6 2** |
| … | … |

- This is the same as multiplying a one-hot vector by the matrix, but a lookup is much cheaper.
- The embedding matrix is **trainable**, learned by back-propagation with the rest of the network.
- After training, tokens with similar meaning end up with **similar vectors**, so the embedding encodes semantics (e.g. king − man + woman ≈ queen).

### Decoder

A linear layer maps the hidden state to one score per vocabulary token. Softmax turns the scores into a probability distribution, and the predicted token is the most likely one (or one sampled from the distribution).

$$p(\text{token} = k \mid h_t) = \text{softmax}(W_{ho} h_t + b_o)_k$$

## Training RNNs

### Backpropagation Through Time (BPTT)

Unfold the network, run the forward pass left to right, compute the loss at each output, then back-propagate **backward in time** (right to left) through the unfolded chain. Since the weights are shared, the gradient for $W$ is the **sum** of the gradients from every time step.

The gradient reaching an early hidden state passes through every step in between:

$$\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=1}^{T} \text{diag}\big(\tanh'(\cdot)\big)\, W_{hh}$$

### Vanishing and Exploding Gradients

That product multiplies $W_{hh}$ by itself $T$ times.

| | Effect |
|---|---|
| factors $< 1$ | gradient shrinks exponentially → **vanishing**: the network cannot learn long-range dependencies |
| factors $> 1$ | gradient grows exponentially → **exploding**: huge updates, loss becomes `nan` |

**Longer input sequence → higher risk.** It is the same problem as a very deep plain network ([Lecture 5](/lectures/lecture-05)), but the depth is the sequence length.

**Fixes**
- **Gated architectures** such as LSTM and GRU (next lecture).
- **ReLU** as the nonlinearity (derivative is 1 for positive inputs, so it does not shrink the gradient).
- **Shorter sequences** for training.
- **Smaller learning rate**.

### Teacher Forcing

During training, what should the input at step $t+1$ be?

| | Input at step $t+1$ | Effect |
|---|---|---|
| Without teacher forcing | the model's **own prediction** at step $t$ | one early mistake ("flying" instead of "people") pushes all later steps off track; slow to train |
| **With teacher forcing** | the **ground-truth** token at step $t$ | every step gets the correct context; fast, stable training |

```
Without:  <Start> → "Two" → "birds" → "flying" ...   (feeds its own outputs back)
With:     <Start>,  "Two",  "people"  (ground truth) → predicts "Two", "birds", "running"
```

At **inference** there is no ground truth, so the model always feeds its own outputs back (autoregressive generation, as in WaveNet).

## RNN Problem Types

### Configurations

| Type | Input → Output | Example |
|---|---|---|
| **One to one** | single → single | image classification (same as a feed-forward network) |
| **One to many** | single → sequence | image captioning, music generation |
| **Many to one** | sequence → single | sentiment analysis (read a review, output positive/negative) |
| **Many to many** | sequence → sequence | machine translation, video captioning |

::: details Two kinds of many-to-many
- **Aligned**: one output per input step (e.g. labelling each video frame, next-character prediction in Lab 6).
- **Unaligned**: read the whole input first, then produce an output of a different length (e.g. translation). This is the encoder–decoder idea.
:::

### Extensions

**Deep RNN**: stack several recurrent layers; the hidden states of layer $l$ are the inputs of layer $l+1$ (`num_layers` in PyTorch).

| (+) | (−) |
|---|---|
| better performance, used for complex problems | can overfit, longer training |

**Bi-directional RNN**: one RNN reads the sequence left → right, a second reads right → left, and their hidden states are combined at each step.

| (+) | (−) |
|---|---|
| higher performance in NLP; useful when both left and right context matter | harder to train; **not suitable for real-time** processing (needs the whole sequence first) |

## Summary

| Concept | Key point |
|---|---|
| Recurrence | $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$, same weights at every step |
| Embedding | trainable lookup table: token ID → vector |
| Decoder | linear + softmax: hidden state → next-token probabilities |
| BPTT | back-propagate through the unfolded network |
| Vanishing/exploding | repeated multiplication by $W_{hh}$; worse for long sequences |
| Teacher forcing | feed ground truth as input during training |

**Next lecture**: LSTM and GRU, gated RNNs that solve the vanishing-gradient problem.
