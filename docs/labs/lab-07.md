# Lab 7: LSTM and GRU in PyTorch

Theory (gates, cell state, why gating fixes vanishing gradients) is in [Lecture 7](/lectures/lecture-07). This lab uses PyTorch's built-in `nn.LSTM` / `nn.GRU`, trains a GRU to denoise a signal, and sets up the hybrid-author assignment.

## RNN Extensions in LSTM/GRU

`nn.LSTM` and `nn.GRU` take the same arguments, so deep and bi-directional variants are just flags.

```python
class example_LSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(example_LSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional=False,
                                  dropout=0.1)

        self.decoder = torch.nn.Linear(hidden_size, output_size)   # hidden_size * 2 if bidirectional

    def forward(self, input_seq, hidden_state):
        pred, hidden = self.lstm(input_seq, hidden_state)
        pred = self.decoder(pred)
        return pred
```

For GRU, replace `torch.nn.LSTM` with `torch.nn.GRU`; everything else is identical.

| Argument | Meaning |
|---|---|
| `num_layers` | number of LSTM/GRU layers stacked (deep RNN) |
| `batch_first` | input/output shape is (batch size, seq len, features) |
| `bidirectional` | add a second right → left pass; output feature size doubles to `hidden_size * 2`, so the decoder input must be `hidden_size * 2` |
| `dropout` | dropout on the outputs of each layer **except the last**; only has an effect when `num_layers > 1` |

::: warning LSTM hidden state is a tuple
| | `hidden_state` passed in / returned |
|---|---|
| `nn.RNN`, `nn.GRU` | `h` with shape (num layers × num directions, batch, hidden size) |
| `nn.LSTM` | **`(h, c)`**, both with that shape |

Passing `None` initialises everything to zeros for either.
:::

## Mini-batch Gradient in RNNs

Lab 6 trained on one sequence at a time. Stacking several sequences of the same length into one tensor gives a mini-batch:

```
# of features   ■■■■■■■■■■■■■■■■■  Input 1  ┐
  /timestep     ■■■■■■■■■■■■■■■■■  Input 2  │ batch size
                ■■■■■■■■■■■■■■■■■  Input 3  │
                ■■■■■■■■■■■■■■■■■  Input 4  ┘
                └── sequence length ──┘
```

**RNN input format in PyTorch** (with `batch_first=True`) = (batch size, sequence length, # of features). The example above is (4, 17, 1).

## Example: Signal Denoising (many-to-many)

**Task**: input a noisy sine wave, output the clean sine wave, one value per time step (aligned many-to-many).

```
Output:  clean sine        ↑   ↑   ↑   ↑   ↑   ↑   ↑
                          GRU→GRU→GRU→GRU→GRU→GRU→GRU
Input:   noisy sine        ↑   ↑   ↑   ↑   ↑   ↑   ↑
```

### Prepare Data

```python
def sinusoidal_generator(X, signal_freq=60.):
    return np.sin(2 * np.pi * (X) / signal_freq)          # period of 60 steps

def add_noise(Y, noise_range=(-0.35, 0.35)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def sample_seq(sequence_length):
    random_offset = random.randint(0, sequence_length)    # random phase
    X = np.arange(sequence_length)

    denoised_output_seq = sinusoidal_generator(X + random_offset)
    noisy_input_seq = add_noise(denoised_output_seq)

    return noisy_input_seq, denoised_output_seq
```

Generate 12,000 pairs of length 100; first 8,000 for training, last 4,000 for testing:

```python
def create_synthetic_dataset(n_samples, sequence_length):
    noisy_seq_inputs = np.zeros((n_samples, sequence_length))
    denoised_seq_outputs = np.zeros((n_samples, sequence_length))

    for i in range(n_samples):
        noisy_inp, denoised_out = sample_seq(sequence_length)
        noisy_seq_inputs[i, :] = noisy_inp
        denoised_seq_outputs[i, :] = denoised_out

    return noisy_seq_inputs, denoised_seq_outputs

noisy_seq_inputs, denoised_seq_outputs = create_synthetic_dataset(n_samples=12000,
                                                                  sequence_length=100)

train_input_seqs, train_output_seqs = noisy_seq_inputs[:8000], denoised_seq_outputs[:8000]
test_input_seqs,  test_output_seqs  = noisy_seq_inputs[8000:], denoised_seq_outputs[8000:]
```

Reshape to (# samples, seq len, # features) — one feature per step:

```python
train_input_seqs  = train_input_seqs.reshape((train_input_seqs.shape[0], -1, 1))    # (8000, 100, 1)
train_output_seqs = train_output_seqs.reshape((train_output_seqs.shape[0], -1, 1))
test_input_seqs   = test_input_seqs.reshape((test_input_seqs.shape[0], -1, 1))      # (4000, 100, 1)
test_output_seqs  = test_output_seqs.reshape((test_output_seqs.shape[0], -1, 1))
```

### Define Model

```python
class Denoiser_GRU(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Denoiser_GRU, self).__init__()

        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                bidirectional=False)

        self.decoder = torch.nn.Linear(hidden_size, output_size)
        self.output_activation = torch.nn.Tanh()             # squeeze output into [-1, 1]

    def forward(self, input_seq, hidden_state):
        pred, hidden = self.gru(input_seq, hidden_state)     # (batch, 100, hidden)
        pred = self.output_activation(self.decoder(pred))    # (batch, 100, 1)
        return pred
```

Input sequence + hidden state → GRU → output sequence → decoder (linear) → tanh. No embedding layer: the input is already a real number, not a token ID.

### Hyperparameters

```python
denoiser_GRU = Denoiser_GRU(input_size=1, hidden_size=30,
                            num_layers=1, output_size=1)

learning_rate = 0.0003
epochs = 100
batchsize = 300

loss_func = torch.nn.L1Loss()                 # mean absolute error (least absolute deviations)
optimizer = torch.optim.Adam(denoiser_GRU.parameters(), lr=learning_rate)
```

### Train Model

```python
train_loss_list = []

train_input_seqs  = torch.from_numpy(train_input_seqs).float()
train_output_seqs = torch.from_numpy(train_output_seqs).float()
test_input_seqs   = torch.from_numpy(test_input_seqs).float()
test_output_seqs  = torch.from_numpy(test_output_seqs).float()

train_batches_features = torch.split(train_input_seqs, batchsize)    # tuple of (300, 100, 1) tensors
train_batches_targets  = torch.split(train_output_seqs, batchsize)
batch_split_num = len(train_batches_features)                       # ceil(8000 / 300) = 27

for epoch in range(epochs):
    for k in range(batch_split_num):
        hidden_state = None                                          # each batch is independent sequences

        pred = denoiser_GRU(train_batches_features[k], hidden_state)

        optimizer.zero_grad()
        loss = loss_func(pred, train_batches_targets[k])
        train_loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    print("Averaged Training Loss for Epoch ", epoch, ": ",
          np.mean(train_loss_list[-batch_split_num:]))
```

27 batches × 100 epochs = 2,700 iterations.

### Visualize & Evaluate

```python
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, linewidth=3, label='Training Loss')
plt.ylabel("training loss")
plt.xlabel("Iterations")
plt.legend()
sns.despine()
```

```python
with torch.no_grad():
    test_prediction = denoiser_GRU(test_input_seqs, None)
    print("Testing Loss (Least Absolute Deviations): ",
          loss_func(test_prediction, test_output_seqs).item())
# Testing Loss (Least Absolute Deviations):  0.0416
```

Training loss falls from ~0.65 to ~0.04. The denoised output tracks the ground truth closely, except for the **first few steps**: the GRU starts from a zero hidden state and needs a few noisy samples before it "knows" the phase of the wave. A bi-directional GRU would fix this, since every step would also see future samples.

## Assignment: Shakespeare + Conan Doyle Hybrid CharRNN

**Dataset**: full Sherlock Holmes collection + *The Tragedy of Coriolanus*, 6,812,144 characters in total (including spaces).

**Goal**: implement an **LSTM or GRU** that generates multi-domain style text: does it start with "The …" (Doyle) or "Thou …" (Shakespeare)?

1. Decide the **training size** to use from both texts (e.g. first 10k, 100k characters …).
2. Pick **LSTM or GRU** and design your architecture: embedding dim, hidden size, number of RNN layers, training sequence length, etc.
3. After training, **print 5 generated text sequences** and plot the training curve to confirm it trained. How do the generated texts differ from an RNN trained on a single domain (only Shakespeare or only Conan Doyle)?
4. Pick **100 validation sequences each** from Conan Doyle and Shakespeare, not included in the training set. Pass each through the trained RNN **with teacher forcing** to evaluate the log probability of generating the ground-truth text.
5. Compare the two distributions and describe how they differ from those of an RNN trained on a single domain ([Lab 6](/labs/lab-06#assignment-arthur-conan-doyle-ai)). **Does the model learn a hybrid writing style, or does it switch back and forth between two distinct styles?**

::: details Sketch: char-level LSTM with the Lab 6 setup
```python
class CharLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers,
                                  batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.decoder = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, ids, hidden_state):          # ids: (batch, seq_len) LongTensor
        out, hidden_state = self.lstm(self.embedding(ids), hidden_state)
        return self.decoder(out), hidden_state     # logits: (batch, seq_len, vocab)
```

- Build the vocabulary from the **combined** text so both authors' characters are covered.
- `CrossEntropyLoss` expects (N, C): reshape logits to `(-1, vocab_size)` and targets to `(-1,)`.
- When generating, keep passing `hidden_state` back in (for LSTM it is the `(h, c)` tuple).
- Take equal amounts from both authors, otherwise the larger corpus (Sherlock Holmes) dominates.
- For step 5, plot the two log-probability histograms next to the Lab 6 (Doyle-only) ones: a hybrid model should push the two peaks **closer together**.
:::

::: tip Outline mismatch
The lab's outline slide lists "Audio noise filtering with RNNs" as the assignment, but the assignment slides themselves are the hybrid CharRNN above.
:::
