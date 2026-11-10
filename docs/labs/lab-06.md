# Lab 6: Character-Level RNN in PyTorch

Theory (recurrence, embedding, BPTT, teacher forcing) is in [Lecture 6](/lectures/lecture-06). This lab trains a character-level RNN that writes text one character at a time.

## PyTorch Building Blocks

### `torch.nn.RNN`

```python
torch.nn.RNN(
    input_size,              # features per time step (= embedding dim)
    hidden_size,             # features in the hidden state h_t
    num_layers=1,            # stacked recurrent layers (deep RNN)
    nonlinearity='tanh',     # 'tanh' or 'relu'
    batch_first=False,
)
```

```python
output, h_n = rnn(x, h_0)    # h_0 = None → zeros
```

| Tensor | Shape (`batch_first=False`, default) |
|---|---|
| `x` | (sequence length, batch size, input size) |
| `output` | (sequence length, batch size, hidden size): top layer's $h_t$ at **every** step |
| `h_0`, `h_n` | (num layers, batch size, hidden size): **last** step's $h$ for every layer |

With `batch_first=True` the input and output become (batch size, sequence length, …), but `h_0`/`h_n` keep the same shape.

### `torch.nn.Embedding`

```python
emb = torch.nn.Embedding(num_embeddings, embedding_dim)   # vocab size × vector size
emb(torch.tensor([4, 2]))     # integer IDs → rows 4 and 2, shape (2, embedding_dim)
```

The input is **integer IDs** (`LongTensor`), not one-hot vectors. Output shape is input shape + `(embedding_dim,)`.

### Decoder

```python
decoder = torch.nn.Linear(hidden_size, output_size)        # output_size = vocab size
```

Produces raw logits; `CrossEntropyLoss` applies softmax internally.

## Example: Shakespeare Text Generation

**Dataset**: full script of *The Tragedy of Coriolanus* as a `.txt` file, 3,801,089 characters (including spaces), 66 unique characters.

**Task**: given the characters so far, predict the next one (many-to-many, aligned).

### Prepare Data

```python
data_size_to_train = 10000                         # only use the first 10k characters

data = open('shakespeare.txt', 'r').read()[:data_size_to_train]
characters = sorted(list(set(data)))               # unique characters, sorted
data_size, vocab_size = len(data), len(characters)

print("Data has {} characters, {} unique".format(data_size, vocab_size))
# Data has 10000 characters, 57 unique
```

Build the lookup tables in both directions:

```python
character_to_num = {ch: i for i, ch in enumerate(characters)}   # {'\n': 0, ' ': 1, '!': 2, ...}
num_to_character = {i: ch for i, ch in enumerate(characters)}
```

Encode the text as a list of integers:

```python
data = list(data)
for i, ch in enumerate(data):
    data[i] = character_to_num[ch]

print(data[:10])        # e.g. [17, 48, 57, ...]
```

### Define Model

```python
import torch

class CharRNN(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_size, hidden_size, num_layers, output_size):
        super(CharRNN, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers,
                                nonlinearity='relu')     # ReLU: less vanishing gradient
        self.decoder = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)                     # (L, 1) → (L, 1, embedding_dim)
        output, hidden_state = self.rnn(embedding, hidden_state)  # (L, 1, hidden_size)
        output = self.decoder(output)                             # (L, 1, vocab_size) logits
        return output, hidden_state.detach()
```

`input_size` must equal `embedding_dim`, because the embedding is the RNN's input.

::: tip Why `hidden_state.detach()`?
The hidden state is carried from one training chunk to the next so the model keeps context across chunks. `detach()` cuts the gradient link to the previous chunk, so back-propagation only runs through the current `training_sequence_len` steps (**truncated BPTT**). Without it, the graph would grow through the whole text and memory would run out (or PyTorch would complain about backward through a freed graph).
:::

### Hyperparameters

```python
import numpy as np

torch.manual_seed(25)

rnn = CharRNN(num_embeddings=vocab_size, embedding_dim=100,
              input_size=100, hidden_size=512, num_layers=3,
              output_size=vocab_size)

lr = 0.001
epochs = 50
training_sequence_len = 50       # characters per training chunk (BPTT length)
validation_sequence_len = 200    # characters to generate after each epoch

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

train_loss_list = []             # track training loss for plotting
```

### Train (Teacher Forcing)

The target is the input **shifted by one character**:

```
input   G  r  e  a  t
target  r  e  a  t  !
```

During training the ground-truth character is always fed in (teacher forcing), regardless of what the model predicted.

```python
data = torch.unsqueeze(torch.tensor(data), dim=1)    # (data_size, 1): sequence length × batch 1

for epoch in range(epochs):

    character_loc = np.random.randint(100)           # random start in the first 100 chars
    iteration = 0
    hidden_state = None                              # h_0 = zeros

    while character_loc + training_sequence_len + 1 < data_size:
        input_seq  = data[character_loc     : character_loc + training_sequence_len]
        target_seq = data[character_loc + 1 : character_loc + training_sequence_len + 1]

        output, hidden_state = rnn(input_seq, hidden_state)

        # (L, 1, vocab) → (L, vocab) and (L, 1) → (L,)
        loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()                              # backprop through time
        optimizer.step()

        character_loc += training_sequence_len       # move to the next chunk
        iteration += 1

    print("Averaged Training Loss for Epoch ", epoch, ": ", np.mean(train_loss_list[-iteration:]))
```

The random start offset means the chunk boundaries differ each epoch, so the model does not always see the same 50-character windows.

### Generate Text (Validation)

At generation time there is no ground truth: start from one random character, then **feed each output back in as the next input**. Still inside the epoch loop:

```python
    # Sample and generate a text sequence after every epoch
    character_loc = 0
    hidden_state = None

    rand_index = np.random.randint(data_size - 1)
    input_seq = data[rand_index : rand_index + 1]            # one character, shape (1, 1)

    print("----------------------------------------")
    with torch.no_grad():
        while character_loc < validation_sequence_len:
            output, hidden_state = rnn(input_seq, hidden_state)

            output = torch.nn.functional.softmax(torch.squeeze(output), dim=0)   # probabilities
            character_distribution = torch.distributions.Categorical(output)
            character_num = character_distribution.sample()  # sample, don't take argmax

            print(num_to_character[character_num.item()], end='')

            input_seq[0][0] = character_num.item()           # output becomes next input
            character_loc += 1
    print("\n----------------------------------------")
```

::: warning `input_seq` is a view of `data`
`data[rand_index : rand_index + 1]` shares memory with `data`, so `input_seq[0][0] = ...` **overwrites the training text**. It is a 1-character change per epoch here, but use `input_seq = data[rand_index : rand_index + 1].clone()` to be safe.
:::

::: tip Sample vs. argmax
Always picking the most likely character tends to fall into loops ("the the the …"). Sampling from the softmax distribution gives more varied, natural text.
:::

**Results**

| Epoch | Avg. loss | Generated text |
|---|---|---|
| 1 | 2.75 | `MENUNUNENNNNNENNENEUNNENNNNUE… Suwirs touy to Ther'krn;` |
| 50 | 0.26 | `the Capitol; who's fide our trumpetersvile in awe, which else / Would feed on one another? What's their abundance;` |

After one epoch it has learned spaces, line breaks and capitalized speaker names; after 50 it produces Shakespeare-like words and punctuation. (With only 10k training characters, a loss of 0.26 also means it is memorizing a lot.)

### Plot Training Loss

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', font_scale=2.5)

plt.figure(figsize=(15, 9))
plt.plot(train_loss_list, linewidth=3, label='Training Loss')
plt.plot(np.convolve(train_loss_list, np.ones(100), 'valid') / 100,     # 100-step moving average
         linewidth=3, label='Rolling Averaged Training Loss')
plt.ylabel("training loss")
plt.xlabel("Iterations")
plt.legend()
sns.despine()
```

The per-chunk loss is very noisy (some chunks are harder than others), so the rolling average shows the trend: about 3 → 0.3 over ~10,000 iterations.

## Assignment: Arthur Conan Doyle AI

**Dataset**: the full Sherlock Holmes collection, 3,011,055 characters (including spaces), 102 unique characters.

### Part 1: Train a Sherlock Holmes text generator

1. Choose how much text to train on (e.g. first 10k, 100k characters …).
2. Design your own RNN: embedding dim, hidden size, number of layers, activation, training sequence length, etc.
3. Print the generated text sequence that you think **most closely resembles Sherlock Holmes**, and plot the training curve.
4. Which hyperparameter affects the quality the most?

### Part 2: Use the model as a Conan Doyle classifier

1. Pick **100 text sequences from Conan Doyle** and **100 from Shakespeare**, none of them in the training set.
2. Pass each sequence through the trained RNN **with teacher forcing** and compute the **log probability** of the model producing the ground-truth text.
3. Compare the two distributions and explain how the model works as a Conan Doyle style classifier.

The log probability is the negative cross-entropy between the model outputs and the ground truth:

$$\log p(x_1, \dots, x_L) = \sum_{t=1}^{L-1} \log p(x_{t+1} \mid x_1, \dots, x_t) = -(L-1)\cdot \text{CE}_\text{mean}$$

A model trained on Conan Doyle should assign **higher** log probability (less surprise) to unseen Conan Doyle text than to Shakespeare.

::: details Sketch: scoring one sequence
```python
def log_prob(rnn, text):
    ids = torch.tensor([character_to_num[ch] for ch in text]).unsqueeze(1)   # (L, 1)
    with torch.no_grad():
        output, _ = rnn(ids[:-1], None)                 # teacher forcing: feed ground truth
        ce = loss_fn(output.squeeze(1), ids[1:].squeeze(1))   # mean CE over L-1 predictions
    return -ce.item()                                   # mean log prob per character
```

- Use sequences of the **same length** for both authors so the scores are comparable (or use the per-character mean as above).
- Shakespeare may contain characters that are not in the Sherlock vocabulary; skip or drop sequences with unknown characters.
- Call `rnn.eval()` first, and start each sequence from `hidden_state = None`.
- Plot both sets of scores as overlapping histograms; a threshold between the two peaks is a simple classifier.
:::

### Tips

**Decide first**: training data size, embedding dim (= RNN input size), hidden size, activation (ReLU / tanh), decoder output size (= vocab size), learning rate, optimizer, number of epochs, training sequence length.

- Getting `nan` during training means it is unstable (exploding gradients): **decrease the learning rate or the training sequence length**. Gradient clipping also helps:
  ```python
  loss.backward()
  torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
  optimizer.step()
  ```
- With ReLU you may be able to use longer sequences.
- Choose the training data size according to your machine.
- More `num_layers` may perform better but trains longer.
- The example runs on CPU with batch size 1. To use a GPU, move the model and `data` to the device (`rnn.cuda()`, `data.cuda()`).
