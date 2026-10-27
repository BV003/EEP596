# Lecture 4: Convolutional Neural Networks

## Why Not Just an MLP?

To feed an image into an MLP you **flatten** it into one long vector. A $32\times32\times3$ image becomes 3072 inputs, and every neuron in the next layer connects to all of them.

- **Too many parameters**: a single fully connected layer can already need millions of weights.
- **Loses spatial structure**: after flattening, the MLP no longer knows which pixels are neighbors.

**CNN idea**: use small filters to extract image features first, then a small MLP to classify.

```
Image → [Conv + Pool] × k → Flatten → Fully connected → Class
         feature extraction            classifier
```

## Convolution

Slide a small **filter** (kernel) over the image. At each position, multiply element-wise and sum. The result is a **feature map**.

$$\begin{bmatrix}1&1&1\\0&1&1\\0&0&1\end{bmatrix} * \begin{bmatrix}1&0&1\\0&1&0\\1&0&1\end{bmatrix} = 1+1+1+1 = 4$$

Classical vision used hand-designed filters (edge detectors, blur). A **CNN learns the filter values** by back-prop.

::: tip Few parameters
1000 filters of size $3\times3$ = 9K parameters, compared with millions for a fully connected layer.
:::

## Stride and Padding

- **Stride $s$**: how far the filter moves each step. Larger stride → smaller output.
- **Padding $p$**: add $p$ rings of zeros around the image so the border pixels are used too and the output does not shrink.

| | Valid | Same |
|---|---|---|
| Padding | none | $p = (f-1)/2$ |
| Output size | smaller than input | same as input |

**Output size** (input $n\times n$, filter $f\times f$):

$$\text{out} = \left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1$$

Example: $n=5, f=3, p=0$: stride 1 → $3\times3$; stride 2 → $2\times2$.

## Volume Convolution (Color / Multi-channel)

- The filter always has the **same depth** as the input. A $5\times5\times3$ RGB input uses a $f\times f\times3$ filter, and one filter produces **one** channel.
- Use $n_f$ filters to get $n_f$ output channels.

Input $5\times5\times3$, 5 filters of $2\times2\times3$ → output $4\times4\times5$.

**Example network**

| Layer | $f$ | $s$ | $p$ | filters | Output |
|---|---|---|---|---|---|
| input | | | | | $39\times39\times3$ |
| conv 1 | 3 | 1 | 0 | 10 | $37\times37\times10$ |
| conv 2 | 5 | 2 | 0 | 20 | $17\times17\times20$ |
| conv 3 | 5 | 2 | 0 | 40 | $7\times7\times40$ |
| flatten | | | | | 1960 |

Height and width shrink, channels grow.

## Pooling

Shrink the feature map by summarizing each small window. **No learnable parameters.**

- **Max pooling**: keep the largest value (most common).
- **Average pooling**: keep the mean.

$$\begin{bmatrix}4&9&2&5\\5&6&2&4\\2&4&5&4\\5&6&8&4\end{bmatrix} \xrightarrow{\text{max, } 2\times2,\ s=2} \begin{bmatrix}9&5\\6&8\end{bmatrix}$$

## Typical CNN Layers

| Layer | Role |
|---|---|
| CONV | extract features with learned filters |
| POOL | shrink size, keep the strongest response |
| FC | final classifier |

## Benefits and Challenges

**Benefits**
- **Parameter sharing**: the same filter (e.g. an edge detector) is useful everywhere in the image.
- **Sparse connections**: each output pixel depends only on a small patch of the previous layer.
- **Translation invariance**: a cat is recognized wherever it is in the image.

**Challenges**
- Convolutions are computationally expensive.
- Each layer only sees a local patch, so you need **deep** networks to understand the whole image.

## Classic Architectures

| Network | Year | Key points |
|---|---|---|
| **LeNet-5** | 1998 | first famous CNN, digit recognition. conv(6, 5×5) → avg pool → conv(16, 5×5) → avg pool → conv(120, 5×5) → FC 84 → FC 10 |
| **AlexNet** | 2012 | won ImageNet. ~60M parameters, **ReLU**, trained on multiple GPUs |
| **VGG-16** | 2014 | only $3\times3$ conv (same) and $2\times2$ max pool, stacked deep and regular. 64 → 128 → 256 → 512 → 512 filters, ~138M parameters |

Trend: deeper networks built from smaller, simpler, repeated blocks.
