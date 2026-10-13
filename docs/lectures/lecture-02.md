# Lecture 2: Logistic Regression, Artificial Neuron, MLP

## Logistic Regression = One Neuron

$$z = w^\top x + b, \qquad \hat y = \sigma(z) = \frac{1}{1 + e^{-z}} = P(y=1 \mid x)$$

## Binary Cross-Entropy Loss

$$J = -\frac{1}{m}\sum_{i=1}^{m}\Big(y^{(i)}\log\hat y^{(i)} + (1-y^{(i)})\log(1-\hat y^{(i)})\Big)$$

The loss is convex in $\hat y$.

## Gradient Descent

$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

## Back-propagation

::: tip Purpose
Compute the gradient of the loss with respect to **every parameter** ($\partial J/\partial w$, $\partial J/\partial b$). Gradient descent needs these to know which direction to move each parameter and by how much.
:::

**Order: Forward → Backward → Update**

**① Forward**: input → loss. Compute the prediction and the loss, and cache the intermediate values ($z$, $\hat y$).

$$x \;\to\; z = w^\top x + b \;\to\; \hat y = \sigma(z) \;\to\; J = L(\hat y, y)$$

**② Backward**: loss → input. Apply the chain rule in reverse, reusing the cached values.

$$\frac{\partial J}{\partial \hat y} \;\to\; \frac{\partial J}{\partial z} = \hat y - y \;\to\; \frac{\partial J}{\partial w} = (\hat y - y)\,x, \quad \frac{\partial J}{\partial b} = \hat y - y$$

**③ Update**: apply gradient descent with these gradients.

::: info Why forward first?
Every factor in the backward pass ($\hat y$, $\sigma'(z)$, $x$) is a value computed during the forward pass.
:::

## MLP

**① Forward**: $x \to h \to \hat y \to J$

$$h = \sigma(W_1 x + b_1), \quad \hat y = \sigma(W_2 h + b_2), \quad J = \tfrac{1}{2}(\hat y - y)^2$$

**② Backward**: $J \to$ output layer $\to$ hidden layer

1. **Output layer** error, then $W_2$:

   $$\delta_2 = (\hat y - y)\,\hat y(1-\hat y), \qquad \frac{\partial J}{\partial W_2} = \delta_2\, h$$

2. **Hidden layer**: pass the error back through $W_2$, then compute $W_1$:

   $$\delta_1 = \delta_2\, W_2\, h(1-h), \qquad \frac{\partial J}{\partial W_1} = \delta_1\, x$$

::: info Why output layer first?
The hidden-layer gradient reuses $\delta_2$. Each layer's error is computed once and passed back to the layer before it, so no chain-rule product is recomputed.
:::

**③ Update**: batch gradient descent. Sum the gradients over all $m$ samples, then update $W_1, W_2, b_1, b_2$.

## Terminology

| Term | Meaning |
|---|---|
| Batch | a subset of the training set |
| Iteration | one batch passed forward and backward |
| Epoch | the whole dataset passed forward and backward once |
