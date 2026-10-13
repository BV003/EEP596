# Lecture 2: Logistic Regression, Artificial Neuron, MLP

## Logistic Regression = One Neuron

$$z = w^\top x + b, \qquad \hat y = \sigma(z) = \frac{1}{1 + e^{-z}} = P(y=1 \mid x)$$

## Binary Cross-Entropy Loss

$$J = -\frac{1}{m}\sum_{i=1}^{m}\Big(y^{(i)}\log\hat y^{(i)} + (1-y^{(i)})\log(1-\hat y^{(i)})\Big)$$

The loss is convex in $\hat y$.

## Gradient Descent

$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

## Back-propagation (Chain Rule)

$$\frac{\partial J}{\partial w} = \frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial z}\cdot\frac{\partial z}{\partial w} = (\hat y - y)\,x, \qquad \frac{\partial J}{\partial b} = \hat y - y$$

## MLP

**Forward:**

$$h = \sigma(W_1 x + b_1), \quad \hat y = \sigma(W_2 h + b_2), \quad J = \tfrac{1}{2}(\hat y - y)^2$$

**Backward:**

$$\frac{\partial J}{\partial W_2} = (\hat y - y)\,\hat y(1-\hat y)\,h$$

$$\frac{\partial J}{\partial W_1} = (\hat y - y)\,\hat y(1-\hat y)\,W_2\,h(1-h)\,x$$

**Batch gradient descent:** sum the gradients over all $m$ samples, then update.

## Terminology

| Term | Meaning |
|---|---|
| Batch | a subset of the training set |
| Iteration | one batch passed forward and backward |
| Epoch | the whole dataset passed forward and backward once |
