# Lecture 1: Introduction

## Deep Learning

- **Machine Learning:** learn an optimal model $H$ from data through training/optimization.
- **Deep Learning** $\subset$ ML: optimize *multiple processing layers* to approximate a function.

**Data splits**

| Set | Role |
|---|---|
| Training | optimization |
| Validation | tuning during training |
| Testing | unbiased final evaluation |

**Classical ML vs. Neural Networks:** classical ML relies on hand-crafted features; neural networks *learn* features, which makes them robust and scalable.

## Artificial Neuron

$$y = f\Big(\sum_{i=1}^{n} x_i w_i + b\Big)$$

Integration → Activation → Output. Training = learning weights $w$ and biases $b$.

::: tip
Without a non-linear activation, a neural network is just a **linear model**.
:::

## Linear Regression

$$\hat y(x) = p_0 + p_1 x + \dots + p_m x^m = \vec p \cdot \vec x$$

**Least squares:**

$$J = \sum_{i=1}^{n} (\vec p \cdot \vec x_i - y_i)^2 \quad\Rightarrow\quad \vec p^{*} = (X^\top X)^{-1} X^\top \vec y$$

**Ridge regression** (adds an L2 penalty):

$$J = (y - Xp)^\top (y - Xp) + \lambda \lVert p \rVert^2 \quad\Rightarrow\quad p = (X^\top X + \lambda I)^{-1} X^\top y$$

| | Least squares | Ridge |
|---|---|---|
| Bias | unbiased | biased |
| Variance | high | low |
