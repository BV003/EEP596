# Lab 1: Python Foundation & Regression

## NumPy Essentials

| Purpose | Functions |
|---|---|
| Combine | `np.concatenate` `np.stack` `np.hstack` `np.vstack` |
| Split | `np.split(arr, [i1, i2], axis=0)` (cumulative indices) |
| Statistics | `np.min` `np.max` `np.mean` `np.sum` (`axis=`) |
| Index lookup | `np.argmin` `np.argmax` `np.argsort` `np.where` |

## Ridge Regression with scikit-learn

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = X[:80], X[80:], y[:80], y[80:]  # 80/20 split

model = Ridge(alpha=1)            # alpha = λ
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

## Assignment

Dataset: CMS calorimeter data (`hgcal.csv`). Predict deposited energy from cell positions.

1. `scale_data()` — standard scaling: $z = (x - \mu)/\sigma$
2. `split_data()` — split an array by ratios along an axis
3. `ridge_regression()` — fit on 80%, predict on 20%, report RMSE on the original scale
