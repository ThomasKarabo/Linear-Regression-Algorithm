# RegressionMetrics

Evaluates regression model performance.

## 📝 Metrics

### 1. Mean Squared Error (MSE)
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

### 2. R² Score (Coefficient of Determination)
\[
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
\]
Where:
- \(\text{SS}_{\text{res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2\) (Residual Sum of Squares)
- \(\text{SS}_{\text{tot}} = \sum_{i=1}^{n} (y_i - \bar{y})^2\) (Total Sum of Squares)
- \(\bar{y}\) = Mean of observed data

## 🛠️ Usage
```python
from regression_metrics import RegressionMetrics

metrics = RegressionMetrics(y_true, y_pred)
print("MSE:", metrics.mse())       # Lower is better
print("R² Score:", metrics.r2_score())  # Closer to 1 is better
