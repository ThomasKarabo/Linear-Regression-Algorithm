# Regression Metrics

Evaluates model performance.

## Setup
- First clone the repo by using the command below </br>
```!git clone https://github.com/ThomasKarabo/linear-regression-from-scratch.git```
- Append the directory of the algorithm by running the code</br>
```python
import sys
sys.path.append('/content/linear-regression-from-scratch/metrics')
```

## Available Metrics
1. **MSE (Mean Squared Error)**: 
   - Average squared difference between predictions and true values
   - Lower values indicate better fit

2. **R² Score**:
   - Measures how much variance the model explains
   - Ranges from -∞ to 1 (higher is better)

## Usage
```python
from regression_metrics import RegressionMetrics

metrics = RegressionMetrics(y_true, y_pred)
print("MSE:", metrics.mse())
print("R² Score:", metrics.r2_score())
```
