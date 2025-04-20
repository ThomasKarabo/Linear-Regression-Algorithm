# Linear Regression Implementation

A from-scratch implementation of linear regression using gradient descent.

## 📝 Mathematical Foundations

### Linear Regression Equation
The model predicts the target variable \( y \) using:
\[
\hat{y} = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
\]
- \( \hat{y} \): Predicted value  
- \( w_0 \): Bias term (intercept)  
- \( w_1, \dots, w_n \): Feature weights  
- \( x_1, \dots, x_n \): Input features  

### Gradient Descent Update Rule
Weights are updated to minimize Mean Squared Error (MSE):
\[
w_j := w_j - \alpha \cdot \frac{\partial}{\partial w_j} \text{MSE}
\]
Where:
- \( \alpha \): Learning rate  
- Partial derivative for weight \( w_j \):
\[
\frac{\partial}{\partial w_j} \text{MSE} = \frac{2}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}
\]

## 🛠️ Usage
```python
from linear_regression import LinearRegression

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
