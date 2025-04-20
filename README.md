# **Linear Regression from Scratch**

This repository contains a complete implementation of **Linear Regression** built from scratch, along with essential preprocessing and evaluation tools. The project demonstrates how machine learning algorithms work under the hood without relying on libraries like `scikit-learn`.

---

## **📁 Repository Structure**
```
linear-regression-from-scratch/
│
├── 📂 models/
│   ├── linear_regression.py     # Linear Regression implementation
│
├── 📂 preprocessing/
│   ├── onehot_encoder.py       # OneHotEncoder for categorical features
│   ├── standard_scaler.py      # StandardScaler for feature normalization
│   ├── simple_imputer.py       # SimpleImputer for handling missing values
│
├── 📂 metrics/
│   ├── regression_metrics.py   # MSE, R² Score, and other regression metrics
│
├── 📂 utils/
│   ├── train_test_split.py     # Custom train-test split implementation
│
├── 📜 notebook.ipynb           # Jupyter Notebook with full pipeline example
├── 📜 README.md                # This file
```

---

## **📦 Classes Overview**

### **1. `LinearRegression`**
**Location:** `models/linear_regression.py`  
**Description:**  
- Implements linear regression using **gradient descent**.
- Supports training with customizable learning rate and iterations.
- Provides predictions using learned weights and bias.

**Methods:**
- `fit(X, y)` - Trains the model on feature matrix `X` and target `y`.
- `predict(X)` - Returns predictions for new data `X`.
- `feature_importance()` - Returns sorted feature weights.

---

### **2. Preprocessing Classes**

#### **`OneHotEncoder`**
**Location:** `preprocessing/onehot_encoder.py`  
**Description:**  
- Encodes categorical variables into one-hot vectors.
- Handles unknown categories during transformation.

**Methods:**
- `fit(X)` - Learns unique categories.
- `transform(X)` - Converts categorical data to one-hot encoding.
- `fit_transform(X)` - Combines `fit` and `transform`.

#### **`StandardScaler`**
**Location:** `preprocessing/standard_scaler.py`  
**Description:**  
- Standardizes features by removing mean and scaling to unit variance.

**Methods:**
- `fit(X)` - Computes mean and standard deviation.
- `transform(X)` - Scales features.
- `fit_transform(X)` - Combines `fit` and `transform`.

#### **`SimpleImputer`**
**Location:** `preprocessing/simple_imputer.py`  
**Description:**  
- Fills missing values using mean, median, or a constant.

**Methods:**
- `fit(X)` - Computes imputation strategy.
- `transform(X)` - Fills missing values.
- `fit_transform(X)` - Combines `fit` and `transform`.

---

### **3. `RegressionMetrics`**
**Location:** `metrics/regression_metrics.py`  
**Description:**  
- Computes evaluation metrics for regression models.

**Methods:**
- `mse()` - Mean Squared Error.
- `r2_score()` - R² Score (coefficient of determination).
- `mae()` - Mean Absolute Error (optional).

---

### **4. `TrainTestSplit`**
**Location:** `utils/train_test_split.py`  
**Description:**  
- Splits data into training and test sets.
- Supports shuffling and random seed for reproducibility.

**Methods:**
- `split(X, y)` - Returns `(X_train, X_test, y_train, y_test)`.

---

## **📒 Jupyter Notebook (`notebook.ipynb`)**
The notebook demonstrates:
1. **Data Loading & Exploration**  
   - Loads the housing dataset.
   - Examines feature distributions.

2. **Preprocessing Pipeline**  
   - Handles numerical and categorical features.
   - Applies scaling and one-hot encoding.

3. **Model Training & Evaluation**  
   - Trains `LinearRegression` from scratch.
   - Evaluates using `RegressionMetrics`.

4. **Feature Importance Analysis**  
   - Identifies which features most influence predictions.

---

## **🚀 How to Use**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/linear-regression-from-scratch.git
   cd linear-regression-from-scratch
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook notebook.ipynb
   ```

4. **Import modules in your own scripts:**
   ```python
   from models.linear_regression import LinearRegression
   from preprocessing.standard_scaler import StandardScaler
   from metrics.regression_metrics import RegressionMetrics
   ```

---

## **🔍 Key Takeaways**
✅ **Built from scratch** – No reliance on `scikit-learn`.  
✅ **Modular design** – Easy to extend with new features.  
✅ **End-to-end pipeline** – From preprocessing to evaluation.  
✅ **Educational** – Great for understanding ML fundamentals.  

---

## **📈 Future Improvements**
- Add **regularization** (Lasso/Ridge).
- Implement **cross-validation**.
- Support **pandas DataFrames** natively.
- Add **visualizations** for residuals and predictions.

---

## **📜 License**
This project is open-source under the **MIT License**.  

---

**🌟 Enjoyed this project?**  
Give it a ⭐ on [GitHub](https://github.com/yourusername/linear-regression-from-scratch)! Contributions are welcome. 🚀
