# Feature Engineering
Feature engineering using Standard Scaler, OneHotEncoder and a SimpleImputer
## 1. Standard Scaler

Normalizes features to have zero mean and unit variance.

### How It Works
- Centers data by subtracting the mean
- Scales data by dividing by standard deviation
- Handles numerical features only

### Usage
```python
from standard_scaler import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # Fit and transform
X_test_scaled = scaler.transform(X_test)  # Apply same scaling

# Key Methods
fit(X): Computes mean and std for later use

transform(X): Applies scaling to new data

fit_transform(X): Combines fit and transform
```
## 2. One Hot Encoder

Converts categorical features to binary columns.

## How It Works
- Creates new binary columns for each category
- Handles unknown categories by ignoring them
- Preserves feature order for consistency

## Usage
```python
from onehot_encoder import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)

# Key Methods
fit(X): Learns unique categories

transform(X): Applies one-hot encoding

fit_transform(X): Combined operation
```

## 3. Simple Imputer

Handles missing values in datasets by filling them with estimated values.

## How It Works
- Identifies missing values (NaN) in input data
- Fills missing values using a specified strategy:
  - Mean: Replace with column average (for numerical data)
  - Median: Replace with column median (robust to outliers)
  - Constant: Fill with a user-defined value

## Usage
```python
from simple_imputer import SimpleImputer

# For numerical data
imputer = SimpleImputer(strategy="mean")  # Also "median" or "constant"
X_imputed = imputer.fit_transform(X_train)

# For constant value filling
imputer = SimpleImputer(strategy="constant", fill_value=-1)

# Key Methods
fit(X): Learns imputation values from the data

transform(X): Applies imputation to new data

fit_transform(X): Combines both steps
```
#### Important Notes
-Automatically handles only numeric columns by default
-For constant strategy, always specify fill_value
-Preserves the original data structure and column order
