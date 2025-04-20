# Train-Test Split

Splits data into training and testing sets.

## How It Works
- Randomly divides data while maintaining proportions
- Optional shuffling for randomness
- Fixed random seed for reproducibility

## Usage
```python
from train_test_split import TrainTestSplit

splitter = TrainTestSplit(test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = splitter.split(X, y)
```
## Parameters
- test_size: Fraction for testing (default: 0.2)
- random_state: Seed for reproducibility
- shuffle: Whether to shuffle data (default: True)
