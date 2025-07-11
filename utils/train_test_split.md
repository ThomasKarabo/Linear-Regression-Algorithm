# Train-Test Split

Splits data into training and testing sets.

## Setup
- First clone the repo by using the command: ```!git clone https://github.com/ThomasKarabo/linear-regression-from-scratch.git```
- Append the directory of the algorithm by running the code</br>
```python
import sys
sys.path.append('/content/linear-regression-from-scratch/utils')
```

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
