import numpy as np
import pandas as pd

class OneHotEncoder:
    def __init__(self):
        self.categories_ = None
    
    def fit(self, X):
        """Determine unique categories for each categorical feature."""
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include='object').values
        self.categories_ = [np.unique(col) for col in X.T]
        return self
    
    def transform(self, X):
        """Convert categorical features to one-hot encoded format."""
        if self.categories_ is None:
            raise ValueError("Must call `fit` before `transform`.")
        
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include='object').values

        encoded_cols = []
        for i, col in enumerate(X.T):
            categories = self.categories_[i]
            encoded = np.zeros((len(col), len(categories)))
            for j, val in enumerate(col):
                if val in categories:
                    idx = np.where(categories == val)[0][0]
                    encoded[j, idx] = 1
            encoded_cols.append(encoded)
        
        return np.hstack(encoded_cols)
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        """Compute mean and standard deviation for scaling."""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        """Scale features based on precomputed mean and std."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Must call `fit` before `transform`.")
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
