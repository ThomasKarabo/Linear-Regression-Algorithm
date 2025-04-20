import numpy as np

class SimpleImputer:
    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
    
    def fit(self, X):
        """Compute imputation value (mean/median/mode) for each feature."""
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("Must specify `fill_value` for strategy='constant'.")
            self.statistics_ = np.array([self.fill_value] * X.shape[1])
        else:
            raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'constant'.")
        return self
    
    def transform(self, X):
        """Fill missing values with precomputed statistics."""
        if self.statistics_ is None:
            raise ValueError("Must call `fit` before `transform`.")
        X_imputed = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X_imputed[mask, i] = self.statistics_[i]
        return X_imputed
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
        
    