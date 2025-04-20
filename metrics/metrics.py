import numpy as np

class RegressionMetrics:
    def __init__(self, y_true, y_pred):
        """
        Initialize with true and predicted values.
        
        Args:
            y_true (array): Ground truth target values.
            y_pred (array): Predicted target values.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Check shapes match
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must match.")
    
    def mse(self):
        """Calculate Mean Squared Error (MSE)."""
        return np.mean((self.y_true - self.y_pred) ** 2)
    
    def r2_score(self):
        """Calculate R-squared (R²) score."""
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)  # Total sum of squares
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0  # Avoid division by zero
    
    def get_metrics(self):
        """Return all metrics as a dictionary."""
        return {
            "MSE": self.mse(),
            "R2_Score": self.r2_score()
        }