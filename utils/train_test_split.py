import numpy as np

class TrainTestSplit:
    def __init__(self, test_size=0.2, random_state=None, shuffle=True):
        """
        Initialize the splitter.
        
        Args:
            test_size (float): Proportion of data for testing (default: 0.2).
            random_state (int): Seed for reproducibility (default: None).
            shuffle (bool): Whether to shuffle data before splitting (default: True).
        """
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
    
    def split(self, X, y=None):
        """
        Split data into training and testing sets.
        
        Args:
            X (array): Feature matrix.
            y (array): Target vector (optional).
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) if y is provided, else (X_train, X_test).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        
        if y is not None:
            y_train, y_test = y[train_indices], y[test_indices]
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test