import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod


class LinearRegression(ABC):
    """
    Abstarct base class for linear regression models.
    Child classes must implement fit(X, y) and set self.w (1D np.ndarray)
    If fit_intercept=True, self.w[0] is the bias term.
    """
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.w = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = self._add_bias(X)
        if self.w is None:
            raise ValueError("Model is not fitted yet. Call . fit(X, y) first.")
        return X @ self.w
    
    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """ R^2 Score"""
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

class GDLinearRegression(LinearRegression):
    """ Gradient Descent linear regression. stores weights in self.w .
    """
    def __init__(self, lr: float = 0.001, n_iters: int = 1000, fit_intercept: bool = True):
        super().__init__(fit_intercept=fit_intercept)
        self.lr = lr
        self.n_iters = n_iters
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.fit_intercept:
            X = self._add_bias(X)
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=float)

        for it in range(self.n_iters):
            y_pred = X @ self.w
            error =  - y
            grad = (1 / n_samples) * (X.T @ error)
            self.w = self.w - self.lr * grad
            # loss
            mse = np.mean(error ** 2)
            self.loss_history.append(mse)


        
class OLSLinearRegression(LinearRegression):
    """
        Closed-form Ordinary Least Squares using psuedoinverse.
    """
    def __init__(self, fit_intercept: bool = True):
        super().__init__(fit_intercept=fit_intercept)
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.fit_intercept:
            X = self._add_bias(X)
        X_transpose = np.transpose(X)
        inverse_xtx = np.linalg.inv(np.matmul(X_transpose, X))
        self.w = np.matmul(inverse_xtx, np.matmul(X_transpose, y))

def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (y - np.mean(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))
    
    return X, y

