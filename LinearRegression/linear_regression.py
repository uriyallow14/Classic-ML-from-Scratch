import numpy as np
from typing import Dict, List


class OLSLinearRegression:
    def __init__(self, fit_intercept=True):
        """
        """
        self.fit_intercept = fit_intercept
        self.w = None
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.apply_bias_trick(X)
        X_transpose = np.transpose(X)
        inverse_xtx = np.linalg.inv(np.matmul(X_transpose, X))
        self.w = np.matmul(inverse_xtx, np.matmul(X_transpose, y))

    def predict(self, x):
        pred = np.dot(self.w, x)
        return pred

    @staticmethod
    def apply_bias_trick(X):
        ones = np.ones(X.shape[0])
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        X = np.hstack((ones, X))
        return X

