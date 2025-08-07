import numpy as np
from typing import Dict, List


class LinearRegressor(object):
    def __init__(self, fit_intercept=True):
        pass


class OLSLinearRegression:
    def __init__(self, solver='ols'):
        """
        fit_intercept - whether too add an intercept to the model or not.
        solver - two possible values:
            1. 'OLS' - calculating weights w using closed-form solution
            2. 'GD' - calculating weights w iteratively, using gradient-descent method
        """
        self.w = None
        self.solver = solver

    
    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A numpy array of shape (m,n_features_) where m is the number of samples.
        :param y: A numpy array of shape (m,) where m is the number of samples.
        """
        X = self.apply_bias_trick(X)
        if self.solver == 'ols':
            X_transpose = np.transpose(X)
            inverse_xtx = np.linalg.inv(np.matmul(X_transpose, X))
            self.w = np.matmul(inverse_xtx, np.matmul(X_transpose, y))

    def predict(self, x):
        """
        Predict the value of samples based on the current weights.
        :param x: A numpy array of shape (m,n_features_) where m is the number of samples.
        :return:
            y_pred: np.ndarray of shape (m,) where each entry is the predicted
                value of the corresponding sample.
        """
        x = self.apply_bias_trick(x)
        pred = np.matmul(x, self.w)
        return pred

    @staticmethod
    def apply_bias_trick(X):
        """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
        ones = np.ones(X.shape[0]).reshape(-1, 1)
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        X = np.hstack((ones, X))
        return X

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

