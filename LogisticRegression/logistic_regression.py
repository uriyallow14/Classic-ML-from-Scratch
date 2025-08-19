import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinearRegression:
    def __init__(self, lr: float = 0.01, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear_pred = self.weights @ X + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * (X.T @ (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db



    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_pred = self.weights @ X + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y_i <= 0.5 else 1 for y_i in y_pred]

        return class_pred
