import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples, targets)
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        
        for _ in range(epochs):
            y_hat = X @ w + b
            error = y_hat - y
            gradient_w = (2 / n_samples) * (X.T @ error)
            gradient_b = (2 / n_samples) * np.sum(error)
            w -= lr * gradient_w
            b -= lr * gradient_b
        return np.round(w, 5), np.round(b, 5)
  
        #换维度
        # n_samples, n_features = X.shape
        # if y.ndim == 1:
        #     y = y.reshape(-1, 1)
        # n_targets = y.shape[1]
        # w = np.zeros((n_features, n_targets))
        # b = np.zeros((1, n_targets))
        # for i in range(epochs):
        #     y_hat = X @ w + b
        #     L = np.mean((y_hat - y)**2)
        #     error = y_hat - y
        #     gradient_w = (2 / n_samples) * (X.T @ error)
        #     gradient_b = (2 / n_samples) * np.sum(error, axis = 0, keepdims=True)
        #     w = w - lr * gradient_w
        #     b = b - lr * gradient_b
        # return (np.round(w, 5).flatten(), np.round(b, 5).item())
