import numpy as np
from data import sigmoid

def train_logistic_regression_numpy(X, y, num_epochs, lr):
    n, d = X.shape
    X_c = np.column_stack([np.ones(shape = (n,)), X])

    W = np.random.normal(size=(d+1,))

    for _ in range(num_epochs):
        zl = X_c @ W
        y_hat = sigmoid(zl)
        loss = (1/2) * np.mean((y_hat - y)**2)
        print(loss)
        if loss <= 1e-5:
            return W
        # gradient = X_c.transpose() @ (y_hat - y) / len(y) linear 
        # d(s(zl)/dW)T @ (dmse/dy), (ds(zl)/d(zl) @ d(X_c @ W)/dW)
        # X_c.T @ diag(s'((zl))).T @ (y_hat -y)/ len(y)
        # X_c.T@ (y_hat - y) * (s(zl) * (1-s(zl)) / len(y)
        gradient = X_c.transpose() @ (y_hat - y) * ((y_hat)*(1-y_hat)) / len(y)
        W = W - lr * gradient

    return W
