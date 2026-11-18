import numpy as np

def train_linear_regression_numpy(X, y, num_epochs, lr):
    n, d = X.shape
    X_c = np.column_stack([np.ones(shape = (n,)), X])

    W = np.random.normal(size=(d+1,))

    for _ in range(num_epochs):
        y_hat = X_c @ W 
        loss = (1/2) * np.mean((y_hat - y)**2)
        print(loss)
        if loss <= 1e-5:
            return W
        gradient = X_c.transpose() @ (y_hat - y) / len(y)
        W = W - lr * gradient

    return W