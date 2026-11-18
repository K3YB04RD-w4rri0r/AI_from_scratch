import numpy as np
import torch

n, d = 20, 10
device = "cpu"

X_train_torch, y_train_torch = torch.randn(size = (n, d)), torch.ones(size = (n,))
X_train_np, y_train_np = np.random.normal(size= (n,d)), np.ones(shape=(n,))
y_train_cat_np = np.random.random_integers(low = 0, high = 2, size=(n,))

def sigmoid(logits):
    return (1/1 + np.exp(-logits))

def gini(y):
    c = np.bincount(y)
    pi = c / len(y)
    return 1 - np.sum(pi**2)


if __name__=="__main__":
    print(X_train_torch.shape, y_train_torch.shape)
    print(X_train_np.shape, y_train_np.shape)
