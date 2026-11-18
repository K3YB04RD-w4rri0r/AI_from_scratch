import numpy as np
import torch
import torch.nn as nn
from data import X_train_torch, y_train_torch, X_train_np, y_train_np, y_train_cat_np
from data import sigmoid, gini

def softmax(logits):
    expd = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return expd / np.sum(expd, axis=1, keepdims=True)

def softmax_logreg_numpy(X_train_np, y_train_np, lr, num_epochs, num_classes): 
    n, d = X_train_np.shape
    X = np.column_stack([np.ones(shape = (n, 1)), X_train_np])
    W = np.random.normal(size = (d+1, num_classes - 1))

    for _ in range(num_epochs):
        probs = softmax(X @ W)
        # crossentrop loss
        # print(y_hat)
        pk = probs[np.arange(len(y_train_np)), y_train_np]
        loss = - np.mean(np.log(pk))
        # d(s(zl)/dw)T @ dCE = (d(s(zl)/dzl) @ d(zl/dw)))T @ dCE 
        # = X.T @ ((d(s(zl)/dzl)  @ dCE)
        gradient = X.transpose() @ (pk - y_train_cat_np) / (len(y_train_np))
        W = W - lr * gradient
    return W

# W = np.round(train_linear_regression_numpy(X_train_np, y_train_np, 100000, 0.001), decimals=2)
# print(W.tolist())
softmax_logreg_numpy(X_train_np, y_train_cat_np, 0.001, 2, 4)
print("hello_world")