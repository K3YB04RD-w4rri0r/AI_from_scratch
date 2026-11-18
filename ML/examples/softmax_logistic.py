import numpy as np


def softmax(logits):
    expd = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return expd / np.sum(expd, axis=1, keepdims=True)


def train_softmaxreg_pytorch(X, y, learning_rate, iterations):
    n, d = X.shape
    C = np.max(y) + 1

    Xc = np.concatenate([np.ones((n, 1)), X], axis=1)
    W = np.zeros((d + 1, C))
    Y = np.eye(C)[y]

    losses = []

    for i in range(iterations):
        logits = Xc @ W
        logits -= np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        if i == 0:
            losses.append(-np.sum(np.log(probs[np.arange(n), y])))

        grad = Xc.T @ (probs - Y)
        W -= learning_rate * grad

        if i < iterations - 1:
            logits = Xc @ W
            logits -= np.max(logits, axis=1, keepdims=True)

            log_probs = logits - np.log(
                np.sum(np.exp(logits), axis=1, keepdims=True)
            )
            losses.append(-np.sum(log_probs[np.arange(n), y]))

    return W.T.tolist(), losses




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
