Classic ML

- Linear and logistic regression with gradient descent (loss, gradient, update loop)
- k-means
- k-nearest neighbours
- PCA via SVD or eigendecomposition
- A decision tree split (Gini or entropy), and the idea of bagging and boosting on top
- Metrics: precision, recall, F1, ROC-AUC, IoU, NMS and mAP for detection

Deep learning core

- A two-layer MLP with manual backprop
- Softmax plus cross-entropy, numerically stable
- SGD with momentum and Adam
- Batch norm and layer norm, forward pass, train versus eval behaviour
- Dropout
- A 2D convolution and max-pool forward pass
- A full PyTorch training loop: dataloader, forward, loss, backward, step, eval

Transformers

- Scaled dot-product attention and multi-head self-attention with masking
- Positional encoding (sinusoidal)
- One transformer block, pre-norm with residuals
- Greedy, top-k and nucleus sampling, and beam search

Generative models

- VAE loss with the reparameterisation trick
- DDPM training step (add noise, predict noise) and the sampling loop
- Contrastive loss (InfoNCE)