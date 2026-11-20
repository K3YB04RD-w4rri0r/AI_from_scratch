

#### Breaking the computational graph

| Operation | Breaks graph? | Explanation |
|---|---|---|
| `torch.tensor(existing_tensor)` | Yes | Creates a new independent tensor |
| `.detach()` | Yes | Explicitly detaches from autograd |
| `.item()` | Yes | Converts tensor to a Python scalar |
| `.numpy()` | Yes | Converts to NumPy |
| `with torch.no_grad()` | Yes | Disables gradient tracking inside the block |
| `.cpu()` | No | Preserves graph |
| `.to(device)` | No | Preserves graph |
| `torch.stack()` | No | Preserves graph |
| `torch.cat()` | No | Preserves graph |
| `torch.sum()` | No | Preserves graph |
| `torch.mean()` | No | Preserves graph |
| `torch.log()` | No | Preserves graph |



#### Optimizers and normalizing returns

- On sgd, the longer the episode the bigger the gradient, so one long episode produces a huge action, which push the softmax to always choose one action

- it is interesting to normalize phi_t over the batch. 
```python
all_lp, all_G = [], []
for trajectory in trajectories:
    all_lp.append(torch.stack([l for r, l in trajectory]))
    all_G.append(calculate_returns(torch.stack([r for r, l in trajectory])))
G = torch.cat(all_G)
G = (G - G.mean()) / (G.std() + 1e-8)          # normalize returns across the batch
loss = -(torch.cat(all_lp) * G).sum() / len(trajectories)
```

- when phi is estimated by adding a baseline value function, the value functin itself should be trained over the whole batch (self contained training loop) careful not to overfit