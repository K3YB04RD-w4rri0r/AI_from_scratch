

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