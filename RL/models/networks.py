import torch.nn as nn
import torch

class policy_NET(nn.Module):
    def __init__(self, device= "cpu"):

        super().__init__()
        
        self.l1 = nn.Linear(4, 50, device=device)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(50,2, device= device)
        self.sftmax = nn.Softmax(dim = -1)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.sftmax(x)
        return x 


class value_NET(nn.Module):
    def __init__(self,device) -> None:
        super().__init__()

        self.l1 = nn.Linear(4, 20, device=device)
        self.act = nn.LeakyReLU()
        self.l2 = nn.Linear(20, 10, device=device)
        self.l3 = nn.Linear(10, 1, device=device)

    def forward(self, x) -> torch.Tensor:
        x =self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.l3(x)

        return x # (B, 1)

