import torch.nn as nn
import torch
"""
Normalization layer, results of a matrix closer to 0-1 for percentage values easier to manage data and excentuate deviations by making mean ~ 0 and var ~ 1

eps = epsilon (There so the values never divide by zero)
scale = gamma
shift = beta
"""
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift