import torch
import torch.nn as nn


class FilmLayer(nn.Module):
    def __init__(self, D_in, D):
        super().__init__()
        self.D = D

        self.weight = nn.Linear(D_in, self.D, 1)
        self.bias = nn.Linear(D_in, self.D, 1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """
        x: (B, F, T, D)
        embedding: (B, 1, 1, D_in)
        """
        # print(embedding.shape)
        # Apply film
        w = self.weight(embedding) # (B, D, F, T)
        b = self.bias(embedding) # (B, D, F, T)
        # print("X", x.shape)
        # print("W", w.shape)
        # print("B", b.shape)

        output = x * w + b

        return output