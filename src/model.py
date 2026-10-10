"""Ham IQ üzerinde çalışan küçük 1D CNN (kademe c için ilk taban çizgisi)."""
import torch
from torch import nn


def block(c_in, c_out, k=7):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class SmallCNN(nn.Module):
    """(batch, 2, 1024) -> (batch, n_classes) ham skorlar (logit)."""

    def __init__(self, n_classes=10, width=32):
        super().__init__()
        self.features = nn.Sequential(
            block(2, width), block(width, width * 2), block(width * 2, width * 2), block(width * 2, width * 4),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width * 4 * 2, n_classes))

    def forward(self, x):
        h = self.features(x)                                         # (B, C, L)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)         # zaman boyunca özetle: ortalama + en büyük
        return self.head(h)
