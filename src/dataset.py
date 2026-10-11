"""PyTorch veri seti: sızıntısız train/val/test ayrımıyla sentetik IQ blokları.

Ayrım mantığı: her örnek (split, sınıf, indeks) üçlüsünden türetilen **bağımsız bir tohumla** üretilir.
Split numarası tohumun parçası olduğu için üç kümenin örnekleri birbirinin kopyası olamaz.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from signals import CLASSES, N, make_sample

SPLIT_ID = {"train": 0, "val": 1, "test": 2}
SNR_RANGE = (-10.0, 20.0)                      # D-006: uniform, dB
BASE_SEED = 20261010


class SignalDataset(Dataset):
    def __init__(self, split, n_per_class, classes=CLASSES, snr_range=SNR_RANGE, n=N, channel=0.0):
        self.classes = list(classes)
        xs, ys, snrs = [], [], []
        for label, name in enumerate(self.classes):
            for i in range(n_per_class):
                rng = np.random.default_rng([BASE_SEED, SPLIT_ID[split], label, i])
                snr = rng.uniform(*snr_range)
                xs.append(make_sample(name, snr, rng, n, channel))
                ys.append(label)
                snrs.append(snr)
        x = np.stack(xs)                                           # (M, n) karmaşık
        self.x = torch.from_numpy(np.stack([x.real, x.imag], axis=1))  # (M, 2, n): kanal 0 = I, kanal 1 = Q
        self.y = torch.tensor(ys, dtype=torch.long)
        self.snr = torch.tensor(snrs, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
