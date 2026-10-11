"""Ders 8a: yeni 4 sınıfın sağlık kontrolü ve galerisi. Çıktı: experiments/08_new_classes.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import SignalDataset
from signals import CLASSES14, N, make_sample

new = CLASSES14[10:]
ds = SignalDataset("val", 50, classes=CLASSES14)
cx = (ds.x[:, 0] + 1j * ds.x[:, 1]).numpy()
print("şekil", tuple(ds.x.shape), "| sonlu:", bool(np.isfinite(cx).all()),
      "| güç min/max: %.4f %.4f" % ((abs(cx) ** 2).mean(1).min(), (abs(cx) ** 2).mean(1).max()))
tr = SignalDataset("train", 50, classes=CLASSES14)
a = {x.numpy().tobytes() for x in ds.x}; b = {x.numpy().tobytes() for x in tr.x}
print("train/val ortak örnek:", len(a & b))

fig, axes = plt.subplots(len(new), 4, figsize=(15, 2.7 * len(new)))
for r, name in enumerate(new):
    rng = np.random.default_rng([5, r])
    for c, (snr, ch) in enumerate(((30, 0.0), (30, 1.0))):
        x = make_sample(name, snr, np.random.default_rng([5, r]), channel=ch)
        axes[r, 2 * c].scatter(x.real, x.imag, s=2, alpha=0.4); axes[r, 2 * c].set_aspect("equal")
        axes[r, 2 * c].set_xlim(-3, 3); axes[r, 2 * c].set_ylim(-3, 3)
        axes[r, 2 * c].set_title(f"{name} IQ, {'bozuk' if ch else 'temiz'} (30 dB)")
        sp = np.fft.fftshift(np.fft.fft(x * np.hanning(N)))
        axes[r, 2 * c + 1].plot(np.fft.fftshift(np.fft.fftfreq(N)), 20 * np.log10(np.abs(sp) / N + 1e-12), lw=0.6)
        axes[r, 2 * c + 1].set_ylim(-90, 0); axes[r, 2 * c + 1].set_title(f"{name} spektrum")
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "experiments" / "08_new_classes.png"
fig.savefig(out, dpi=85); print("yazıldı:", out)
