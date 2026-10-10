"""Ders 3: veri setini üret, sağlık kontrollerini yap, galeri çiz.

Çalıştır:  .venv\\Scripts\\python src\\dataset_check.py
Çıktı:     experiments/03_dataset_overview.png
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import SignalDataset
from signals import CLASSES, N, make_sample

t0 = time.time()
splits = {s: SignalDataset(s, n_per_class=n) for s, n in (("train", 500), ("val", 100), ("test", 100))}
print(f"üretim süresi: {time.time() - t0:.1f} sn")

# --- Kontroller ---------------------------------------------------------------
ok = True
for name, ds in splits.items():
    cx = ds.x[:, 0] + 1j * ds.x[:, 1]
    power = (cx.abs() ** 2).mean(dim=1)
    counts = np.bincount(ds.y.numpy(), minlength=len(CLASSES))
    print(f"{name:5s} şekil={tuple(ds.x.shape)} sınıf başı={counts.min()}-{counts.max()} "
          f"güç[min,max]=[{power.min():.4f},{power.max():.4f}] SNR[min,max]=[{ds.snr.min():.1f},{ds.snr.max():.1f}]")
    ok &= bool(abs(power - 1).max() < 1e-3) and counts.min() == counts.max()

hashes = {s: {ds.x[i].numpy().tobytes() for i in range(len(ds))} for s, ds in splits.items()}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = len(hashes[a] & hashes[b])
    print(f"sızıntı kontrolü {a}/{b}: ortak örnek = {overlap}")
    ok &= overlap == 0

again = SignalDataset("val", n_per_class=100)
same = bool((again.x == splits["val"].x).all())
print("tekrarlanabilirlik (aynı tohum -> aynı veri):", same)
ok &= same
print("TÜM KONTROLLER:", "GEÇTİ" if ok else "BAŞARISIZ")

# --- Galeri -------------------------------------------------------------------
fig, axes = plt.subplots(len(CLASSES), 3, figsize=(12, 2.2 * len(CLASSES)))
for r, name in enumerate(CLASSES):
    rng = np.random.default_rng([1, r])
    x20 = make_sample(name, 20, rng)
    x0 = make_sample(name, 0, rng)
    ax = axes[r, 0]
    ax.scatter(x20.real, x20.imag, s=2, alpha=0.4); ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_ylabel(name, fontsize=11, fontweight="bold")
    for c, (x, snr) in enumerate(((x20, 20), (x0, 0)), start=1):
        sp = np.fft.fftshift(np.fft.fft(x * np.hanning(N)))
        axes[r, c].plot(np.fft.fftshift(np.fft.fftfreq(N)), 20 * np.log10(np.abs(sp) / N + 1e-12), lw=0.6)
        axes[r, c].set_ylim(-80, 0)
    if r == 0:
        axes[r, 0].set_title("IQ düzlemi (20 dB)"); axes[r, 1].set_title("spektrum, 20 dB"); axes[r, 2].set_title("spektrum, 0 dB")
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "experiments" / "03_dataset_overview.png"
fig.savefig(out, dpi=90)
print("yazıldı:", out)
