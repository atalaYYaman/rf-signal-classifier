"""Ders 6a: kanal bozulmaları. Önce regresyon kontrolü, sonra temiz/bozuk karşılaştırma görseli.

Çalıştır:  .venv\\Scripts\\python src\\channel_demo.py
Çıktı:     experiments/06_channel_demo.png
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signals import CLASSES, N, make_sample

ROOT = Path(__file__).resolve().parent.parent

# 1) Regresyon: channel=0 iken üretici ders 5'teki çıktıyla birebir aynı mı?
h = hashlib.md5()
for i, c in enumerate(CLASSES):
    h.update(make_sample(c, 7.5, np.random.default_rng([9, i])).tobytes())
ref = (ROOT / "experiments" / ".clean_fingerprint").read_text().strip()
print("regresyon (channel=0 eski veri ile aynı):", "GEÇTİ" if h.hexdigest() == ref else f"BAŞARISIZ {h.hexdigest()} != {ref}")

# 2) Görsel: aynı tohum, temiz vs bozuk (SNR 20 dB)
show = ["tone", "qpsk", "qam16", "ofdm"]
fig, axes = plt.subplots(len(show), 4, figsize=(16, 2.8 * len(show)))
for r, name in enumerate(show):
    for c, (ch, label) in enumerate(((0.0, "temiz"), (1.0, "bozuk"))):
        x = make_sample(name, 20, np.random.default_rng([3, CLASSES.index(name)]), channel=ch)
        a = axes[r, 2 * c]
        a.scatter(x.real, x.imag, s=2, alpha=0.4); a.set_aspect("equal"); a.set_xlim(-3, 3); a.set_ylim(-3, 3)
        a.set_title(f"{name}: {label}, IQ düzlemi")
        sp = np.fft.fftshift(np.fft.fft(x * np.hanning(N)))
        b = axes[r, 2 * c + 1]
        b.plot(np.fft.fftshift(np.fft.fftfreq(N)), 20 * np.log10(np.abs(sp) / N + 1e-12), lw=0.6)
        b.set_ylim(-80, 0); b.set_title(f"{name}: {label}, spektrum")
fig.suptitle("Kanal bozulmaları (şiddet 1.0): çok yollu + zamanlama kayması + frekans kayması + faz gürültüsü + IQ dengesizliği", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "experiments" / "06_channel_demo.png", dpi=90)
print("yazıldı: experiments/06_channel_demo.png")
