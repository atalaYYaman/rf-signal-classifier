"""Ders 1: IQ nedir? Üç sinyali kendimiz üretip dört açıdan bakıyoruz.

Çalıştır:  .venv\\Scripts\\python src\\iq_intro.py
Çıktı:     experiments/01_iq_intro.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)          # tohum: her çalıştırmada aynı sonuç
fs = 1000.0                              # örnekleme hızı: saniyede 1000 IQ örneği
N = 1000                                 # 1 saniyelik blok
t = np.arange(N) / fs


def add_noise(x, snr_db):
    """Karmaşık sinyale, istenen SNR'da (dB) karmaşık Gauss gürültü ekler."""
    p_sig = np.mean(np.abs(x) ** 2)
    p_noise = p_sig / 10 ** (snr_db / 10)
    n = np.sqrt(p_noise / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + n


# 1) Saf ton: tek frekansta dönen bir ok. x[n] = exp(j 2π f t)
f_tone = 50.0
tone = np.exp(1j * 2 * np.pi * f_tone * t)

# 2) AM: aynı taşıyıcının boyu 5 Hz'lik bir mesajla büyüyüp küçülür
msg = np.sin(2 * np.pi * 5 * t)
am = (1 + 0.6 * msg) * np.exp(1j * 2 * np.pi * f_tone * t)

# 3) QPSK: her sembol 4 fazdan biri (45°, 135°, 225°, 315°). Sembol başına 10 örnek.
sps = 10
bits = rng.integers(0, 4, N // sps)
phases = np.pi / 4 + bits * np.pi / 2
qpsk = np.repeat(np.exp(1j * phases), sps)

signals = {"Saf ton (50 Hz)": tone, "AM (mesaj 5 Hz)": am, "QPSK": qpsk}
snr_db = 20

fig, axes = plt.subplots(3, 4, figsize=(17, 9.5))
for row, (name, x) in zip(axes, signals.items()):
    xn = add_noise(x, snr_db)

    # Sütun 1: I ve Q zaman içinde (ilk 0,1 sn)
    ax = row[0]
    ax.plot(t[:100], xn.real[:100], label="I (gerçek)")
    ax.plot(t[:100], xn.imag[:100], label="Q (sanal)")
    ax.set_title(f"{name}\nI ve Q zamanda"); ax.set_xlabel("s"); ax.legend(loc="upper right", fontsize=7)

    # Sütun 2: IQ düzlemi (I'ya karşı Q) — sinyalin "izdüşümü"
    ax = row[1]
    ax.scatter(xn.real, xn.imag, s=3, alpha=0.5)
    ax.set_title("IQ düzlemi (takımyıldız)"); ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # Sütun 3: genlik |x| zamanda — AM'i burada görürsünüz
    ax = row[2]
    ax.plot(t[:300], np.abs(xn[:300]))
    ax.set_title("Genlik |x| (zarf)"); ax.set_xlabel("s"); ax.set_ylim(0, 2.2)

    # Sütun 4: spektrum — frekansa göre güç (dB)
    ax = row[3]
    spec = np.fft.fftshift(np.fft.fft(xn * np.hanning(N)))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
    ax.plot(freqs, 20 * np.log10(np.abs(spec) / N + 1e-12))
    ax.set_title("Spektrum"); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.set_ylim(-100, 0)
    ax.grid(alpha=0.3)

fig.suptitle(f"IQ'ya dört açıdan bakış (SNR = {snr_db} dB, fs = {fs:.0f} Hz)", fontsize=14)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "experiments" / "01_iq_intro.png"
fig.savefig(out, dpi=110)
print("yazıldı:", out)
