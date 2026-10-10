"""Ders 2: Gürültü ve SNR. Aynı sinyal, azalan SNR'da.

Çalıştır:  .venv\\Scripts\\python src\\snr_intro.py
Çıktı:     experiments/02_snr_intro.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

rng = np.random.default_rng(7)
SPS = 10                                  # sembol başına örnek


def add_noise(x, snr_db):
    """Örnek başına SNR'ı (dB) verilen değere ayarlayarak karmaşık Gauss gürültü ekler."""
    p_noise = np.mean(np.abs(x) ** 2) / 10 ** (snr_db / 10)
    n = np.sqrt(p_noise / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + n


def qpsk_symbols(n_sym):
    k = rng.integers(0, 4, n_sym)
    return k, np.exp(1j * (np.pi / 4 + k * np.pi / 2))


def Q(x):                                  # Gauss kuyruk olasılığı
    return 0.5 * erfc(x / np.sqrt(2))


snrs_show = [20, 10, 5, 0, -5]
fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(3, len(snrs_show), height_ratios=[1, 1, 1.15])

# Satır 1: QPSK takımyıldızı, SNR düştükçe
k, s = qpsk_symbols(3000)
x = np.repeat(s, SPS)
for j, snr in enumerate(snrs_show):
    ax = fig.add_subplot(gs[0, j])
    xn = add_noise(x, snr)[::SPS]         # sembol başına 1 örnek (gürültülü, süzülmemiş)
    ax.scatter(xn.real, xn.imag, s=2, alpha=0.4)
    ax.set_title(f"QPSK IQ düzlemi, SNR = {snr} dB")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.grid(alpha=0.3)

# Satır 2: aynı SNR'larda spektrum: ton (dar çizgi) ve QPSK (geniş tümsek)
N = 4000
t = np.arange(N)
tone = np.exp(1j * 2 * np.pi * 0.1 * t)
qp = np.repeat(qpsk_symbols(N // SPS)[1], SPS)
win = np.hanning(N)
for j, snr in enumerate(snrs_show):
    ax = fig.add_subplot(gs[1, j])
    for sig, name in ((tone, "ton"), (qp, "QPSK")):
        sp = np.fft.fftshift(np.fft.fft(add_noise(sig, snr) * win))
        ax.plot(np.fft.fftshift(np.fft.fftfreq(N)), 20 * np.log10(np.abs(sp) / N + 1e-12), lw=0.7, label=name)
    ax.set_title(f"Spektrum, SNR = {snr} dB"); ax.set_ylim(-90, 0); ax.grid(alpha=0.3)
    ax.set_xlabel("normalize frekans")
    if j == 0:
        ax.legend(fontsize=8); ax.set_ylabel("dB")

# Satır 3: sembol hata oranı (SER) vs SNR, ölçüm ve kuram
ax = fig.add_subplot(gs[2, :])
snr_grid = np.arange(-10, 16, 1)
k, s = qpsk_symbols(40000)
x = np.repeat(s, SPS)
ser_meas, ser_raw = [], []
for snr in snr_grid:
    xn = add_noise(x, snr)
    # a) süzgeçsiz: sembolün yalnızca ilk örneğine bak
    raw = xn[::SPS]
    # b) sembol başına 10 örneği ortala (eşleşmiş süzgecin en basit hâli)
    avg = xn.reshape(-1, SPS).mean(axis=1)
    for est, store in ((raw, ser_raw), (avg, ser_meas)):
        kh = (np.angle(est) % (2 * np.pi) // (np.pi / 2)).astype(int)   # çeyrek düzlem -> sembol
        store.append(np.mean(kh != k))
lin = 10 ** (snr_grid / 10)
pe = Q(np.sqrt(lin))                      # tek örnek (süzgeçsiz)
pe_avg = Q(np.sqrt(SPS * lin))            # 10 örnek ortalaması
ax.semilogy(snr_grid, ser_raw, "o", label="ölçüm: tek örnek")
ax.semilogy(snr_grid, 1 - (1 - pe) ** 2, "-", color="C0", alpha=.6, label="kuram: tek örnek")
ax.semilogy(snr_grid, np.maximum(ser_meas, 1e-5), "s", label="ölçüm: 10 örnek ortalaması")
ax.semilogy(snr_grid, 1 - (1 - pe_avg) ** 2, "-", color="C1", alpha=.6, label="kuram: 10 örnek ortalaması")
ax.axhline(0.75, color="k", ls=":", lw=1); ax.text(-9.8, 0.62, "rastgele tahmin (%75 hata)", fontsize=8)
ax.set_ylim(1e-4, 1); ax.set_xlabel("örnek başına SNR (dB)"); ax.set_ylabel("sembol hata oranı (SER)")
ax.set_title("QPSK: SNR düştükçe hata artar. Ortalama almak (işleme kazancı) yaklaşık +10 dB kazandırır")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "experiments" / "02_snr_intro.png"
fig.savefig(out, dpi=105)
print("yazıldı:", out)
print("SER @ 0 dB: tek örnek %.3f | 10 ort. %.4f" % (ser_raw[list(snr_grid).index(0)], ser_meas[list(snr_grid).index(0)]))
