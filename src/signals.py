"""Sentetik IQ sinyal üreticileri (başlangıç alt kümesi: 10 sınıf).

Konvansiyonlar (docs/DECISIONS.md D-006):
- Blok uzunluğu N = 1024 karmaşık örnek.
- Frekanslar "normalize": 1.0 = örnekleme hızı; geçerli aralık (-0.5, 0.5).
- Her üretici sinyali **birim güce** (ortalama |x|^2 = 1) normalize eder.
- Gürültü, birim güçlü sinyale eklenir: SNR = örnek başına sinyal gücü / gürültü gücü.
- Sonda tüm blok yeniden birim güce çekilir: mutlak seviye bir ipucu olamaz.
"""
import numpy as np

N = 1024
CLASSES = ["noise", "tone", "am", "fm", "bpsk", "qpsk", "qam16", "fsk2", "ofdm", "radar"]


def _unit_power(x):
    p = np.mean(np.abs(x) ** 2)
    return x / np.sqrt(p) if p > 0 else x


def _carrier(rng, n, f_range=0.2):
    f0 = rng.uniform(-f_range, f_range)
    return np.exp(1j * (2 * np.pi * f0 * np.arange(n) + rng.uniform(0, 2 * np.pi)))


def _rrc(beta, span, sps):
    """Kök yükseltilmiş kosinüs süzgeci (sembolleri bant sınırlı hâle getirir)."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def _linear_mod(rng, constellation, n):
    """Doğrusal sayısal modülasyon: rastgele semboller -> RRC süzgeci -> taşıyıcı."""
    sps = int(rng.integers(4, 13))
    beta = rng.uniform(0.25, 0.5)
    n_sym = n // sps + 16
    syms = constellation[rng.integers(0, len(constellation), n_sym)]
    up = np.zeros(n_sym * sps, dtype=complex)
    up[::sps] = syms
    y = np.convolve(up, _rrc(beta, 8, sps), mode="same")
    start = int(rng.integers(0, 8 * sps))
    y = y[start:start + n]
    return _unit_power(y * _carrier(rng, n, f_range=0.05))


_QPSK = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
_QAM16 = np.array([a + 1j * b for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)], dtype=complex)


def gen_noise(rng, n=N):
    return np.zeros(n, dtype=complex)         # yalnızca gürültü eklenecek (bkz. make_sample)


def gen_tone(rng, n=N):
    return _carrier(rng, n, f_range=0.4)


def gen_am(rng, n=N):
    t = np.arange(n)
    msg = np.sin(2 * np.pi * rng.uniform(0.002, 0.01) * t + rng.uniform(0, 2 * np.pi))
    m = rng.uniform(0.3, 0.9)
    return _unit_power((1 + m * msg) * _carrier(rng, n, f_range=0.3))


def gen_fm(rng, n=N):
    t = np.arange(n)
    fm = rng.uniform(0.002, 0.01)
    fd = rng.uniform(0.02, 0.08)
    ph = 2 * np.pi * rng.uniform(-0.3, 0.3) * t + (fd / fm) * np.sin(2 * np.pi * fm * t + rng.uniform(0, 2 * np.pi))
    return np.exp(1j * (ph + rng.uniform(0, 2 * np.pi)))


def gen_bpsk(rng, n=N):
    return _linear_mod(rng, np.array([1 + 0j, -1 + 0j]), n)


def gen_qpsk(rng, n=N):
    return _linear_mod(rng, _QPSK, n)


def gen_qam16(rng, n=N):
    return _linear_mod(rng, _QAM16, n)


def gen_fsk2(rng, n=N):
    sps = int(rng.integers(8, 25))
    dev = rng.uniform(0.02, 0.1)
    bits = rng.integers(0, 2, n // sps + 2) * 2 - 1
    f = np.repeat(bits * dev, sps)[:n]
    ph = 2 * np.pi * np.cumsum(f) + rng.uniform(0, 2 * np.pi)
    return np.exp(1j * ph) * _carrier(rng, n, f_range=0.1)


def gen_ofdm(rng, n=N):
    nfft, cp, used = 64, 16, 48
    n_sym = n // (nfft + cp) + 2
    out = []
    for _ in range(n_sym):
        X = np.zeros(nfft, dtype=complex)
        idx = np.r_[1:used // 2 + 1, nfft - used // 2:nfft]
        X[idx] = _QPSK[rng.integers(0, 4, len(idx))]
        x = np.fft.ifft(X) * np.sqrt(nfft)
        out.append(np.r_[x[-cp:], x])
    y = np.concatenate(out)
    start = int(rng.integers(0, nfft + cp))
    return _unit_power(y[start:start + n] * _carrier(rng, n, f_range=0.05))


def gen_radar(rng, n=N):
    pri = int(rng.integers(100, 301))
    width = int(rng.integers(10, 51))
    gate = np.zeros(n)
    for s in range(int(rng.integers(0, pri)), n, pri):
        gate[s:s + width] = 1.0
    return _unit_power(gate * _carrier(rng, n, f_range=0.3))


GENERATORS = {
    "noise": gen_noise, "tone": gen_tone, "am": gen_am, "fm": gen_fm, "bpsk": gen_bpsk,
    "qpsk": gen_qpsk, "qam16": gen_qam16, "fsk2": gen_fsk2, "ofdm": gen_ofdm, "radar": gen_radar,
}


def make_sample(name, snr_db, rng, n=N):
    """Tek bir IQ bloğu üretir: sinyal (birim güç) + karmaşık Gauss gürültü, sonra birim güce normalize."""
    s = GENERATORS[name](rng, n)
    sigma2 = 10 ** (-snr_db / 10) if name != "noise" else 1.0
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return _unit_power(s + noise).astype(np.complex64)
