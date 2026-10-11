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
# D-008: genişletilmiş küme. Yeni sınıflar sona eklenir; ilk 10 sınıfın etiket indeksi (ve dolayısıyla tohumu) değişmez.
CLASSES14 = CLASSES + ["psk8", "qam64", "ook", "gfsk"]


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


_PSK8 = np.exp(1j * np.arange(8) * np.pi / 4)
_QAM64 = np.array([a + 1j * b for a in range(-7, 8, 2) for b in range(-7, 8, 2)], dtype=complex)


def gen_psk8(rng, n=N):
    return _linear_mod(rng, _PSK8, n)


def gen_qam64(rng, n=N):
    return _linear_mod(rng, _QAM64, n)


def gen_ook(rng, n=N):
    return _linear_mod(rng, np.array([0 + 0j, 1 + 0j]), n)


def gen_gfsk(rng, n=N):
    """Gauss süzgeçli FSK: bitler gauss süzgeçten geçip frekansı sürer (BT 0.3-0.5, mod. indeksi h 0.5-1)."""
    sps = int(rng.integers(8, 25))
    bt = rng.uniform(0.3, 0.5)
    h = rng.uniform(0.5, 1.0)
    bits = rng.integers(0, 2, n // sps + 8) * 2 - 1
    up = np.repeat(bits.astype(float), sps)
    tt = np.arange(-2 * sps, 2 * sps + 1) / sps
    g = np.exp(-0.5 * (tt / (np.sqrt(np.log(2)) / (2 * np.pi * bt))) ** 2)
    f = np.convolve(up, g / g.sum(), mode="same") * (h / (2 * sps))
    start = int(rng.integers(0, 4 * sps))
    f = f[start:start + n]
    return np.exp(1j * (2 * np.pi * np.cumsum(f) + rng.uniform(0, 2 * np.pi))) * _carrier(rng, n, f_range=0.1)


GENERATORS = {
    "noise": gen_noise, "tone": gen_tone, "am": gen_am, "fm": gen_fm, "bpsk": gen_bpsk,
    "qpsk": gen_qpsk, "qam16": gen_qam16, "fsk2": gen_fsk2, "ofdm": gen_ofdm, "radar": gen_radar,
    "psk8": gen_psk8, "qam64": gen_qam64, "ook": gen_ook, "gfsk": gen_gfsk,
}


def apply_channel(x, rng, severity):
    """Alıcıya varmadan önce sinyale uygulanan kanal/donanım bozulmaları (D-007).

    severity 0..1 aralıkların ölçeğidir. Sıra: çok yollu -> örnekleme/zamanlama kayması -> frekans kayması
    -> faz gürültüsü -> IQ dengesizliği. Gürültü (AWGN) bundan sonra eklenir.
    """
    n = len(x)
    k = np.arange(n)
    # 1) çok yollu yayılma: doğrudan yol + 0-3 gecikmiş yansıma (3-12 dB zayıf, 1-5 örnek gecikme)
    h = np.zeros(6, dtype=complex)
    h[0] = 1.0
    for _ in range(int(rng.integers(0, 4))):
        h[int(rng.integers(1, 6))] += severity * 10 ** (-rng.uniform(3, 12) / 20) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    x = np.convolve(x, h / np.sqrt(np.sum(np.abs(h) ** 2)))[:n]
    # 2) örnekleme hızı ofseti (±%2) ve kesirli gecikme: doğrusal aradeğerleme
    pos = np.clip(k * (1 + severity * rng.uniform(-0.02, 0.02)) + severity * rng.uniform(0, 1), 0, n - 1)
    x = np.interp(pos, k, x.real) + 1j * np.interp(pos, k, x.imag)
    # 3) taşıyıcı frekans kayması (±0.03 döngü/örnek)
    x = x * np.exp(2j * np.pi * severity * rng.uniform(-0.03, 0.03) * k)
    # 4) faz gürültüsü: rastgele yürüyüş, örnek başına σ ≤ 0.01 rad
    x = x * np.exp(1j * np.cumsum(severity * rng.uniform(0, 0.01) * rng.standard_normal(n)))
    # 5) IQ dengesizliği: kazanç ±1 dB, faz ±5°
    g = 10 ** (severity * rng.uniform(-1, 1) / 20)
    phi = np.deg2rad(severity * rng.uniform(-5, 5))
    x = x.real + 1j * g * (x.imag * np.cos(phi) + x.real * np.sin(phi))
    return _unit_power(x)


def make_sample(name, snr_db, rng, n=N, channel=0.0):
    """Tek bir IQ bloğu üretir: sinyal (birim güç) [+ kanal bozulmaları] + karmaşık Gauss gürültü,
    sonra birim güce normalize. channel=0 ise bozulma yok ve eski veri birebir aynıdır."""
    s = GENERATORS[name](rng, n)
    if channel > 0:
        s = apply_channel(s, rng, channel)
    sigma2 = 10 ** (-snr_db / 10) if name != "noise" else 1.0
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return _unit_power(s + noise).astype(np.complex64)
