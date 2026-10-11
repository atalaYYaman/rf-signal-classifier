"""El yapımı özellikler (kademe b). Şemadaki 21 özelliğin 12'sine karşılık gelen 20 sayı.

Karşılık gelen şema özellikleri: F2 OBW, F3 SFM, F4 PMR/çizgi sayısı, F5 CV_a, F6 κ_a, F7 anlık frekans,
F11 CSR, F12 üs testi (x^2, x^4), F13 sembol hızı (siklik), F14 CP korelasyonu, F15 kümülantlar, F16 darbe.
Uygulanmayanlar: F1 (SNR kapısı), F8 chirp, F9 simetri, F10 ayna, F17 STFT, F18 histogram, F19-F21.
Eşikler yok: özellikler sayı olarak döner, sınıflandırmayı random forest yapar.
"""
import numpy as np

NAMES = ["pmr_db", "n_lines", "sfm", "csr", "obw", "cv_a", "kurt_a", "p10_p90", "fi_std", "fi_kurt",
         "fi_bimod", "fi_std_smooth", "pow2_db", "pow4_db", "cyc_db", "cp_abs", "cp_peak", "c20", "c40", "c42"]
EPS = 1e-12


def _peak_to_median_db(p):
    return 10 * np.log10(p.max(1) / (np.median(p, 1) + EPS) + EPS)


def feature_matrix(x):
    """x: (M, N) karmaşık. Dönüş: (M, 20) float64."""
    x = np.asarray(x, dtype=np.complex128)
    M, N = x.shape
    hann = np.hanning(N)

    # Welch PSD (4 segment x 256)
    seg = 256
    K = N // seg
    w = np.hanning(seg)
    P = (np.abs(np.fft.fft(x[:, :K * seg].reshape(M, K, seg) * w, axis=2)) ** 2).mean(1)
    P = np.fft.fftshift(P, axes=1)
    tot = P.sum(1)
    med = np.median(P, 1)
    pmr = _peak_to_median_db(P)
    n_lines = (P > med[:, None] * 10 ** 1.3).sum(1)                    # medyanın 13 dB üstü
    sfm = np.exp(np.mean(np.log(P + EPS), 1)) / P.mean(1)
    csr = P.max(1) / tot
    cdf = np.cumsum(P, 1) / tot[:, None]
    obw = (np.argmax(cdf >= 0.995, 1) - np.argmax(cdf >= 0.005, 1)) / seg

    # Zarf
    a = np.abs(x)
    cv = a.std(1) / a.mean(1)
    kurt = (a ** 4).mean(1) / ((a ** 2).mean(1) ** 2)
    p10, p90 = np.percentile(a, [10, 90], axis=1)
    p10_p90 = p10 / (p90 + EPS)

    # Anlık frekans
    fi = np.angle(x[:, 1:] * np.conj(x[:, :-1])) / (2 * np.pi)
    mu, sd = fi.mean(1), fi.std(1)
    z = (fi - mu[:, None]) / (sd[:, None] + EPS)
    fi_kurt = (z ** 4).mean(1)
    skew = (z ** 3).mean(1)
    fi_bimod = (skew ** 2 + 1) / (fi_kurt + EPS)                       # >0.55: iki tepeli dağılım işareti
    L = 8
    fs_ = fi[:, : (fi.shape[1] // L) * L].reshape(M, -1, L).mean(2)    # 8 örnek ortalaması: gürültüyü kırar
    fi_std_smooth = fs_.std(1)

    # Üs testleri ve siklik özellik
    pw = {}
    for m in (2, 4):
        S = np.abs(np.fft.fft((x ** m) * hann, axis=1)) ** 2
        pw[m] = _peak_to_median_db(S)
    e = a ** 2 - (a ** 2).mean(1, keepdims=True)
    E = np.abs(np.fft.fft(e * hann, axis=1)) ** 2
    cyc = _peak_to_median_db(E[:, 4:N // 2])                           # çok düşük frekanslar atılır

    # OFDM: CP korelasyonu (gecikme 64, pencere 16)
    r = x[:, :-64] * np.conj(x[:, 64:])
    cs = np.cumsum(r, axis=1)
    win = np.abs(cs[:, 16:] - cs[:, :-16])
    cp_abs = win.max(1) / (16 * (a ** 2).mean(1))
    cp_peak = win.max(1) / (win.mean(1) + EPS)

    # Kümülantlar (birim C21'e normalize)
    c21 = (a ** 2).mean(1)
    m2 = (x ** 2).mean(1)
    c20 = np.abs(m2) / c21
    c40 = np.abs((x ** 4).mean(1) - 3 * m2 ** 2) / c21 ** 2
    c42 = ((a ** 4).mean(1) - np.abs(m2) ** 2 - 2 * c21 ** 2) / c21 ** 2

    F = np.stack([pmr, n_lines, sfm, csr, obw, cv, kurt, p10_p90, sd, fi_kurt, fi_bimod, fi_std_smooth,
                  pw[2], pw[4], cyc, cp_abs, cp_peak, c20, c40, c42], axis=1)
    return F.astype(np.float64)
