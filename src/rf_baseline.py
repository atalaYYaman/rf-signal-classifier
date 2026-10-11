"""Ders 5: 20 el yapımı özellik + random forest, CNN ile aynı veri ve test seti üzerinde.

Çalıştır:  .venv\\Scripts\\python src\\rf_baseline.py
Çıktı:     experiments/05_rf.png, 05_rf_metrics.json
Not: CNN sayıları experiments/04_run1_metrics.json'dan okunur (aynı SignalDataset ayarları: seed, n_train/val/test).
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from dataset import SignalDataset
from features import NAMES, feature_matrix
from signals import CLASSES

OUT = Path(__file__).resolve().parent.parent / "experiments"
cnn = json.load(open(OUT / "04_run1_metrics.json"))
cfg = cnn["config"]                                   # CNN ile birebir aynı veri boyutları
t0 = time.time()


def load(split, n):
    ds = SignalDataset(split, n)
    x = (ds.x[:, 0] + 1j * ds.x[:, 1]).numpy()
    return feature_matrix(x), ds.y.numpy(), ds.snr.numpy()


Xtr, ytr, _ = load("train", cfg["n_train"])
Xva, yva, _ = load("val", cfg["n_val"])
Xte, yte, snr = load("test", cfg["n_test"])
print(f"özellikler hazır: {time.time() - t0:.0f} sn | train {Xtr.shape} test {Xte.shape}", flush=True)
assert np.isfinite(Xtr).all() and np.isfinite(Xte).all(), "NaN/inf özellik var"

rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
rf.fit(Xtr, ytr)
print(f"RF eğitildi: {time.time() - t0:.0f} sn", flush=True)
val_acc = float((rf.predict(Xva) == yva).mean())
pred = rf.predict(Xte)
test_acc = float((pred == yte).mean())
print(f"RF val doğruluk {val_acc:.3f} | TEST doğruluk {test_acc:.3f} | CNN test {cnn['test_acc']:.3f}", flush=True)

edges = np.array(cnn["snr_bins"])
snr_acc = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (snr >= lo) & (snr < hi if hi < 20 else snr <= hi)
    snr_acc.append(float((pred[m] == yte[m]).mean()))
print("SNR aralığı      RF     CNN")
for lo, hi, r, c in zip(edges[:-1], edges[1:], snr_acc, cnn["snr_acc"]):
    print(f"[{lo:+3d},{hi:+3d}) dB   {r:.3f}  {c:.3f}")
per_class = {c: float((pred[yte == i] == i).mean()) for i, c in enumerate(CLASSES)}

pi = permutation_importance(rf, Xva, yva, n_repeats=3, random_state=0, n_jobs=-1)
order = np.argsort(pi.importances_mean)[::-1]
print("permütasyon önemi (val):", ", ".join(f"{NAMES[i]}={pi.importances_mean[i]:.3f}" for i in order[:8]), flush=True)


def conf(mask):
    cm = np.zeros((len(CLASSES),) * 2)
    for t_, p_ in zip(yte[mask], pred[mask]):
        cm[t_, p_] += 1
    return cm / cm.sum(1, keepdims=True).clip(1)


fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(2, 3)
mid = (edges[:-1] + edges[1:]) / 2
ax = fig.add_subplot(gs[0, 0])
ax.bar(mid - 1.1, snr_acc, width=2.1, label=f"RF + 20 özellik ({test_acc:.3f})")
ax.bar(mid + 1.1, cnn["snr_acc"], width=2.1, label=f"CNN, ham IQ ({cnn['test_acc']:.3f})")
ax.axhline(0.1, color="k", ls=":"); ax.set_ylim(0, 1); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("test doğruluk")
ax.set_title("SNR'a göre doğruluk: RF vs CNN"); ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="y")

ax = fig.add_subplot(gs[1, 0])
xx = np.arange(len(CLASSES))
ax.bar(xx - 0.2, [per_class[c] for c in CLASSES], width=0.4, label="RF")
ax.bar(xx + 0.2, [cnn["per_class"][c] for c in CLASSES], width=0.4, label="CNN")
ax.set_xticks(xx); ax.set_xticklabels(CLASSES, rotation=45); ax.set_ylim(0, 1)
ax.set_title("Sınıfa göre doğruluk (tüm SNR)"); ax.legend(); ax.grid(alpha=0.3, axis="y")

ax = fig.add_subplot(gs[0, 1])
ax.barh([NAMES[i] for i in order[::-1]], pi.importances_mean[order[::-1]], xerr=pi.importances_std[order[::-1]])
ax.set_title("Özellik önemi (permütasyon, val)\nbir özellik karıştırılınca doğruluk ne kadar düşer")
ax.grid(alpha=0.3, axis="x")

for gcell, (mask, title) in zip((gs[0, 2], gs[1, 2]), ((snr >= 10, "RF karışıklık, SNR ≥ 10 dB"), (snr < 0, "RF karışıklık, SNR < 0 dB"))):
    a = fig.add_subplot(gcell)
    a.imshow(conf(mask), vmin=0, vmax=1, cmap="Blues"); a.set_title(title)
    a.set_xticks(range(10)); a.set_xticklabels(CLASSES, rotation=60, fontsize=8)
    a.set_yticks(range(10)); a.set_yticklabels(CLASSES, fontsize=8)
    a.set_xlabel("tahmin"); a.set_ylabel("gerçek")

ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
ax.text(0, 1, "Özet\n\n"
        f"RF: {test_acc:.3f}  ·  CNN: {cnn['test_acc']:.3f}\n"
        f"Parametre: RF 300 ağaç  ·  CNN {cnn['n_params']:,}\n"
        f"Girdi: RF 20 sayı  ·  CNN 2×1024 ham IQ\n"
        f"Eğitim: RF ~{time.time() - t0:.0f} sn (özellikler dahil)\n"
        f"        CNN ~{cnn['seconds'] / 60:.0f} dk", va="top", fontsize=12, family="monospace")
fig.suptitle("Ders 5: kademe (b) özellik + random forest, kademe (c) CNN ile aynı test setinde", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "05_rf.png", dpi=100)

json.dump(dict(test_acc=test_acc, val_acc=val_acc, snr_bins=[int(e) for e in edges], snr_acc=snr_acc,
               per_class=per_class, feature_names=NAMES,
               perm_importance={NAMES[i]: float(pi.importances_mean[i]) for i in range(len(NAMES))},
               seconds=time.time() - t0, cnn_test_acc=cnn["test_acc"]), open(OUT / "05_rf_metrics.json", "w"), indent=1)
print("yazıldı:", OUT / "05_rf.png", flush=True)
