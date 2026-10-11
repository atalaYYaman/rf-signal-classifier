"""Ders 8b: 14 sınıf, bozuk veri. RF ve CNN karşılaştırması + hangi sınıflar karışıyor.

Çalıştır:  .venv\\Scripts\\python src\\exp8_report.py   (önce experiments/08_runs.sh bitmiş olmalı)
Çıktı:     experiments/08_report.png, 08_report.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from dataset import SignalDataset
from features import feature_matrix
from model import SmallCNN
from signals import CLASSES14

OUT = Path(__file__).resolve().parent.parent / "experiments"
CH, C = 1.0, CLASSES14
NC = len(C)
EDGES = np.array([-10, -5, 0, 5, 10, 15, 20])


def load(split, n):
    ds = SignalDataset(split, n, classes=C, channel=CH)
    return ds, feature_matrix((ds.x[:, 0] + 1j * ds.x[:, 1]).numpy())


test_ds, Xte = load("test", 300)
yte, snr = test_ds.y.numpy(), test_ds.snr.numpy()
preds = {}

for n in (1500, 6000):
    tr_ds, Xtr = load("train", n)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xtr, tr_ds.y.numpy())
    preds[f"RF n={n}"] = rf.predict(Xte)
    print(f"RF n={n} bitti", flush=True)

for n in (1500, 6000):
    ck = OUT / f"08_c14_n{n}.pt"
    if ck.exists():
        net = SmallCNN(NC, 32)
        net.load_state_dict(torch.load(ck, weights_only=True))
        net.eval()
        with torch.no_grad():
            preds[f"CNN n={n}"] = torch.cat([net(test_ds.x[i:i + 512]) for i in range(0, len(test_ds), 512)]).argmax(1).numpy()


def bins(p):
    return [float((p[(snr >= lo) & (snr < hi if hi < 20 else snr <= hi)] == yte[(snr >= lo) & (snr < hi if hi < 20 else snr <= hi)]).mean())
            for lo, hi in zip(EDGES[:-1], EDGES[1:])]


rep = {}
print(f"\n{'sistem':12s} {'tüm':>6s} {'ilk10':>6s} {'yeni4':>6s}   SNR bin: " + " ".join(f"{lo:+d}" for lo in EDGES[:-1]))
for k, p in preds.items():
    ok = p == yte
    rep[k] = dict(acc=float(ok.mean()), old10=float(ok[yte < 10].mean()), new4=float(ok[yte >= 10].mean()), snr=bins(p),
                  per_class={c: float(ok[yte == i].mean()) for i, c in enumerate(C)})
    print(f"{k:12s} {rep[k]['acc']:6.3f} {rep[k]['old10']:6.3f} {rep[k]['new4']:6.3f}   " + " ".join(f"{a:.2f}" for a in rep[k]["snr"]))


def conf(p):
    cm = np.zeros((NC, NC))
    for t, q in zip(yte, p):
        cm[t, q] += 1
    return cm / cm.sum(1, keepdims=True)


best = {"RF": "RF n=6000", "CNN": "CNN n=6000"}
for name, k in best.items():
    if k not in preds:
        continue
    cm = conf(preds[k]); np.fill_diagonal(cm, 0)
    top = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:6], cm.shape))[0]
    print(f"\n{k} en sık karışıklıklar (gerçek → tahmin, oran):")
    for t, q in top:
        print(f"  {C[t]:6s} → {C[q]:6s}  {cm[t, q]:.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax = axes[0, 0]
mid = (EDGES[:-1] + EDGES[1:]) / 2
for k, v in rep.items():
    ax.plot(mid, v["snr"], "o-", label=f"{k} ({v['acc']:.3f})")
ax.set_ylim(0, 1); ax.axhline(1 / NC, color="k", ls=":"); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("test doğruluk")
ax.set_title("SNR'a göre doğruluk (14 sınıf, bozuk veri)"); ax.legend(); ax.grid(alpha=0.3)
ax = axes[1, 0]
xx = np.arange(NC)
for j, k in enumerate(("RF n=6000", "CNN n=6000")):
    if k in rep:
        ax.bar(xx + (j - 0.5) * 0.4, [rep[k]["per_class"][c] for c in C], width=0.4, label=k)
ax.set_xticks(xx); ax.set_xticklabels(C, rotation=60); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis="y")
ax.set_title("Sınıfa göre doğruluk")
for ax, k in ((axes[0, 1], "RF n=6000"), (axes[1, 1], "CNN n=6000")):
    if k in preds:
        ax.imshow(conf(preds[k]), vmin=0, vmax=1, cmap="Blues"); ax.set_title(f"{k}: karışıklık (tüm SNR)")
        ax.set_xticks(range(NC)); ax.set_xticklabels(C, rotation=70, fontsize=8)
        ax.set_yticks(range(NC)); ax.set_yticklabels(C, fontsize=8); ax.set_xlabel("tahmin"); ax.set_ylabel("gerçek")
for ax, k in ((axes[0, 2], "RF n=6000"), (axes[1, 2], "CNN n=6000")):
    if k in preds:
        m = snr >= 10
        cm = np.zeros((NC, NC))
        for t, q in zip(yte[m], preds[k][m]):
            cm[t, q] += 1
        ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues"); ax.set_title(f"{k}: karışıklık, SNR ≥ 10 dB")
        ax.set_xticks(range(NC)); ax.set_xticklabels(C, rotation=70, fontsize=8)
        ax.set_yticks(range(NC)); ax.set_yticklabels(C, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "08_report.png", dpi=95)
json.dump(rep, open(OUT / "08_report.json", "w"), indent=1)
print("\nyazıldı: experiments/08_report.png")
