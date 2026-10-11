"""Ders 6b: temiz/bozuk veri üzerinde RF ve CNN. 2x2 matris: (eğitim verisi) x (test verisi).

Çalıştır:  .venv\\Scripts\\python src\\robustness.py
Önkoşul:   experiments/04_run1.pt (temiz CNN). Bozuk CNN için: python src/train.py --channel 1 --tag 06_cnn_ch1
Çıktı:     experiments/06_robustness.png, 06_robustness.json
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
from signals import CLASSES

OUT = Path(__file__).resolve().parent.parent / "experiments"
CH = 1.0
cfg = json.load(open(OUT / "04_run1_metrics.json"))["config"]
EDGES = np.array([-10, -5, 0, 5, 10, 15, 20])


def feats(split, n, channel):
    ds = SignalDataset(split, n, channel=channel)
    return feature_matrix((ds.x[:, 0] + 1j * ds.x[:, 1]).numpy()), ds.y.numpy(), ds.snr.numpy(), ds


def by_snr(pred, y, snr):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (snr >= lo) & (snr < hi if hi < 20 else snr <= hi)
        out.append(float((pred[m] == y[m]).mean()))
    return out


@torch.no_grad()
def cnn_pred(path, ds):
    net = SmallCNN(len(CLASSES), cfg["width"])
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return torch.cat([net(ds.x[i:i + 512]) for i in range(0, len(ds), 512)]).argmax(1).numpy()


# --- veri (test setleri: temiz ve bozuk, aynı tohum/indeks) --------------------
data = {}
for ch in (0.0, CH):
    data[ch] = {s: feats(s, cfg[f"n_{s}"], ch) for s in ("train", "test")}

res = {}          # (model, eğitim, test) -> dict(acc, snr)
for train_ch in (0.0, CH):
    Xtr, ytr, _, _ = data[train_ch]["train"]
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    for test_ch in (0.0, CH):
        Xte, yte, snr, _ = data[test_ch]["test"]
        p = rf.predict(Xte)
        res[("RF", train_ch, test_ch)] = dict(acc=float((p == yte).mean()), snr=by_snr(p, yte, snr))

for train_ch, ck in ((0.0, "04_run1.pt"), (CH, "06_cnn_ch1.pt")):
    if not (OUT / ck).exists():
        print("(atlandı: CNN", ck, "yok)")
        continue
    for test_ch in (0.0, CH):
        _, yte, snr, ds = data[test_ch]["test"]
        p = cnn_pred(OUT / ck, ds)
        res[("CNN", train_ch, test_ch)] = dict(acc=float((p == yte).mean()), snr=by_snr(p, yte, snr))

print("\nmodel  eğitim   test    doğruluk")
for (m, tr, te), v in res.items():
    print(f"{m:4s}  {'bozuk' if tr else 'temiz'}   {'bozuk' if te else 'temiz'}   {v['acc']:.3f}   SNR bin: "
          + " ".join(f"{a:.2f}" for a in v["snr"]))

# --- grafik ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
mid = (EDGES[:-1] + EDGES[1:]) / 2
keys = [k for k in res if k[2] == CH]                       # bozuk test üzerinde tüm sistemler
for ax, test_ch, title in ((axes[0], 0.0, "Temiz test verisi"), (axes[1], CH, "Bozuk test verisi")):
    ks = [k for k in res if k[2] == test_ch]
    w = 4.4 / max(len(ks), 1)
    for i, k in enumerate(ks):
        ax.bar(mid + (i - (len(ks) - 1) / 2) * w, res[k]["snr"], width=w * 0.95,
               label=f"{k[0]} (eğitim: {'bozuk' if k[1] else 'temiz'}) {res[k]['acc']:.3f}")
    ax.axhline(0.1, color="k", ls=":"); ax.set_ylim(0, 1); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("test doğruluk")
    ax.set_title(title); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")
fig.suptitle("Ders 6: kanal bozulmaları — RF ve CNN, temiz/bozuk eğitim", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "06_robustness.png", dpi=100)
json.dump({f"{m}|train={'bozuk' if tr else 'temiz'}|test={'bozuk' if te else 'temiz'}": v for (m, tr, te), v in res.items()},
          open(OUT / "06_robustness.json", "w"), indent=1)
print("yazıldı: experiments/06_robustness.png")
