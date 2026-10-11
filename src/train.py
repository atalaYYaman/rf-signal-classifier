"""Ders 4: ilk model. Eğit, değerlendir, SNR'a göre doğruluk eğrisini ve karışıklık matrisini çiz.

Çalıştır:  .venv\\Scripts\\python src\\train.py
Çıktı:     experiments/04_run1.png, 04_run1_metrics.json (+ 04_run1.pt, git'e girmez)
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SignalDataset
from model import SmallCNN
from signals import CLASSES, CLASSES14

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--channel", type=float, default=0.0, help="kanal bozulma şiddeti 0..1 (D-007)")
ap.add_argument("--tag", default="04_run1", help="çıktı dosyalarının öneki")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--seed", type=int, default=0, help="model başlangıcı ve karıştırma tohumu (veri tohumu sabit)")
ap.add_argument("--n_train", type=int, default=1500, help="sınıf başına eğitim örneği")
ap.add_argument("--classes", type=int, default=10, choices=(10, 14), help="sınıf kümesi (D-008)")
ap.add_argument("--width", type=int, default=32, help="CNN taban kanal sayısı")
args = ap.parse_args()

CFG = dict(seed=args.seed, n_train=args.n_train, n_val=200, n_test=300, batch=128, epochs=args.epochs, lr=2e-3, width=args.width,
           channel=args.channel, tag=args.tag, n_classes=args.classes)
CLASSES = CLASSES14 if args.classes == 14 else CLASSES   # bundan sonra tüm betik seçilen kümeyi kullanır
OUT = Path(__file__).resolve().parent.parent / "experiments"
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # D-002: cihaz koda sabitlenmez
print("cihaz:", device)

t0 = time.time()
train_ds = SignalDataset("train", CFG["n_train"], classes=CLASSES, channel=CFG["channel"])
val_ds = SignalDataset("val", CFG["n_val"], classes=CLASSES, channel=CFG["channel"])
test_ds = SignalDataset("test", CFG["n_test"], classes=CLASSES, channel=CFG["channel"])
print(f"veri üretildi: {time.time() - t0:.0f} sn | train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")

g = torch.Generator().manual_seed(CFG["seed"])
train_dl = DataLoader(train_ds, batch_size=CFG["batch"], shuffle=True, generator=g)
val_dl = DataLoader(val_ds, batch_size=512)

model = SmallCNN(len(CLASSES), CFG["width"]).to(device)
n_params = sum(p.numel() for p in model.parameters())
print("parametre sayısı:", n_params)
opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(ds):
    model.eval()
    logits = torch.cat([model(x.to(device)).cpu() for x, _ in DataLoader(ds, batch_size=512)])
    loss = loss_fn(logits, ds.y).item()
    pred = logits.argmax(1)
    return loss, (pred == ds.y).float().mean().item(), pred


hist = {"train_loss": [], "val_loss": [], "val_acc": []}
for ep in range(1, CFG["epochs"] + 1):
    model.train()
    run, seen = 0.0, 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        run += loss.item() * len(y); seen += len(y)
    sched.step()
    vl, va, _ = evaluate(val_ds)
    hist["train_loss"].append(run / seen); hist["val_loss"].append(vl); hist["val_acc"].append(va)
    print(f"epoch {ep:2d}/{CFG['epochs']}  train_loss {run / seen:.3f}  val_loss {vl:.3f}  val_acc {va:.3f}  ({time.time() - t0:.0f} sn)")

# --- Test: yalnızca bir kez, eğitim bittikten sonra ---------------------------
tl, ta, pred = evaluate(test_ds)
y, snr = test_ds.y.numpy(), test_ds.snr.numpy()
pred = pred.numpy()
print(f"TEST doğruluk: {ta:.3f}")

edges = np.arange(-10, 21, 5)
snr_acc = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (snr >= lo) & (snr < hi if hi < 20 else snr <= hi)
    snr_acc.append(float((pred[m] == y[m]).mean()))
    print(f"  SNR [{lo:+d},{hi:+d}) dB: doğruluk {snr_acc[-1]:.3f} (n={m.sum()})")

per_class = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)}


def conf(mask):
    cm = np.zeros((len(CLASSES), len(CLASSES)))
    for t_, p_ in zip(y[mask], pred[mask]):
        cm[t_, p_] += 1
    return cm / cm.sum(axis=1, keepdims=True).clip(1)


# --- Grafik -------------------------------------------------------------------
fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0])
ax.plot(hist["train_loss"], label="train loss"); ax.plot(hist["val_loss"], label="val loss")
ax.set_xlabel("epoch"); ax.set_ylabel("kayıp"); ax.set_title("Öğrenme eğrisi"); ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
ax.plot(np.arange(1, CFG["epochs"] + 1), hist["val_acc"], "o-"); ax.set_ylim(0, 1)
ax.set_xlabel("epoch"); ax.set_ylabel("val doğruluk"); ax.set_title("Doğrulama doğruluğu"); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
mid = (edges[:-1] + edges[1:]) / 2
ax.bar(mid, snr_acc, width=4.2); ax.axhline(1 / len(CLASSES), color="k", ls=":"); ax.set_ylim(0, 1)
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("test doğruluk"); ax.set_title("SNR'a göre doğruluk (kesikli: rastgele tahmin)")
ax.grid(alpha=0.3, axis="y")

ax = fig.add_subplot(gs[1, 1])
ax.bar(CLASSES, [per_class[c] for c in CLASSES]); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=45)
ax.set_title("Sınıfa göre doğruluk (tüm SNR)"); ax.grid(alpha=0.3, axis="y")

ax = fig.add_subplot(gs[0, 2])
ax.imshow(conf(snr >= 10), vmin=0, vmax=1, cmap="Blues")
ax.set_title("Karışıklık matrisi, SNR ≥ 10 dB")
ax2 = fig.add_subplot(gs[1, 2])
ax2.imshow(conf(snr < 0), vmin=0, vmax=1, cmap="Blues")
ax2.set_title("Karışıklık matrisi, SNR < 0 dB")
for a in (ax, ax2):
    a.set_xticks(range(len(CLASSES))); a.set_xticklabels(CLASSES, rotation=60, fontsize=8)
    a.set_yticks(range(len(CLASSES))); a.set_yticklabels(CLASSES, fontsize=8)
    a.set_xlabel("tahmin"); a.set_ylabel("gerçek")
fig.suptitle(f"{CFG['tag']} | kanal bozulması={CFG['channel']} | seed={CFG['seed']} | {CFG['n_train'] * len(CLASSES):,} örnek, "
             f"{n_params:,} parametre, {CFG['epochs']} epoch, {device} | test doğruluk {ta:.3f}", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / f"{CFG['tag']}.png", dpi=100)

json.dump(dict(config=CFG, n_params=n_params, device=str(device), history=hist, test_acc=ta,
               snr_bins=[int(e) for e in edges], snr_acc=snr_acc, per_class=per_class,
               seconds=time.time() - t0), open(OUT / f"{CFG['tag']}_metrics.json", "w"), indent=1)
torch.save(model.state_dict(), OUT / f"{CFG['tag']}.pt")
print("yazıldı:", OUT / f"{CFG['tag']}.png")
