"""Ders 7: RF'ye de aynı 'adil şans': eğitim verisini artırınca bozuk veride doğruluk nasıl değişir?

Çalıştır:  .venv\\Scripts\\python src\\rf_scale.py
Çıktı:     experiments/07_rf_scale.json
Test seti, CNN deneyleriyle aynıdır (channel=1.0, sınıf başına 300 örnek, SNR -10..+20 dB).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dataset import SignalDataset
from features import feature_matrix

OUT = Path(__file__).resolve().parent.parent / "experiments"
CH = 1.0


def feats(split, n):
    ds = SignalDataset(split, n, channel=CH)
    return feature_matrix((ds.x[:, 0] + 1j * ds.x[:, 1]).numpy()), ds.y.numpy()


Xte, yte = feats("test", 300)
res = {}
for n_per_class in (1500, 6000):
    Xtr, ytr = feats("train", n_per_class)
    accs = []
    for seed in (0, 1, 2):
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        accs.append(float((rf.predict(Xte) == yte).mean()))
    res[n_per_class] = dict(accs=accs, mean=float(np.mean(accs)), std=float(np.std(accs)))
    print(f"RF n_train/sınıf={n_per_class}: doğruluk {accs} ort {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)
json.dump(res, open(OUT / "07_rf_scale.json", "w"), indent=1)
