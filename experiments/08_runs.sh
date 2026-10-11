#!/bin/bash
# Ders 8: 14 sınıf, bozuk veri (channel=1.0). Bütçe: 1500 ve 6000 örnek/sınıf, width 32, 15 epoch.
cd "C:/Atalay-Projects/DeepLearningProjects"
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run() { tag=$1; shift; echo "== $tag $(date +%T)"; $PY -u src/train.py --classes 14 --channel 1.0 --tag "$tag" "$@" > "experiments/$tag.log" 2>&1; tail -9 "experiments/$tag.log" | head -2; }
run 08_c14_n1500 --n_train 1500
run 08_c14_n6000 --n_train 6000
echo "== bitti $(date +%T)"
