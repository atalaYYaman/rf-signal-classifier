#!/bin/bash
# Ders 7: CNN'e adil şans, hepsi bozuk veri (channel=1.0), sırayla, test seti aynı.
cd "C:/Atalay-Projects/DeepLearningProjects"
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run() { tag=$1; shift; echo "== $tag $(date +%T)"; $PY -u src/train.py --channel 1.0 --tag "$tag" "$@" > "experiments/$tag.log" 2>&1; tail -9 "experiments/$tag.log" | head -2; }
run 07_seed1   --seed 1
run 07_seed2   --seed 2
run 07_ep40    --epochs 40
run 07_data4x  --n_train 6000
run 07_w64     --width 64
echo "== bitti $(date +%T)"
