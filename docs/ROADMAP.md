# Yol haritası

Durum: ☐ başlamadı · ◐ sürüyor · ☑ bitti

## Aşama 0 — Temel ◐
- ☑ Proje amacı, donanım ve kararlar (D-001…D-004)
- ☑ Doküman iskeleti (CLAUDE.md, docs/, skill)
- ☑ Ortam kurulumu: venv, PyTorch (CPU), "merhaba tensör" ve mini eğitim testi
- ☐ Sinyal şemasının (Artifact) sınıf listesini projeye aktarma → `docs/SIGNAL_CLASSES.md`

## Aşama 1 — Veri
- ☑ IQ nedir, sinyal nasıl temsil edilir (ders 1: `src/iq_intro.py`)
- ☑ Gürültü ve SNR (ders 2: `src/snr_intro.py`)
- ◐ Sentetik üretici: 10 sınıf Python'da yazıldı (`src/signals.py`); `sigs.js` ile spektrum doğrulaması açık
- ◐ Gürültü (SNR) ☑ · kanal bozulmaları (frekans kayması, çok yollu) ☐
- ☑ Veri seti sınıfı, sızıntısız train/val/test ayrımı (`src/dataset.py`, `src/dataset_check.py`)

## Aşama 2 — İlk model
- ☐ Kademe (a): şemanın kural tabanlı ağacı (eşik kalibrasyonu)
- ☑ Kademe (b): 20 özellik (şemadaki 12 F'ye karşılık) + random forest — test %88,8, CNN %89,8 (`src/features.py`, `src/rf_baseline.py`)
- ◐ Kademe (c): küçük CNN, ham IQ — ilk koşu bitti (test %89,8; ≥0 dB %99+, −10…−5 dB %54). İyileştirme açık
- ☑ Eğitim döngüsü, kayıp/doğruluk grafikleri (`src/train.py`)
- ☑ Karışıklık matrisi, SNR'a göre doğruluk eğrisi

## Aşama 3 — Büyütme
- ☐ **Önce:** veriyi zorlaştır (kanal bozulmaları: frekans kayması, çok yollu, IQ dengesizliği; daha çok sınıf: 8PSK, 64QAM, OOK, GFSK…) ve RF ile CNN'i yeniden kıyasla
- ☐ Daha fazla sınıf, daha zor koşullar
- ☐ Model alternatifleri (ResNet tipi, dikkat/Transformer vb.) ve kıyas
- ☐ Büyük eğitimler için A2000/Colab geçişi

## Aşama 4 — Gerçek veri ve entegrasyon
- ☐ Gerçek kayıtlarla test, domain gap analizi
- ☐ İnce ayar / veri artırma
- ☐ Saykan SDR'a entegrasyon (ONNX / TorchScript değerlendirmesi)
