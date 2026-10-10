# Yol haritası

Durum: ☐ başlamadı · ◐ sürüyor · ☑ bitti

## Aşama 0 — Temel ◐
- ☑ Proje amacı, donanım ve kararlar (D-001…D-004)
- ☑ Doküman iskeleti (CLAUDE.md, docs/, skill)
- ☑ Ortam kurulumu: venv, PyTorch (CPU), "merhaba tensör" ve mini eğitim testi
- ☐ Sinyal şemasının (Artifact) sınıf listesini projeye aktarma → `docs/SIGNAL_CLASSES.md`

## Aşama 1 — Veri
- ☐ IQ nedir, sinyal nasıl temsil edilir (öğrenme oturumu)
- ☐ Sentetik üretici: `sigs.js`'ten Python'a port, 8–10 sinyalle başla (D-005), JS ile spektrum doğrulaması
- ☐ Gürültü (SNR) ve basit kanal bozulmaları
- ☐ Veri seti sınıfı, train/val/test ayrımı

## Aşama 2 — İlk model
- ☐ Kademe (a): şemanın kural tabanlı ağacı (eşik kalibrasyonu)
- ☐ Kademe (b): 21 özellik + random forest
- ☐ Kademe (c): küçük CNN, ham IQ
- ☐ Eğitim döngüsü, kayıp/doğruluk grafikleri
- ☐ Karışıklık matrisi, SNR'a göre doğruluk eğrisi

## Aşama 3 — Büyütme
- ☐ Daha fazla sınıf, daha zor koşullar
- ☐ Model alternatifleri (ResNet tipi, dikkat/Transformer vb.) ve kıyas
- ☐ Büyük eğitimler için A2000/Colab geçişi

## Aşama 4 — Gerçek veri ve entegrasyon
- ☐ Gerçek kayıtlarla test, domain gap analizi
- ☐ İnce ayar / veri artırma
- ☐ Saykan SDR'a entegrasyon (ONNX / TorchScript değerlendirmesi)
