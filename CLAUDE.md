# RF Sinyal Sınıflandırma — Derin Öğrenme Projesi

## Amaç
Karmaşık IQ örneklerinden belirli radyo frekans (RF) sinyal/modülasyon tiplerini ayırt eden bir PyTorch modeli geliştirmek.
Önce sentetik veriyle eğitilir, sonra gerçek kayıtlarla test edilir. Model en sonda Python tabanlı bir SDR yazılımına (Saykan SDR) entegre edilecek.

## Kullanıcı
- Derin öğrenmede yeni, temel seviyede eğitim almış. **Jargonu ilk kullanımda açıkla**, yeni terimi `docs/GLOSSARY.md`'ye ekle.
- Amaç hem ürün hem **öğrenme**: önce "ne ve neden", sonra kod. Küçük adımlarla, çalıştırarak ilerle.
- Dil: Türkçe. Kod, değişken ve dosya adları İngilizce.

## Önce oku
1. `docs/ROADMAP.md` — neredeyiz, sırada ne var
2. `docs/DECISIONS.md` — alınmış kararlar (yeniden tartışma)
3. `docs/PROJECT_LOG.md` — son birkaç giriş
4. Sinyal şeması (kaynak bilgi): https://claude.ai/artifact/12wMuY7hny8NuetqedHWJE (Artifact, "Sinyal Sınıflandırma Şeması")

## Çalışma kuralları
- Her oturum sonunda `docs/PROJECT_LOG.md`'ye giriş ekle (skill: `oturum-kapat`).
- Önemli bir seçim yapıldığında `docs/DECISIONS.md`'ye gerekçesiyle yaz.
- Yeni kavram çıkınca `docs/GLOSSARY.md`'ye bir cümlelik açıklamayla ekle.
- Deneyler tekrarlanabilir olmalı: rastgelelik tohumu (seed), config dosyası, sonuç kaydı.
- Train / validation / test ayrımı **sinyal örneği düzeyinde** ve sızıntısız yapılmalı (data leakage kontrolü).
- Bir şeyi çalıştırmadan "çalışıyor" deme; çıktıyı göster.

## Donanım ve ortam
- Ana makine: Windows 11, Ryzen 5 3500X (6 çekirdek), 16 GB RAM, AMD RX 5600 XT (6 GB). Python 3.14.3 kurulu.
- Büyük işler için zaman zaman RTX A2000'li makineler; ya da Google Colab. Ayrıntı: `docs/DECISIONS.md` D-002.
- Sanal ortam (venv) kullan; sistem Python'una paket kurma.

## Klasör düzeni (planlanan)
```
data/        ham ve üretilmiş veri (git'e girmez)
src/         kütüphane kodu (veri üretimi, model, eğitim)
configs/     deney ayarları
experiments/ sonuçlar, grafikler
notebooks/   keşif amaçlı defterler
docs/        proje dokümanları
```
