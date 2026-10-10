# Kararlar

Format: **Karar → Gerekçe → Ne zaman yeniden bakılır**

## D-001 — Çatı: PyTorch (2026-10-10)
- **Karar:** Model ve eğitim PyTorch ile yazılır.
- **Gerekçe:** Entegre edileceği Saykan SDR yazılımı Python tabanlı; PyTorch araştırma/öğrenme için de standart.
- **Yeniden bak:** Dağıtım aşamasında (ONNX'e aktarma gerekirse).

## D-002 — Hesaplama stratejisi (2026-10-10) — **ÖN KARAR, doğrulanacak**
- **Durum:** RX 5600 XT (RDNA1 / gfx1010) Windows'ta PyTorch'un resmî ROCm desteği kapsamında değil; DirectML yolu var ama bakım modunda ve kırılgan.
- **Karar (geçici):**
  1. İlk modeller **CPU'da** eğitilir. Başlangıç modelleri küçük (128–1024 örnekli IQ girdisi); 6 çekirdekte makul sürede biter.
  2. GPU gereken büyük eğitimler **RTX A2000 makinesi** veya **Colab** üzerinde yapılır. Kod aynı kalmalı: cihaz `torch.device` ile seçilir, koda "cuda" sabitlenmez.
  3. RX 5600 XT için DirectML, **ilk model çalıştıktan sonra** zaman kutulu bir deney olarak denenir (başarısızsa kaybı küçük).
- **Gerekçe:** Önce çalışan bir hat, sonra hız. Donanım sorunu öğrenmeyi bloklamasın.
- **Ölçüm (2026-10-10):** küçük 1D CNN (2→32→64 kanal), batch 256 × 1024 IQ örneği: ~0,15 sn/adım CPU'da (6 iş parçacığı). İlk modeller için CPU yeterli.
- **Yeniden bak:** CPU eğitim süresi bir deney için ~1 saati aşarsa.

## D-003 — Python ortamı (2026-10-10) — **ÖN KARAR**
- Sistem Python'u 3.14.3 çok yeni; PyTorch ve bilimsel kütüphaneler için hazır paketler 3.12/3.13'te daha garanti.
- İlk kurulumda 3.14 ile venv denenir; kurulum sorun çıkarırsa 3.12 yüklenir. Sonuç buraya yazılır.
- **Sonuç (2026-10-10): ONAYLANDI.** Python 3.14.3 venv'inde (`.venv`) torch 2.14.1+cpu, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2 sorunsuz kuruldu ve çalıştı. 3.12'ye geçmeye gerek yok.

## D-004 — Veri stratejisi (2026-10-10)
- Sentetik veriyle başla (kontrollü SNR, kanal bozulmaları, etiketler kesin), sonra gerçek kayıtlarla test et.
- Mevcut Artifact şemasındaki sinyal tanımları ve özellik sözlüğü, sentetik üreticinin başlangıç spesifikasyonu olarak kullanılacak.
- **Risk:** Sentetik→gerçek geçişte performans düşer (domain gap). Gerçek kayıtlara erken, küçük de olsa bakmak planlanmalı.

## D-005 — Şema merkezli, kademeli kıyaslı yaklaşım (2026-10-10)
- **Karar:**
  1. Sinyal şeması (62 sinyal / 10 aile / 21 özellik / karar ağacı) projenin **sınıf listesi, veri fabrikası ve kıyas ölçütü** olarak kullanılır.
  2. **Üç kademeli kıyas** aynı veride, SNR'a göre doğruluk eğrisiyle yapılır: (a) kural tabanlı ağaç, (b) 21 özellik + klasik ML (random forest), (c) ham IQ + derin ağ.
  3. `sigs.js` üreticileri **Python/NumPy'a port edilir**; iki sürümün aynı spektrumu verdiği doğrulanır.
  4. Başlangıç alt kümesi 8–10 sinyal: gürültü, saf ton, AM, FM, BPSK, QPSK, 16-QAM, FSK, OFDM (+ opsiyonel radar darbesi). Çalışan hat sonrası aile aile genişletilir; sınıflandırma hiyerarşik (aile → sinyal).
- **Gerekçe:** Elle tasarlanmış uzman bilgisiyle öğrenilen çözümü ölçerek kıyaslamak hem öğretici hem savunulabilir; hazır etiket ve üreticiler işi azaltır.
- **Riskler:** Döngüsellik (aynı üreticiyle test), kalibre olmayan eşikler, idealize JS üretici, 62 sınıfın bir anda zorluğu. Gerçek kayda erken bakış D-004 ile zorunlu.
- **Yeniden bak:** Kademe (c), kademe (b)'den anlamlı fark üretmezse veya port doğrulanamazsa.
