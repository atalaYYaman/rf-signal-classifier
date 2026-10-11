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

## D-006 — Veri konvansiyonları (2026-10-10)
- **Karar:**
  1. **Blok uzunluğu N = 1024** karmaşık örnek; model girdisi `(2, 1024)` tensör (kanal 0 = I, kanal 1 = Q).
  2. **SNR = örnek başına** sinyal gücü / gürültü gücü (şemadaki bant içi SNR, F1, değildir; bant genişliği oranı kadar fark vardır). Eğitimde **uniform [−10, +20] dB**.
  3. Sinyal birim güçle üretilir; gürültü eklenir; sonra blok **yeniden birim güce** normalize edilir. Mutlak seviye bir ipucu olamaz. "noise" sınıfı yalnızca birim güçlü gürültüdür.
  4. **Sızıntısız ayrım:** her örnek `default_rng([BASE_SEED, split, sınıf, indeks])` ile üretilir; split numarası tohumun parçasıdır. Aynı veri her çalıştırmada aynıdır.
  5. Başlangıç 10 sınıf: noise, tone, am, fm, bpsk, qpsk, qam16, fsk2, ofdm, radar.
- **Sapma notu (D-005'ten):** `src/signals.py` `sigs.js`'in birebir portu **değildir**; şemanın sinyal tanımlarına bakılarak yeniden yazıldı (RRC süzgeçli doğrusal modülasyonlar, rastgele parametreler). Python ↔ JS spektrum doğrulaması **hâlâ açık iş** (ROADMAP).
- **Bilinen sınırlar:** Kanal bozulması yok (çok yollu yayılma, frekans kayması, IQ dengesizliği); parametre aralıkları (sembol hızı, sapma vb.) varsayımdır, şemadaki gerçek aralıklarla karşılaştırılmadı.

### D-005 — Güncelleme (2026-10-11): kademe (b) vs (c) ilk sonuç
- Aynı veri, aynı test seti (3000 örnek, 10 sınıf, SNR −10…+20 dB): **RF + 20 özellik %88,8** (~30 sn) · **CNN %89,8** (~19 dk, 104 bin parametre).
- Fark ~1 puan ve tek koşu; yalnızca 0…+10 dB aralığında CNN'in önde olması muhtemel (RF %96,5/%98,8 vs CNN %99,0/%99,8). Diğer aralıklarda fark ölçüm belirsizliği içinde (her aralıkta n≈500, ±2 puan).
- **Yorum sınırı:** Özellikleri aynı üreticinin sınıflarını bilerek seçtik (döngüsellik, D-005 riski). Görev yüksek SNR'da kolay. Bu sonuç "derin öğrenme gereksiz" **demek değildir**; "bu kolay veride kazancı küçük" demektir.
- **Sonraki adım:** veriyi zorlaştırıp (kanal bozulmaları, daha çok sınıf) iki yaklaşımı yeniden kıyasla. → Bozulmalar ders 6'da yapıldı (aşağıda D-007 ve D-005 ders 6 güncellemesi). D-005'in "yeniden bak" koşulu (c ile b arasında anlamlı fark yoksa) şu an kısmen tetikleniyor; karar zorlaştırılmış veriden sonra verilecek.

## D-007 — Kanal bozulma modeli (2026-10-11)
- **Karar:** `signals.apply_channel(x, rng, severity)` sinyale, gürültüden **önce**: çok yollu yayılma (0–3 yansıma, 3–12 dB zayıf, 1–5 örnek gecikme) → örnekleme hızı ofseti (±%2) + kesirli gecikme → taşıyıcı frekans kayması (±0,03) → faz gürültüsü (rastgele yürüyüş, σ ≤ 0,01 rad/örnek) → IQ dengesizliği (±1 dB, ±5°) uygular. `severity` 0..1 ölçeği; `channel=0` eski veriyi **birebir** üretir (regresyon parmak izi `experiments/.clean_fingerprint`, `src/channel_demo.py`).
- **Deney protokolü:** her model hem temiz hem bozuk veriyle eğitilir, ikisinde de test edilir (2×2). Test seti aynı tohum/indeksle üretilir.
- **Sınır:** Bozulma aralıkları varsayımdır; gerçek kayıtlara kalibre edilmedi. Sınıfların taşıyıcıları zaten rastgele olduğundan frekans kayması etkisi sınırlı.

### D-005 — Güncelleme 2 (2026-10-11): bozuk kanalda kademe (b) vs (c)
| Eğitim → Test | RF | CNN |
|---|---|---|
| temiz → temiz | 0,888 | 0,898 |
| temiz → bozuk | 0,730 | 0,757 |
| bozuk → temiz | 0,878 | 0,861 |
| bozuk → bozuk | **0,858** | 0,834 |
- Beklenti "CNN bozulmaya daha dayanıklıdır" **desteklenmedi**: bozuk veride RF ~2,4 puan önde (CNN düşük SNR'da da geride: −10…−5 dB 0,45 vs 0,55).
- İki model de temiz eğitimle bozuk veride ~15 puan kaybediyor; bozulmayı eğitim setine katmak kaybı büyük ölçüde geri alıyor (**veri çeşitlendirme** ucuz ve etkili).
- **Sınırlar:** tek tohum; CNN ayarlanmadı (15 epoch, 104 bin parametre, 15000 örnek), RF'nin özellikleri bu sınıflar bilinerek seçildi. CNN'in yetersiz kapasite/eğitimden mi yoksa yapısal olarak mı geride kaldığı **bilinmiyor**; denenecek (daha çok veri/epoch/kapasite).

### D-005 — Güncelleme 3 (2026-10-11): CNN'e adil şans (bozuk veri, aynı test seti)
| Koşu | Test doğruluk | Not |
|---|---|---|
| CNN baseline (3 tohum) | 0,834 / 0,839 / 0,842 | ort 0,838 ± 0,003 (gürültü tabanı) |
| CNN 40 epoch | 0,843 | train loss 0,115, val loss 0,466: ezberleme; tohum aralığında |
| CNN kapasite ×4 (409 bin parametre) | 0,863 | +2,5 puan |
| CNN veri ×4 (60.000 örnek) | **0,877** | +3,9 puan |
| RF (15k örnek, 3 tohum) | 0,860 ± 0,003 | |
| RF veri ×4 (3 tohum) | 0,875 ± 0,000 | +1,5 puan |
- **Sonuç:** CNN'in geride kalması büyük ölçüde **veri/kapasite eksikliğiydi**; 4× veriyle CNN 0,877, RF 0,875: **başa baş**. Ders 6'daki "RF önde" sonucu eğitim bütçesinin ürünüydü.
- Daha çok epoch işe yaramadı (ezberleme). Veri ve kapasite işe yaradı. CNN veri artışından RF'ten daha çok yararlandı (+3,9 vs +1,5); bu eğilim sürerse fark CNN lehine açılabilir, ama **birleşik (veri ×4 + kapasite ×4) koşu denenmedi**, ekstrapolasyon varsayımdır.
- **Sınırlar:** bu deneyde tek veri artışı basamağı (×4); test seti paylaşıldığı için ±0,65 puanlık örnekleme belirsizliği iki modelde de ortak; düşük SNR aralığında RF'in 4× veri sonucu SNR bazında hesaplanmadı. Süreler (CNN 4,5–17 dk) ölçüm tutarsızlığı nedeniyle güvenilmez.
