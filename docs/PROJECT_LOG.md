# Proje günlüğü

En yeni giriş en altta. Her oturum sonunda: **Ne yaptık / Ne öğrendik / Açık kalanlar**.

## 2026-10-10 — Oturum 1: Başlangıç ve temeller
**Ne yaptık**
- Claude masaüstü uygulamasının özelliklerini ve projeye uygun olanları tartıştık (CLAUDE.md, skill, plan modu, terminal/diff paneli, `/code-review`, `/loop`).
- Hedef ve kısıtlar netleşti: sentetik veriyle başla, gerçek veriyle test et; PyTorch; yerelde RX 5600 XT / Ryzen 5 3500X, ağır işler için RTX A2000 veya Colab.
- Doküman iskeleti kuruldu.
- Makine kontrolü: Python 3.14.3, Git kurulu, RAM 16 GB.

**Ne öğrendik**
- RX 5600 XT, Windows'ta PyTorch ROCm desteğinde yok; bu yüzden ilk eğitimler CPU'da (bkz. D-002).

**Açık kalanlar**
- Ortam kurulumu, Python sürümü doğrulaması (D-003)
- Claude'un "bakalım" dediği fikir: kural tabanlı şema ile derin öğrenmenin ilişkisi

### 2026-10-10 (devam) — Fikir, D-005 ve ortam kurulumu
**Ne yaptık**
- Şemanın `data.js`/`sigs.js` içeriğini inceledik: 62 sinyal, 10 aile, 21 özellik, karar ağacı, tohumlu JS üreticiler.
- D-005 kararı: şema = sınıf listesi + veri fabrikası + kıyas ölçütü; 3 kademeli kıyas (kural ağacı / özellik+RF / ham IQ+DL); üreticiyi Python'a port et; 8–10 sinyalle başla.
- `.venv` oluşturuldu (Python 3.14.3). Kurulanlar: torch 2.14.1+cpu, numpy, scipy, scikit-learn, matplotlib. `requirements.txt` ve `.gitignore` yazıldı; `src/ configs/ experiments/ notebooks/ data/` klasörleri açıldı.
- Duman testi: tensör oluşturma ve 20 adımlık mini CNN eğitimi çalıştı (~0,15 sn/adım, CPU).

**Ne öğrendik**
- Python 3.14 PyTorch ile sorunsuz çalıştı; D-003 doğrulandı.
- Küçük modeller için CPU yeterli; GPU ihtiyacı daha büyük modeller ve veri setleriyle ortaya çıkacak.

**Açık kalanlar**
- `git init` (kullanıcı onayı bekliyor)
- Sinyal sınıf listesini `docs/SIGNAL_CLASSES.md`'ye aktarma
- Aşama 1: IQ'yu öğrenme oturumu, sonra Python üretici portu

### 2026-10-10 (devam 2) — Dersler 1–3: IQ, SNR, veri seti
**Ne yaptık**
- `git init`, 3 commit. Uzak repo henüz açılmadı (kullanıcı kararı bekliyor; `gh` oturumu mevcut).
- Ders 1 (`src/iq_intro.py`): ton, AM, QPSK; IQ düzlemi / zarf / spektrum.
- Ders 2 (`src/snr_intro.py`): SNR düşerken takımyıldız ve spektrum; QPSK SER ölçümü kuramla örtüştü (10 örnek ortalaması ≈ +10 dB işleme kazancı).
- Ders 3: `src/signals.py` (10 sınıf üretici), `src/dataset.py` (sızıntısız split), `src/dataset_check.py` (kontroller + galeri). D-006 yazıldı.
- Kontroller: şekiller `(M,2,1024)`, güç = 1,0000, sınıf dengeli, train/val/test arası ortak örnek = 0, aynı tohum = aynı veri. 7000 örnek ~7 sn'de üretiliyor.

**Ne öğrendik**
- Takımyıldız 5 dB civarında karışırken spektrumda "dijital sinyal var" bilgisi 0 dB'ye kadar kalıyor.
- OFDM, 0 dB'de gürültüden spektrumda neredeyse ayırt edilemiyor (güç geniş banda yayılıyor); düşük SNR'da en zor sınıf olması bekleniyor.
- BPSK/QPSK/QAM IQ düzleminde net noktalar yerine halka gibi görünüyor: aşırı örnekleme + RRC süzgeci; noktalar yalnızca sembol anlarında oluşur.

**Açık kalanlar**
- `signals.py` ↔ `sigs.js` spektrum doğrulaması (D-005 maddesi, D-006 sapma notu)
- Kanal bozulmaları
- Ders 4: ilk model (küçük CNN), eğitim döngüsü, SNR'a göre doğruluk eğrisi
- Uzak repo kararı

### 2026-10-10 (devam 3) — Ders 4: ilk model
**Ne yaptık**
- `src/model.py` (4 bloklu 1D CNN, 104 bin parametre), `src/train.py` (eğitim, değerlendirme, grafik). 15 epoch, batch 128, Adam + kosinüs LR, 15000 train örneği, CPU, ~18,5 dk. Çıktılar: `experiments/04_run1.*`.
- **Sonuç:** test doğruluk 0,898 (val 0,903). SNR'a göre: ≥0 dB ~%99+, −5…0 dB %88, −10…−5 dB %54.
- Karışıklık (SNR<0): BPSK/QPSK/16-QAM birbirine, OFDM→noise, FM→ton/noise/FSK.
- İkinci koşu birincinin kayıplarını üçüncü ondalığa kadar tekrarladı: tohumlama çalışıyor.

**Ne öğrendik**
- Tek bir doğruluk sayısı yanıltıcı; model yüksek SNR'da kusursuz, tüm zayıflık düşük SNR'da.
- 10 sınıf yüksek SNR'da çok kolay; modelleri ayırt etmek için zorluk gerekecek.
- Süreç: uzun işleri log dosyasına yazdır, `tail` ardında çalıştırma; çalışan süreci ilerleme görünmüyor diye öldürme.

**Açık kalanlar**
- Kademe (b): özellik + random forest (ders 5 önerisi)
- CNN iyileştirme (uzun eğitim / büyük model) — denenmedi
- Kanal bozulmaları, `sigs.js` doğrulaması, uzak repo kararı (şimdilik yerel)
