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
