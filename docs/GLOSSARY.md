# Sözlük

Kural: yeni terim çıkınca buraya bir cümlelik açıklamayla eklenir. Alfabetik değil, konuya göre.

## Sinyal tarafı
- **RF (radyo frekansı):** Havada yayılan elektromanyetik dalgalar; radyo, Wi-Fi, telsiz hep burada.
- **Modülasyon:** Bilgiyi bir taşıyıcı dalganın genliğini, frekansını veya fazını değiştirerek üzerine "yazmak". Örnekler: AM, FM, BPSK, QPSK, QAM.
- **IQ (I/Q) örnekleri:** Sinyali iki sayıyla gösterme biçimi: I (gerçek kısım) ve Q (sanal kısım). Bir sinyal = çok sayıda (I, Q) çifti. Faz ve genlik bilgisini birlikte taşır.
- **Örnekleme hızı (fs):** Saniyede kaç IQ örneği alındığı. Büyüdükçe daha geniş bant görürsünüz.
- **Bant genişliği:** Sinyalin frekans ekseninde kapladığı genişlik.
- **SNR (sinyal/gürültü oranı, dB):** Sinyal gücünün gürültüye oranı. 0 dB'de sinyal ile gürültü eşit güçtedir; düştükçe sınıflandırmak zorlaşır.
- **İşleme kazancı:** Birçok örneği birleştirerek (ortalama, FFT) SNR'ı yapay olarak yükseltmek; N örnek ≈ +10·log10(N) dB. Ton gibi dar sinyaller bundan en çok yararlanır.
- **Sembol hata oranı (SER):** Alıcının bir sembolü yanlış kestiği oran; SNR düştükçe artar. QPSK'da rastgele tahminde %75'tir.
- **Spektrum / waterfall:** Sinyalin frekansa göre gücü / bunun zamanla akan görüntüsü.
- **Blok:** Modele tek seferde verilen sabit uzunluktaki IQ parçası (bizde 1024 örnek).
- **Normalizasyon (birim güç):** Bloğu ortalama gücü 1 olacak şekilde ölçeklemek; model mutlak sinyal seviyesine bakıp hile yapamaz.
- **Pulse shaping / RRC süzgeci:** Sembol geçişlerini yumuşatıp spektrumu dar tutan süzgeç; gerçek vericilerin yaptığı şey.
- **Taşıyıcı ofseti:** Sinyalin merkez frekansının beklenenden kayması; modelin buna dayanıklı olması istenir.
- **Takımyıldız (constellation):** IQ düzleminde sembollerin kümelendiği noktalar; modülasyonu ayırt etmede güçlü ipucu.

## Derin öğrenme temelleri
- **Model:** Girdiyi (IQ) alıp çıktı (sınıf tahmini) üreten parametreli fonksiyon.
- **Parametre / ağırlık:** Modelin eğitimle öğrendiği sayılar.
- **Tensör:** PyTorch'un çok boyutlu sayı dizisi; NumPy dizisine benzer ama GPU'da çalışabilir.
- **Dataset / DataLoader (PyTorch):** Dataset örnekleri tutar ve verir; DataLoader bunları batch'ler hâlinde, karıştırarak modele servis eder.
- **Etiket (label):** Doğru cevap; örn. "QPSK".
- **Sınıflandırma:** Girdiyi önceden tanımlı sınıflardan birine atama problemi.
- **Kayıp (loss):** Modelin ne kadar yanıldığının sayısı; eğitim bunu küçültür.
- **Eğitim (training), epoch, batch:** Model veriyi parçalar (batch) hâlinde görür; tüm verinin bir turu = 1 epoch.
- **Optimizasyon (optimizer):** Kaybı azaltmak için ağırlıkları güncelleyen yöntem (örn. Adam).
- **Öğrenme oranı (learning rate):** Her güncellemenin adım büyüklüğü.
- **Train / validation / test:** Eğitim / ayar ve izleme / son kez dokunulan değerlendirme verisi. Test seti eğitimde veya ayarda kullanılmaz.
- **Aşırı öğrenme (overfitting):** Model eğitim verisini ezberler, yeni veride kötü çalışır.
- **Veri sızıntısı (data leakage):** Test bilgisinin eğitime karışması; sonuçlar olduğundan iyi görünür.
- **CNN (evrişimli ağ):** Yerel örüntüleri tarayan filtrelerle çalışan ağ; IQ dizisinde zamansal örüntüleri bulur.
- **Logit:** Modelin softmax'tan önceki ham sınıf skorları; en büyüğü tahmindir.
- **BatchNorm / Dropout:** Eğitimi kararlılaştıran ve ezberlemeyi azaltan katmanlar; ilki aktivasyonları normalize eder, ikincisi rastgele nöronları kapatır.
- **Öğrenme eğrisi:** Epoch'a göre train/val kaybı grafiği. İkisi birbirinden ayrılıyorsa ezberleme var demektir.
- **Karışıklık matrisi (confusion matrix):** Hangi sınıfın hangisiyle karıştığını gösteren tablo.
- **Taban çizgisi (baseline):** Kıyas için kullanılan basit model.
- **Domain gap:** Sentetik veri ile gerçek veri arasındaki fark yüzünden performansın düşmesi.

## Donanım ve ortam
- **CPU / GPU:** İşlemci / çok sayıda paralel çekirdekli grafik işlemci; derin öğrenmede GPU eğitimi çok hızlandırır.
- **CUDA:** NVIDIA GPU'ların yazılım katmanı (RTX A2000 bunu kullanır).
- **ROCm:** AMD'nin CUDA karşılığı; RX 5600 XT için Windows'ta resmî desteği yoktur.
- **DirectML:** Windows'un GPU hızlandırma katmanı; AMD kartlarda alternatif, ama bakım modunda.
- **venv:** Projeye özel izole Python ortamı.
- **Colab:** Google'ın tarayıcıdaki ücretsiz/ücretli GPU'lu not defteri ortamı.
