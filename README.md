🏭 ScrappiX: Veri Odaklı Kestirimci Kalite ve Kalıp Ömür Analiz Sistemi
ScrappiX, polimer (enjeksiyon) ve elastomer (kompresyon) kalıplama süreçlerinde, makinelere pahalı sensörler takmak yerine doğrudan ürünün boyutsal ölçüm verilerini (Metroloji) kullanarak kalıbın Kalan Faydalı Ömrünü (RUL) tahmin eden bir Kestirimci Kalite (Predictive Quality) yazılımıdır.

📌 Projenin Amacı ve Çözdüğü Problem
Seri üretimde kalite kontroller genellikle periyodik (örneğin 4 saatte bir) yapılır. Bu iki kontrol arasında kalan süreye "Kör Nokta" (Blind Spot) denir. Eğer kalıp bu süre içinde aşınır ve tolerans dışına çıkarsa, fark edilene kadar binlerce hatalı (hurda) parça üretilir.

ScrappiX, IATF 16949 İstatistiksel Proses Kontrol (SPC) yöntemleri ile Makine Öğrenmesi (Regresyon) algoritmalarını birleştirerek; hatayı oluşmadan önce tahmin eder ve olası Kalitesizlik Maliyetini (COPQ) canlı olarak hesaplayarak önler.

🚀 Temel Özellikler ve Çalışma Mantığı
🔍 Sensörsüz RUL Tahmini: Titreşim veya sıcaklık sensörlerine ihtiyaç duymaz. Doğrudan ürünün iç çap, dış çap ve yükseklik ölçümlerindeki milimetrik trendleri (Linear Regression) analiz ederek kalıbın kaç baskı sonra bozulacağını (RUL) bulur.

🔗 En Zayıf Halka Teorisi (Weakest Link): Sistemin ömrünü, incelenen ölçüler (iç çap, dış çap, vb.) arasından en hızlı bozulan ölçü belirler. Yazılım, global alarmı bu "en zayıf halkaya" göre verir.

💰 Finansal Tasarruf Modeli: Sistem, kalıbın kalan ömrü ile kalite kontrol periyodu arasındaki ilişkiyi hesaplar. Eğer hata, kaliteci gelmeden önce oluşacaksa, o aradaki üretimi "Riskli" kabul eder ve kurtarılan potansiyel hurda maliyetini TL/USD/EUR cinsinden ekrana yansıtır.

📊 Detaylı MSA ve IATF 16949 Standartları: Anlık olarak Cp, Cpk, Pp, Ppk, Çarpıklık (Skewness) ve Basıklık (Kurtosis) değerlerini hesaplar.

🛠️ Kullanılan Teknolojiler
Veri Analitiği ve İşleme: Pandas, NumPy, SciPy

Makine Öğrenmesi: Scikit-learn (Linear Regression, Isolation Forest)

Veri Görselleştirme: Plotly (İnteraktif Trend ve Histogram grafikleri)

Web Arayüzü: Streamlit

📂 Proje Yapısı
app.py: Sistemin son kullanıcıya hitap eden, etkileşimli Streamlit web arayüzü ve entegre karar destek mekanizması.

main.py: Algoritmaların test edildiği, anomalilerin (Isolation Forest) tespit edildiği terminal tabanlı çekirdek analiz motoru.

requirements.txt: Gerekli Python kütüphanelerinin listesi.

⚙️ Kurulum ve Kullanım
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

Depoyu klonlayın:

Bash

git clone https://github.com/kullaniciadiniz/scrappix.git
cd scrappix
Gerekli kütüphaneleri yükleyin:

Bash

pip install -r requirements.txt
Streamlit uygulamasını başlatın:

Bash

streamlit run app.py
Veri Yükleme:
Arayüz açıldığında, sol taraftaki menüden tolerans ve maliyet ayarlarınızı yapın. Ardından ilk 4 sütunu [Parça No, İç Çap, Dış Çap, Yükseklik] olan herhangi bir .csv veya .xlsx dosyasını sisteme yükleyin veya "Simülasyon Verisi Oluştur" butonuna basarak sistemi test edin.

🎓 Akademik Altyapı
Bu yazılım, bir yüksek lisans tezinin parçası olarak geliştirilmiştir. Makine verisi yerine "Ürün Metroloji" verisi kullanarak kalıp ömrü tahmin etme yaklaşımı, literatürdeki güncel çalışmalarla (örn: Böttjer et al., 2022) desteklenmektedir.
