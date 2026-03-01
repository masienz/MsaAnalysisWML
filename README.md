# 🏭 ScrappiX: Veri Odaklı Kestirimci Kalite ve Kalıp Ömür Analiz Sistemi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![IATF](https://img.shields.io/badge/Standard-IATF%2016949-success)

**ScrappiX**, polimer (enjeksiyon) ve elastomer (kompresyon) kalıplama süreçlerinde, makinelere pahalı sensörler takmak yerine **doğrudan ürünün boyutsal ölçüm verilerini (Metroloji)** kullanarak kalıbın Kalan Faydalı Ömrünü (RUL) tahmin eden bir Kestirimci Kalite (Predictive Quality) yazılımıdır.

## 📌 Projenin Amacı ve Çözdüğü Problem

Seri üretimde kalite kontroller genellikle periyodik (örneğin 4 saatte bir) yapılır. Bu iki kontrol arasında kalan süreye **"Kör Nokta" (Blind Spot)** denir. Eğer kalıp bu süre içinde aşınır ve tolerans dışına çıkarsa, fark edilene kadar binlerce hatalı (hurda) parça üretilir.

ScrappiX, IATF 16949 İstatistiksel Proses Kontrol (SPC) yöntemleri ile Makine Öğrenmesi (Regresyon) algoritmalarını birleştirerek; hatayı oluşmadan önce tahmin eder ve olası **Kalitesizlik Maliyetini (COPQ)** canlı olarak hesaplayarak önler.

## 🚀 Temel Özellikler ve Çalışma Mantığı

* **🔍 Sensörsüz RUL Tahmini:** Titreşim veya sıcaklık sensörlerine ihtiyaç duymaz. Doğrudan ürünün iç çap, dış çap ve yükseklik ölçümlerindeki milimetrik trendleri analiz ederek kalıbın kaç baskı sonra bozulacağını (RUL) bulur.
* **🔗 En Zayıf Halka Teorisi (Weakest Link):** Sistemin ömrünü, incelenen ölçüler (iç çap, dış çap, vb.) arasından en hızlı bozulan ölçü belirler. Yazılım, global alarmı bu "en zayıf halkaya" göre verir.
* **💰 Finansal Tasarruf Modeli:** Sistem, kalıbın kalan ömrü ile kalite kontrol periyodu arasındaki ilişkiyi hesaplar. Eğer hata, kaliteci gelmeden önce oluşacaksa, o aradaki üretimi "Riskli" kabul eder ve kurtarılan potansiyel hurda maliyetini TL/USD/EUR cinsinden ekrana yansıtır.
* **📊 Detaylı MSA ve IATF 16949 Standartları:** Anlık olarak `Cp`, `Cpk`, `Pp`, `Ppk`, `Çarpıklık` ve `Basıklık` değerlerini hesaplar.

## 🛠️ Kullanılan Teknolojiler

* **Veri Analitiği:** `Pandas`, `NumPy`, `SciPy`
* **Makine Öğrenmesi:** `Scikit-learn` (Linear Regression, Isolation Forest)
* **Veri Görselleştirme:** `Plotly` (İnteraktif Grafik Motoru)
* **Web Arayüzü:** `Streamlit`

## 📂 Proje Yapısı

* `app.py`: Sistemin son kullanıcıya hitap eden, etkileşimli Streamlit web arayüzü ve entegre karar destek mekanizması.
* `main.py`: Algoritmaların test edildiği, anomalilerin tespit edildiği terminal tabanlı çekirdek analiz motoru.
* `requirements.txt`: Gerekli Python kütüphanelerinin listesi.

## ⚙️ Kurulum ve Kullanım

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla terminalinize (komut satırına) yapıştırın:

**1. Depoyu klonlayın ve klasöre girin:**
`git clone https://github.com/kullaniciadiniz/scrappix.git`
`cd scrappix`

**2. Gerekli kütüphaneleri yükleyin:**
`pip install -r requirements.txt`

**3. Streamlit uygulamasını başlatın:**
`streamlit run app.py`

**4. Veri Yükleme ve Test:**
Arayüz açıldığında, sol taraftaki menüden tolerans ve maliyet ayarlarınızı yapın. Ardından ilk 4 sütunu `[Parça No, İç Çap, Dış Çap, Yükseklik]` olan herhangi bir `.csv` veya `.xlsx` dosyasını sisteme yükleyin veya **"Simülasyon Verisi Oluştur"** butonuna basarak sistemi anında test edin.

## 🎓 Akademik Altyapı
Bu yazılım, bir yüksek lisans tezinin parçası olarak geliştirilmiştir. Makine verisi yerine "Ürün Metroloji" verisi kullanarak kalıp ömrü tahmin etme yaklaşımı, literatürdeki güncel akademik çalışmalarla desteklenmektedir.
