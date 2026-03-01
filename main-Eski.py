import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import os
import joblib

# --- AYARLAR ---
DOSYA_YOLU = r'C:\Users\HP\Desktop\Proje\data\uretim_verisi.xlsx'

# --- MÜHENDİSLİK TOLERANSLARI (Manuel Girilmeli) ---
# Örneğin parça 100mm ise ve tolerans +/- 0.5 ise:
HEDEF_DEGER = 100.0  # Nominal Değer
UST_TOLERANS = 100.5 # USL (Upper Specification Limit)
ALT_TOLERANS = 99.5  # LSL (Lower Specification Limit)

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

print(f"--- IATF 16949 STANDARDINA GÖRE ANALİZ BAŞLATILIYOR ---")

# --------------------------------------------------------------------------------
# 1. VERİ YÜKLEME
# --------------------------------------------------------------------------------
def load_user_data():
    try:
        if DOSYA_YOLU.endswith('.xlsx'):
            df = pd.read_excel(DOSYA_YOLU, header=None)
        elif DOSYA_YOLU.endswith('.csv'):
            df = pd.read_csv(DOSYA_YOLU, header=None)
        else:
            raise ValueError("Format hatası.")
            
        df = df.iloc[:, :2] 
        df.columns = ['Parca_No', 'Olcum_Degeri']
        df['Olcum_Degeri'] = pd.to_numeric(df['Olcum_Degeri'], errors='coerce')
        df = df.dropna()
        return df
    except FileNotFoundError:
        print("Dosya bulunamadı!")
        exit()

# --------------------------------------------------------------------------------
# 2. IATF 16949 - SÜREÇ YETERLİLİK ANALİZİ (Cp, Cpk)
# --------------------------------------------------------------------------------
def calculate_capability(df):
    print("Süreç Yeterlilik (Capability) Analizi yapılıyor...")
    
    data = df['Olcum_Degeri']
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1) # Örneklem standart sapması
    
    # Cp: Sürecin potansiyeli (Merkezden sapmayı umursamaz, sadece yayılıma bakar)
    # Formül: (USL - LSL) / 6*sigma
    Cp = (UST_TOLERANS - ALT_TOLERANS) / (6 * std_dev)
    
    # Cpk: Sürecin gerçek yeteneği (Merkezden kaymayı da hesaba katar)
    # Formül: min( (USL - Mean) / 3*sigma , (Mean - LSL) / 3*sigma )
    cpu = (UST_TOLERANS - mean) / (3 * std_dev)
    cpl = (mean - ALT_TOLERANS) / (3 * std_dev)
    Cpk = min(cpu, cpl)
    
    print(f"\n--- KALİTE RAPORU ---")
    print(f"Hedef Değer: {HEDEF_DEGER}")
    print(f"Ortalama: {mean:.4f}")
    print(f"Standart Sapma (Sigma): {std_dev:.4f}")
    print(f"Cp Değeri: {Cp:.2f}")
    print(f"Cpk Değeri: {Cpk:.2f}")
    
    # Yorumlama
    if Cpk < 1.0:
        durum = "YETERSİZ (Süreç toleransları tutturamıyor, çok fazla hurda var)"
    elif 1.0 <= Cpk < 1.33:
        durum = "KABUL EDİLEBİLİR (Ancak iyileştirme gerekir)"
    elif Cpk >= 1.33:
        durum = "YETERLİ (Endüstriyel standart)"
    elif Cpk >= 1.67:
        durum = "MÜKEMMEL (Six Sigma seviyesine yakın)"
        
    print(f"SONUÇ: {durum}")
    print("---------------------\n")
    
    return Cp, Cpk, mean, std_dev

# --------------------------------------------------------------------------------
# 3. YAPAY ZEKA MODELLEME (Aynı kalıyor)
# --------------------------------------------------------------------------------
def ai_analysis(df):
    # Anomali
    iso = IsolationForest(contamination=0.03, random_state=42)
    df['Anomali'] = iso.fit_predict(df[['Olcum_Degeri']])
    joblib.dump(iso, 'models/anomali_modeli.pkl')
    
    # Tahmin
    X = df[['Parca_No']].values.reshape(-1, 1)
    y = df['Olcum_Degeri'].values
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'models/tahmin_modeli.pkl')
    
    son_parca = df['Parca_No'].max()
    gelecek_no = np.arange(son_parca + 1, son_parca + 101).reshape(-1, 1)
    gelecek_tahmin = model.predict(gelecek_no)
    
    return df, gelecek_no, gelecek_tahmin

# --------------------------------------------------------------------------------
# 4. GÖRSELLEŞTİRME (Histogram ve Çan Eğrisi Eklendi)
# --------------------------------------------------------------------------------
def create_dashboard(df, gelecek_no, gelecek_tahmin, mean, std_dev, Cpk):
    plt.figure(figsize=(16, 10))
    
    # GRAFİK 1: Trend Analizi (Sol Üst)
    plt.subplot(2, 2, 1)
    plt.plot(df['Parca_No'], df['Olcum_Degeri'], 'b-', alpha=0.3)
    plt.scatter(df['Parca_No'], df['Olcum_Degeri'], s=10, c='blue', alpha=0.5)
    
    # Hataları Kırmızı İşaretle
    hatalar = df[df['Anomali'] == -1]
    plt.scatter(hatalar['Parca_No'], hatalar['Olcum_Degeri'], s=30, c='red', label='Anomali')
    
    # Gelecek Tahmini
    plt.plot(gelecek_no, gelecek_tahmin, 'g--', linewidth=2, label='AI Tahmini')
    
    # Limitler
    plt.axhline(UST_TOLERANS, color='red', linestyle='-', linewidth=2, label='USL (Üst Limit)')
    plt.axhline(ALT_TOLERANS, color='red', linestyle='-', linewidth=2, label='LSL (Alt Limit)')
    
    plt.title('Zaman Serisi ve AI Tahmini')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # GRAFİK 2: Histogram ve Çan Eğrisi (Process Capability) (Sağ Üst)
    plt.subplot(2, 2, 2)
    
    # Histogram çiz
    plt.hist(df['Olcum_Degeri'], bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    
    # Çan Eğrisi (Normal Dağılım) ekle
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std_dev)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Dağılım')
    
    # Limit Çizgileri
    plt.axvline(UST_TOLERANS, color='red', linestyle='dashed', linewidth=2, label='USL')
    plt.axvline(ALT_TOLERANS, color='red', linestyle='dashed', linewidth=2, label='LSL')
    plt.axvline(HEDEF_DEGER, color='green', linestyle='-', linewidth=1, label='Hedef')
    
    plt.title(f'Süreç Yeterlilik Analizi (Cpk: {Cpk:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/Kalite_Raporu_MSA.png')
    print("Grafik kaydedildi: 'outputs/Kalite_Raporu_MSA.png'")

if __name__ == "__main__":
    df = load_user_data()
    # 1. Six Sigma Analizi
    Cp, Cpk, mean, std = calculate_capability(df)
    # 2. AI Analizi
    df, g_no, g_tahmin = ai_analysis(df)
    # 3. Raporlama
    create_dashboard(df, g_no, g_tahmin, mean, std, Cpk)
