import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="ScrappiX Pro", layout="wide")

# CSS: Kaymaları önleyen sadeleştirilmiş stil
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🏭 ScrappiX Pro: Kalite ve Ömür Analizi")

# --- ÜST BÖLÜM: DOSYA YÜKLEME VE BİLGİ ---
with st.container():
    col_upload, col_info = st.columns([1, 2])
    
    with col_upload:
        uploaded_file = st.file_uploader("📂 Dosyanı Buraya Yükle (Excel/CSV)", type=["xlsx", "csv"])
    
    with col_info:
        st.info("👈 **İpucu:** Sol menüden 'Baskı Başına Ürün' miktarını girmeyi unutmayın.")

# --- YAN MENÜ (AYARLAR) ---
st.sidebar.header("⚙️ Parametreler")
baski_basina_urun = st.sidebar.number_input("1 Baskıda Çıkan Ürün Miktarı", min_value=1, value=1, step=1)
nominal_deger = st.sidebar.number_input("Nominal (Hedef) Değer", value=100.0, step=0.1)
tolerans = st.sidebar.number_input("Tolerans (+/-)", value=0.5, step=0.01)

ust_limit = nominal_deger + tolerans
alt_limit = nominal_deger - tolerans

# --- FONKSİYONLAR ---
def load_data(file):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file, header=None)
    else:
        df = pd.read_csv(file, header=None)
    df = df.iloc[:, :2]
    df.columns = ['Parca_No', 'Olcum']
    df['Olcum'] = pd.to_numeric(df['Olcum'], errors='coerce')
    df = df.dropna()
    return df

def calculate_msa_stats(data, usl, lsl):
    mean = np.mean(data)
    std_dev_sample = np.std(data, ddof=1)
    std_dev_overall = np.std(data, ddof=0)
    
    Cp = (usl - lsl) / (6 * std_dev_sample)
    Cpk = min((usl - mean) / (3 * std_dev_sample), (mean - lsl) / (3 * std_dev_sample))
    Pp = (usl - lsl) / (6 * std_dev_overall)
    Ppk = min((usl - mean) / (3 * std_dev_overall), (mean - lsl) / (3 * std_dev_overall))
    
    if len(data) >= 3:
        stat, p_value = shapiro(data)
    else:
        p_value = 0
    
    return {
        "mean": mean, "Cpk": Cpk, "Ppk": Ppk, "Is_Normal": p_value > 0.05
    }

def calculate_rul(model, current_parca_no, ust_limit, alt_limit, urun_miktari):
    # Eğim ve Kesişim (y = ax + b)
    egim = model.coef_[0]
    kesim = model.intercept_
    
    kalan_parca = 0
    sinir_tipi = "Stabil" # Varsayılan değer
    
    # HATA DÜZELTİLDİ: Değişken isimleri eşitlendi
    if egim > 0.00001:
        hedef_parca = (ust_limit - kesim) / egim
        kalan_parca = hedef_parca - current_parca_no
        sinir_tipi = "Üst Limit"
    elif egim < -0.00001:
        hedef_parca = (alt_limit - kesim) / egim
        kalan_parca = hedef_parca - current_parca_no
        sinir_tipi = "Alt Limit"
    else:
        kalan_parca = 9999999
        sinir_tipi = "Stabil"
        
    # Sonucu tamsayıya ve baskı adedine çevir
    kalan_baski = int(kalan_parca / urun_miktari)
    
    return kalan_baski, sinir_tipi

# --- EKRAN AKIŞI ---
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Hesaplamalar
    stats = calculate_msa_stats(df['Olcum'], ust_limit, alt_limit)
    
    X = df[['Parca_No']].values.reshape(-1, 1)
    y = df['Olcum'].values
    model = LinearRegression()
    model.fit(X, y)
    
    son_parca = df['Parca_No'].max()
    
    # Hata veren fonksiyon çağrısı artık düzeltildi
    kalan_baski, sinir_tipi = calculate_rul(model, son_parca, ust_limit, alt_limit, baski_basina_urun)
    
    st.divider()
    
    # 1. KPI KARTLARI
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Ortalama", f"{stats['mean']:.3f} mm")
    kpi2.metric("Cpk (Yeterlilik)", f"{stats['Cpk']:.2f}")
    kpi3.metric("Ppk (Performans)", f"{stats['Ppk']:.2f}")
    kpi4.metric("Kalan Baskı Ömrü", f"{kalan_baski}", f"{sinir_tipi}")
    
    st.divider()

    # 2. GRAFİKLER
    tab1, tab2, tab3 = st.tabs(["📉 Kalıp Ömrü", "📊 MSA Dağılımı", "📋 Veri Listesi"])
    
    with tab1:
        fig = go.Figure()
        # Gerçek Veri
        fig.add_trace(go.Scatter(x=df['Parca_No'], y=df['Olcum'], name='Ölçüm', opacity=0.5))
        
        # Gelecek Tahmini
        gelecek_adim = 500 * baski_basina_urun
        gelecek_no = np.arange(son_parca + 1, son_parca + gelecek_adim).reshape(-1, 1)
        gelecek_pred = model.predict(gelecek_no)
        
        fig.add_trace(go.Scatter(x=gelecek_no.flatten(), y=gelecek_pred, name='Tahmin', line=dict(color='green', dash='dash')))
        
        # Limitler
        fig.add_hline(y=ust_limit, line_color="red")
        fig.add_hline(y=alt_limit, line_color="red")
        
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig_hist = px.histogram(df, x="Olcum", nbins=40, marginal="box", title="Normal Dağılım Analizi")
        fig_hist.add_vline(x=ust_limit, line_color="red")
        fig_hist.add_vline(x=alt_limit, line_color="red")
        fig_hist.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.dataframe(df, use_container_width=True)

else:
    st.warning("👆 Analize başlamak için lütfen yukarıdaki alana dosyanızı yükleyin.")
