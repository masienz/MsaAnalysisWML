import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Endüstriyel Kestirimci Kalite", layout="wide")

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 3rem;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stMetric {background-color: #0e1117; border: 1px solid #404040; border-radius: 5px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🏭 Endüstriyel Kestirimci Kalite ve Kalıp Ömür Analiz Sistemi")

# --- YAN MENÜ ---
st.sidebar.title("⚙️ Üretim Parametreleri")

st.sidebar.subheader("⏱️ Çevrim ve Kontrol")
cevrim_suresi = st.sidebar.number_input("Çevrim Süresi (Saniye)", value=30, step=1)
kontrol_sikligi = st.sidebar.number_input("Kalite Kontrol Sıklığı (Saat)", value=4.0, step=0.5)
baski_basina_urun = st.sidebar.number_input("Göz Sayısı (Adet)", value=4, step=1)

st.sidebar.subheader("💰 Maliyet")
para_birimi = st.sidebar.selectbox("Para Birimi", ["TL", "USD ($)", "EUR (€)"])
simge = para_birimi.split("(")[-1].replace(")", "") 
urun_maliyeti = st.sidebar.number_input(f"Birim Ürün Maliyeti ({simge})", value=15.0, step=0.5)

st.sidebar.subheader("📐 Toleranslar")
def tolerance_input(label, default_nom, default_plus, default_minus):
    with st.sidebar.expander(f"{label} Ayarları", expanded=False):
        nom = st.number_input(f"{label} Nominal", value=default_nom, step=0.01)
        tol_plus = st.number_input(f"{label} Tolerans (+)", value=default_plus, step=0.01)
        tol_minus = st.number_input(f"{label} Tolerans (-)", value=default_minus, step=0.01)
        return {"Nom": nom, "USL": nom + tol_plus, "LSL": nom - tol_minus}

limits = {
    "İç Çap":  tolerance_input("1. İç Çap", 20.00, 0.10, 0.10),
    "Dış Çap": tolerance_input("2. Dış Çap", 45.00, 0.20, 0.20),
    "Yükseklik": tolerance_input("3. Yükseklik", 10.00, 0.10, 0.05)
}

# --- FONKSİYONLAR ---

def generate_demo_data():
    """Ultra Hassas Simülasyon Verisi"""
    n = 1000 
    ic_cap = np.random.normal(limits["İç Çap"]["Nom"], 0.0012, n) + np.linspace(0, 0.008, n)
    dis_cap = np.random.normal(limits["Dış Çap"]["Nom"], 0.002, n) 
    yukseklik = np.random.normal(limits["Yükseklik"]["Nom"], 0.001, n) - np.linspace(0, 0.005, n)
    return pd.DataFrame({'Parca_No': range(1, n + 1), 'Ic_Cap': ic_cap, 'Dis_Cap': dis_cap, 'Yukseklik': yukseklik})

def calculate_msa_stats(data, specs):
    mean = np.mean(data)
    std_sample = np.std(data, ddof=1)
    std_pop = np.std(data, ddof=0)
    usl, lsl = specs["USL"], specs["LSL"]
    
    Cp = (usl - lsl) / (6 * std_sample)
    Cpk = min((usl - mean) / (3 * std_sample), (mean - lsl) / (3 * std_sample))
    Pp = (usl - lsl) / (6 * std_pop)
    Ppk = min((usl - mean) / (3 * std_pop), (mean - lsl) / (3 * std_pop))
    
    return {
        "Mean": mean, "Std": std_sample, 
        "Cp": Cp, "Cpk": Cpk, "Pp": Pp, "Ppk": Ppk,
        "Skew": skew(data), "Kurt": kurtosis(data)
    }

def analyze_trend_pure(df, col_name, specs, goz_sayisi):
    X = df[['Parca_No']].values.reshape(-1, 1)
    y = df[col_name].values
    model = LinearRegression()
    model.fit(X, y)
    
    egim = model.coef_[0]
    kesim = model.intercept_
    current_parca = df['Parca_No'].max()
    
    kalan_parca = 9999999
    limit_tipi = "Stabil"
    
    if abs(egim) > 0.0000005: 
        target = specs["USL"] if egim > 0 else specs["LSL"]
        kalan_parca = ((target - kesim) / egim) - current_parca
        limit_tipi = "Üst Limit" if egim > 0 else "Alt Limit"
    
    kalan_baski = int(kalan_parca / goz_sayisi) if kalan_parca > 0 else 0
    return {"Model": model, "RUL": kalan_baski, "Limit": limit_tipi}

# --- EKRAN AKIŞI ---

with st.container():
    c1, c2 = st.columns([1, 2])
    with c1:
        uploaded_file = st.file_uploader("📂 Veri Dosyası Yükle (.xlsx / .csv)", type=["xlsx", "csv"])
        
        # --- DÜZELTME BURADA YAPILDI ---
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file, header=None)
                else:
                    df_upload = pd.read_excel(uploaded_file, header=None)
                
                # Excel'den gelen veriyi standart formata çevir (İlk 4 sütun)
                if len(df_upload.columns) >= 4:
                    # Başlık satırı varsa ve metin ise (str) temizle, yoksa direkt al
                    # Basitlik için verinin sayısal olduğunu varsayıyoruz ve ilk 4 sütunu alıyoruz
                    df_clean = df_upload.iloc[:, :4]
                    # Eğer ilk satır metin ise (Header), onu atla
                    if isinstance(df_clean.iloc[0,0], str):
                         df_clean = df_clean.iloc[1:]
                    
                    df_clean.columns = ['Parca_No', 'Ic_Cap', 'Dis_Cap', 'Yukseklik']
                    # Sayısala çevir (Hata varsa NaN yap ve temizle)
                    for col in df_clean.columns:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    df_clean = df_clean.dropna()
                    
                    st.session_state['data'] = df_clean
                    st.success(f"✅ Dosya Yüklendi: {len(df_clean)} satır veri alındı.")
                else:
                    st.error("Hata: Dosyada en az 4 sütun olmalı (Parça No | İç Çap | Dış Çap | Yükseklik)")
            except Exception as e:
                st.error(f"Dosya okuma hatası: {e}")

        if st.button("🧪 Simülasyon Verisi Oluştur"):
            st.session_state['data'] = generate_demo_data()
            st.rerun()
            
    with c2:
        saatlik_baski = 3600 / cevrim_suresi
        kacirilan_baski = int(saatlik_baski * kontrol_sikligi)
        st.info(f"**Sistem Mantığı:** {kontrol_sikligi} saatlik kontrol periyodu içinde {kacirilan_baski} baskı üretilir. RUL bu değerden küçükse risk alarmı verilir.")

if 'data' in st.session_state:
    df = st.session_state['data']
    
    results = {}
    ruls = {}
    
    cols = ["Ic_Cap", "Dis_Cap", "Yukseklik"]
    labels = ["İç Çap", "Dış Çap", "Yükseklik"]
    
    for col, label in zip(cols, labels):
        stats = calculate_msa_stats(df[col], limits[label])
        trend = analyze_trend_pure(df, col, limits[label], baski_basina_urun)
        results[col] = {"stats": stats, "trend": trend}
        ruls[label] = trend["RUL"] if trend["RUL"] > 0 else 99999999

    # KPI HESAPLAMALARI
    en_kritik_hat = min(ruls, key=ruls.get)
    min_rul = ruls[en_kritik_hat]
    
    saatlik_baski = 3600 / cevrim_suresi
    blind_spot_baski = saatlik_baski * kontrol_sikligi
    
    gercek_tasarruf = 0
    durum_tipi = "success"
    durum_mesaji = "Süreç Stabil"
    
    if min_rul < blind_spot_baski:
        kurtarilan_adet = int(blind_spot_baski * baski_basina_urun)
        gercek_tasarruf = kurtarilan_adet * urun_maliyeti
        durum_mesaji = f"⚠️ RİSK: '{en_kritik_hat}' ölçüsü {min_rul} baskı sonra tolerans dışına çıkacak!"
        durum_tipi = "error"
    elif min_rul < (blind_spot_baski * 10):
        durum_mesaji = f"⚠️ UYARI: '{en_kritik_hat}' bakım sinyali veriyor."
        durum_tipi = "warning"
        
    # KPI KARTLARI
    st.divider()
    k1, k2, k3 = st.columns(3)
    omur_yazi = "Sonsuz" if min_rul > 1000000 else f"{min_rul} Baskı"
    k1.metric("En Kritik Kalan Ömür", omur_yazi, f"Hat: {en_kritik_hat}", delta_color="inverse")
    k2.metric("Potansiyel Tasarruf", f"{gercek_tasarruf:,.0f} {simge}", "Hurda Önleme")
    
    genel_ikon = "✅ Stabil"
    if durum_tipi == "error": genel_ikon = "🚨 ACİL BAKIM"
    elif durum_tipi == "warning": genel_ikon = "⚠️ PLANLI BAKIM"
    k3.metric("Operasyonel Durum", genel_ikon)
    
    if durum_tipi != "success":
        if durum_tipi == "error": st.error(durum_mesaji)
        else: st.warning(durum_mesaji)

    # SEKMELER (DETAYLI ANALİZ)
    tab1, tab2, tab3 = st.tabs(["🔴 İç Çap", "🔵 Dış Çap", "🟠 Yükseklik"])
    
    def render_tab(tab, col_key, label_key):
        res = results[col_key]
        trend = res["trend"]
        specs = limits[label_key]
        stats = res["stats"]
        
        with tab:
            # 1. BÖLÜM: TREND GRAFİĞİ
            st.subheader("📉 Trend ve Regresyon Analizi")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Parca_No'], y=df[col_key], name='Ölçüm', mode='markers', opacity=0.6, marker=dict(size=4)))
            
            # Trend Çizgisi
            son = df['Parca_No'].max()
            ek_uzunluk = max(len(df) * 0.2, 100)
            gelecek = np.arange(son, son + ek_uzunluk).reshape(-1, 1)
            pred = trend["Model"].predict(gelecek)
            
            fig.add_trace(go.Scatter(x=gelecek.flatten(), y=pred, name='Eğilim Yönü', line=dict(color='orange', width=3)))
            
            fig.add_hline(y=specs["USL"], line_color="red", annotation_text="USL")
            fig.add_hline(y=specs["LSL"], line_color="red", annotation_text="LSL")
            fig.add_hline(y=specs["Nom"], line_color="green", line_dash="dot", opacity=0.5)
            
            fig.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # 2. BÖLÜM: MSA VE HİSTOGRAM
            st.subheader("📊 MSA İstatistikleri ve Dağılım")
            col_msa_stats, col_msa_graph = st.columns([1, 2])
            
            with col_msa_stats:
                m1, m2 = st.columns(2)
                m1.metric("Ortalama", f"{stats['Mean']:.4f}")
                m2.metric("Std. Sapma", f"{stats['Std']:.5f}")
                
                m3, m4 = st.columns(2)
                m3.metric("Cp", f"{stats['Cp']:.2f}")
                m4.metric("Cpk", f"{stats['Cpk']:.2f}", delta_color="normal" if stats['Cpk']>1.33 else "inverse")
                
                m5, m6 = st.columns(2)
                m5.metric("Pp", f"{stats['Pp']:.2f}")
                m6.metric("Ppk", f"{stats['Ppk']:.2f}")

                m7, m8 = st.columns(2)
                m7.metric("Çarpıklık", f"{stats['Skew']:.3f}")
                m8.metric("Basıklık", f"{stats['Kurt']:.3f}")

            with col_msa_graph:
                fig_hist = px.histogram(df, x=col_key, nbins=40, title=f"{label_key} Frekans Dağılımı")
                fig_hist.add_vline(x=specs["USL"], line_color="red", annotation_text="USL")
                fig_hist.add_vline(x=specs["LSL"], line_color="red", annotation_text="LSL")
                fig_hist.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0), showlegend=False, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # 3. BÖLÜM: HAM VERİ
            with st.expander(f"📋 {label_key} Veri Listesi", expanded=True):
                st.dataframe(df[['Parca_No', col_key]].sort_values(by='Parca_No', ascending=False), use_container_width=True)
    
    render_tab(tab1, "Ic_Cap", "İç Çap")
    render_tab(tab2, "Dis_Cap", "Dış Çap")
    render_tab(tab3, "Yukseklik", "Yükseklik")
