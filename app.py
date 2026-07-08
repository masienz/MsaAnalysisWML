import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis
import json
import os

# --- SAYFA VE ŞABLON AYARLARI ---
st.set_page_config(page_title="Ölçüm Makinesi Kestirimci Kalite", layout="wide")

TEMPLATE_FILE = "sablonlar.json"

def load_templates():
    if os.path.exists(TEMPLATE_FILE):
        with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_template(name, data):
    templates = load_templates()
    templates[name] = data
    with open(TEMPLATE_FILE, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=4)

# Session State Başlangıç Değerleri (secilen_olculer EKLENDİ)
if 'ayarlar' not in st.session_state:
    st.session_state.ayarlar = {
        'secilen_olculer': ["İç Çap", "Dış Çap", "Yükseklik"],
        'cevrim_suresi': 30, 'kontrol_sikligi': 4.0, 'baski_basina': 4,
        'maliyet': 15.0, 'gurultu_filtresi': 5,
        'ic_nom': 20.0, 'ic_ust': 0.10, 'ic_alt': 0.10,
        'dis_nom': 45.0, 'dis_ust': 0.20, 'dis_alt': 0.20,
        'yuk_nom': 10.0, 'yuk_ust': 0.10, 'yuk_alt': 0.05
    }

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 3rem;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stMetric {background-color: #0e1117; border: 1px solid #404040; border-radius: 5px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🏭 Ölçüm Makinesi - Kestirimci Kalite ve Kalıp Ömür Analizi")

# --- YAN MENÜ ---
st.sidebar.title("💾 Şablon Yönetimi")
mevcut_sablonlar = load_templates()
secilen_sablon = st.sidebar.selectbox("Kayıtlı Şablonlar", ["Yeni/Varsayılan"] + list(mevcut_sablonlar.keys()))

if st.sidebar.button("📂 Şablonu Yükle"):
    if secilen_sablon != "Yeni/Varsayılan":
        st.session_state.ayarlar.update(mevcut_sablonlar[secilen_sablon])
        st.rerun()

st.sidebar.divider()
st.sidebar.title("⚙️ Üretim Parametreleri")

a = st.session_state.ayarlar

st.sidebar.subheader("⏱️ Çevrim ve Kontrol")
cevrim_suresi = st.sidebar.number_input("Çevrim Süresi (Saniye)", value=a['cevrim_suresi'], step=1)
kontrol_sikligi = st.sidebar.number_input("Kalite Kontrol Sıklığı (Saat)", value=a['kontrol_sikligi'], step=0.5)
baski_basina_urun = st.sidebar.number_input("Çevrim Başına Üretim", value=a['baski_basina'], step=1)
gurultu_filtresi = st.sidebar.slider("Gürültü Filtresi (Puan)", min_value=1, max_value=30, value=a.get('gurultu_filtresi', 5), help="Değer arttıkça anlık ölçüm hataları yok sayılır, kalıbın gerçek aşınma ömrü uzar.")

st.sidebar.subheader("💰 Maliyet")
para_birimi = st.sidebar.selectbox("Para Birimi", ["TL", "USD ($)", "EUR (€)"])
simge = para_birimi.split("(")[-1].replace(")", "") 
urun_maliyeti = st.sidebar.number_input(f"Birim Ürün Maliyeti ({simge})", value=a['maliyet'], step=0.5)

st.sidebar.subheader("📋 Veri Seçimi")
# DÜZELTME: Seçilen ölçüler artık kaydedilen şablondan geliyor
secilen_olculer = st.sidebar.multiselect(
    "Excel'deki Ölçüleriniz:", 
    ["İç Çap", "Dış Çap", "Yükseklik"], 
    default=a.get('secilen_olculer', ["İç Çap", "Dış Çap", "Yükseklik"])
)

st.sidebar.subheader("📐 Toleranslar")
def tolerance_input(label, key_prefix):
    with st.sidebar.expander(f"{label} Ayarları", expanded=False):
        nom = st.number_input(f"{label} Nominal", value=a[f'{key_prefix}_nom'], step=0.01)
        tol_plus = st.number_input(f"{label} Tolerans (+)", value=a[f'{key_prefix}_ust'], step=0.01)
        tol_minus = st.number_input(f"{label} Tolerans (-)", value=a[f'{key_prefix}_alt'], step=0.01)
        return {"Nom": nom, "USL": nom + tol_plus, "LSL": nom - tol_minus, "raw": (nom, tol_plus, tol_minus)}

limits = {}
if "İç Çap" in secilen_olculer: limits["İç Çap"] = tolerance_input("İç Çap", "ic")
if "Dış Çap" in secilen_olculer: limits["Dış Çap"] = tolerance_input("Dış Çap", "dis")
if "Yükseklik" in secilen_olculer: limits["Yükseklik"] = tolerance_input("Yükseklik", "yuk")

st.sidebar.divider()
kayit_adi = st.sidebar.text_input("Şablon Adı (Örn: R002 Kalıbı)")
if st.sidebar.button("💾 Mevcut Ayarları Kaydet"):
    if kayit_adi:
        yeni_ayarlar = {
            'secilen_olculer': secilen_olculer, # DÜZELTME: Listeyi kaydet
            'cevrim_suresi': cevrim_suresi, 'kontrol_sikligi': kontrol_sikligi, 'baski_basina': baski_basina_urun,
            'maliyet': urun_maliyeti, 'gurultu_filtresi': gurultu_filtresi,
            'ic_nom': limits.get("İç Çap", {}).get("raw", (20.0, 0.1, 0.1))[0],
            'ic_ust': limits.get("İç Çap", {}).get("raw", (20.0, 0.1, 0.1))[1],
            'ic_alt': limits.get("İç Çap", {}).get("raw", (20.0, 0.1, 0.1))[2],
            'dis_nom': limits.get("Dış Çap", {}).get("raw", (45.0, 0.2, 0.2))[0],
            'dis_ust': limits.get("Dış Çap", {}).get("raw", (45.0, 0.2, 0.2))[1],
            'dis_alt': limits.get("Dış Çap", {}).get("raw", (45.0, 0.2, 0.2))[2],
            'yuk_nom': limits.get("Yükseklik", {}).get("raw", (10.0, 0.1, 0.05))[0],
            'yuk_ust': limits.get("Yükseklik", {}).get("raw", (10.0, 0.1, 0.05))[1],
            'yuk_alt': limits.get("Yükseklik", {}).get("raw", (10.0, 0.1, 0.05))[2]
        }
        save_template(kayit_adi, yeni_ayarlar)
        st.sidebar.success(f"'{kayit_adi}' başarıyla kaydedildi!")
    else:
        st.sidebar.error("Lütfen bir şablon adı girin.")

# --- FONKSİYONLAR ---
def clean_turkish_numbers(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

def calculate_msa_stats(data, specs):
    mean = np.mean(data)
    std_sample = np.std(data, ddof=1)
    std_pop = np.std(data, ddof=0)
    usl, lsl = specs["USL"], specs["LSL"]
    
    Cp = (usl - lsl) / (6 * std_sample) if std_sample > 0 else 0
    Cpk = min((usl - mean) / (3 * std_sample), (mean - lsl) / (3 * std_sample)) if std_sample > 0 else 0
    Pp = (usl - lsl) / (6 * std_pop) if std_pop > 0 else 0
    Ppk = min((usl - mean) / (3 * std_pop), (mean - lsl) / (3 * std_pop)) if std_pop > 0 else 0
    
    return {"Mean": mean, "Std": std_sample, "Cp": Cp, "Cpk": Cpk, "Pp": Pp, "Ppk": Ppk, "Skew": skew(data), "Kurt": kurtosis(data)}

def find_regime_change(x, y, min_segment_frac=0.1, min_segment_abs=3):
    """Kalıp ömrü boyunca sabit bir aşınma hızı yoktur: süreç uzun süre durağan kalıp
    sonra aşınmaya başlayabilir. Veriyi ikiye bölüp toplam hata karesini (SSE) en çok
    azaltan noktayı bulur; bu nokta mevcut aşınma rejiminin başlangıcıdır."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    min_seg = max(min_segment_abs, int(n * min_segment_frac))
    if n < 2 * min_seg:
        return None

    zero = np.zeros(1)
    cx = np.concatenate([zero, np.cumsum(x)])
    cy = np.concatenate([zero, np.cumsum(y)])
    cxy = np.concatenate([zero, np.cumsum(x * y)])
    cx2 = np.concatenate([zero, np.cumsum(x * x)])
    cy2 = np.concatenate([zero, np.cumsum(y * y)])

    def seg_sse(lo, hi):
        n_ = hi - lo
        if n_ < 2:
            return 0.0
        sx, sy = cx[hi] - cx[lo], cy[hi] - cy[lo]
        sxy, sx2, sy2 = cxy[hi] - cxy[lo], cx2[hi] - cx2[lo], cy2[hi] - cy2[lo]
        sxx = sx2 - sx * sx / n_
        syy = sy2 - sy * sy / n_
        sxy_c = sxy - sx * sy / n_
        if sxx <= 1e-12:
            return syy
        return syy - (sxy_c * sxy_c) / sxx

    single_sse = seg_sse(0, n)
    best_i, best_sse = None, np.inf
    for i in range(min_seg, n - min_seg):
        sse = seg_sse(0, i) + seg_sse(i, n)
        if sse < best_sse:
            best_sse, best_i = sse, i

    # Bölme, tek doğruya kıyasla en az %3 iyileşme sağlamıyorsa rejim değişikliği yok say
    if best_i is not None and best_sse < single_sse * 0.97:
        return best_i
    return None

def find_all_regime_changes(x, y, min_segment_frac=0.15, min_segment_abs=5, max_segments=4):
    """Seriyi BIC (Bayesian Information Criterion) cezalı, en fazla `max_segments` evreye
    bölen açgözlü (greedy) bir segmentasyon yapar. Amaç: kalıbın kendi geçmişinde daha önce
    benzer bir düşüş yaşayıp yaşamadığını ve toparlandıysa ne hızda toparlandığını tespit
    etmek. Basit "yerel %3 iyileşme" eşiği (find_regime_change), tekrar tekrar uygulanınca
    (özyinelemeli) küçülen alt segmentlerde gürültüden bile "iyileşme" bulabiliyordu (saf
    tek-yönlü trend + gürültüde %33 sahte bölünme oranı ölçüldü) — BIC, her ek segmentin
    getirdiği parametre (eğim+kesim) maliyetini global olarak cezalandırdığı için bu riski
    ortadan kaldırır. Minimum segment boyutu ORİJİNAL seri uzunluğuna göre sabittir (alt
    segmentlere göre KÜÇÜLMEZ), bu da aşırı parçalanmayı engeller."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    min_seg = max(min_segment_abs, int(n * min_segment_frac))
    if n < 2 * min_seg:
        return [(0, n)]

    zero = np.zeros(1)
    cx = np.concatenate([zero, np.cumsum(x)])
    cy = np.concatenate([zero, np.cumsum(y)])
    cxy = np.concatenate([zero, np.cumsum(x * y)])
    cx2 = np.concatenate([zero, np.cumsum(x * x)])
    cy2 = np.concatenate([zero, np.cumsum(y * y)])

    def seg_sse(lo, hi):
        n_ = hi - lo
        if n_ < 2:
            return 0.0
        sx, sy = cx[hi] - cx[lo], cy[hi] - cy[lo]
        sxy, sx2, sy2 = cxy[hi] - cxy[lo], cx2[hi] - cx2[lo], cy2[hi] - cy2[lo]
        sxx = sx2 - sx * sx / n_
        syy = sy2 - sy * sy / n_
        sxy_c = sxy - sx * sy / n_
        if sxx <= 1e-12:
            return max(syy, 1e-12)
        return max(syy - (sxy_c * sxy_c) / sxx, 1e-12)

    def total_bic(boundaries):
        total_sse = sum(seg_sse(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1))
        k = len(boundaries) - 1
        return n * np.log(total_sse / n) + (2 * k) * np.log(n)

    boundaries = [0, n]
    best_bic = total_bic(boundaries)
    while len(boundaries) - 1 < max_segments:
        best_candidate = None
        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 2 * min_seg:
                continue
            local_split = find_regime_change(x[lo:hi], y[lo:hi], min_segment_frac, min_segment_abs)
            if local_split is None:
                continue
            split = lo + local_split
            candidate_bounds = sorted(boundaries + [split])
            b = total_bic(candidate_bounds)
            if b < best_bic - 1e-9 and (best_candidate is None or b < best_candidate[0]):
                best_candidate = (b, split)
        if best_candidate is None:
            break
        best_bic, split = best_candidate
        boundaries = sorted(boundaries + [split])
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

# Fiziksel olarak KALICI aşınmanın hangi yönde olduğu bilinen ölçüler: Dış Çap dişi
# (female) bir kavite tarafından şekillenir — kavite aşınırsa BÜYÜR (USL yönü). Bu
# yönde bir yaklaşma gerçek/kalıcı aşınma sayılır ve toparlanma VARSAYILMAZ; ters yönde
# (LSL'ye yaklaşma) tipik olarak kalıp yüzeyindeki birikinti/kir (fouling) sorumludur ve
# temizlikle giderilebilir, bu yüzden orada toparlanma emsali dikkate alınabilir.
KALICI_ASINMA_YONU = {"Dış Çap": "USL"}

def analyze_trend_pure(df, col_name, specs, baski_basina_urun, window, kalici_asinma_yonu=None):
    df = df.sort_values('Parca_No')

    # 0. BASKI SEVİYESİNE TOPLULAŞTIRMA: çok gözlü (çok kaviteli) kalıplarda bir tek
    # baskıda üretilen TÜM parçalar aynı kavite setinden aynı anda çıkar. Parça bazında
    # regresyon, kavite-arası/kesim-pozisyonu farkını (gerçek R002 verisinde toplam
    # varyansın %96'sı!) aşınma trendiyle karıştırıp güveniliriliği düşürür (R²≈0.05).
    # Baskı ortalaması alınca bu baskı-içi gürültü ayıklanır (aynı veride R²≈0.81'e çıktı).
    if baski_basina_urun and baski_basina_urun > 1 and len(df) >= 2 * baski_basina_urun:
        baski_idx = (df['Parca_No'] - df['Parca_No'].min()) // baski_basina_urun
        df = df.groupby(baski_idx, as_index=False).agg({'Parca_No': 'mean', col_name: 'mean'})

    # 1. ÖN REGRESYON: trend yönünü/eğimini kabaca belirle (ham veri üzerinde)
    X_all = df[['Parca_No']].values.reshape(-1, 1)
    y_all = df[col_name].values
    pre_model = LinearRegression().fit(X_all, y_all)
    residuals = y_all - pre_model.predict(X_all)
    resid_std = residuals.std()

    # 2. AYKIRI DEĞER (SERSERİ NOKTA) TEMİZLİĞİ: trendden ayrıştırılmış KALINTI üzerinden
    # (Ham/kaydırılmamış hareketli ortalama üzerinden filtreleme, aşınma trendinin
    #  ucundaki en kritik noktaları "aykırı" sayıp atma ve RUL'de gecikme yanlılığı
    #  (lag bias) yaratma riski taşıdığı için kalıntı bazlı yapılır.)
    # NOT: Eşik 2.0 yerine 3.0 sigma — döngüsel (düşüş-toparlanma) bir seride, tek bir
    # düz çizgiye göre en çok sapan nokta genellikle GERÇEK bir tepe/dip (toparlanma
    # kanıtı) olur; 2.0 sigma bu gerçek dönüş noktalarını "aykırı" sayıp siliyordu
    # (gerçek R002 verisinde toparlanma tepe noktası böyle kayboluyordu).
    if resid_std > 0:
        valid_data = df[np.abs(residuals) <= (3.0 * resid_std)]
    else:
        valid_data = df

    if len(valid_data) < 5:
        valid_data = df # Yeterli veri kalmadıysa mecburen hepsini kullan

    valid_data = valid_data.sort_values('Parca_No').copy()
    # Sadece GÖRSELLEŞTİRME için yumuşatılmış çizgi (nihai tahmine girmez)
    valid_data['Smoothed'] = valid_data[col_name].rolling(window=window, min_periods=1, center=True).mean()

    # 3. REJİM TESPİTİ: kalıp uzun süre durağan kalıp sonra aşınmaya başlamış olabilir.
    # Tüm geçmişe TEK doğru uydurmak bu başlangıcı sulandırıp RUL'ü büyütür. Regresyonu
    # sadece MEVCUT aşınma rejimine (son değişim noktasından sonrasına) dayandırıyoruz.
    split_idx = find_regime_change(valid_data['Parca_No'].values, valid_data[col_name].values)
    regime_data = valid_data.iloc[split_idx:] if split_idx is not None else valid_data
    if len(regime_data) < 5:
        regime_data = valid_data

    # 4. NİHAİ REGRESYON: mevcut rejimin temizlenmiş HAM verisi üzerinden (gecikme yanlılığı yok)
    X = regime_data[['Parca_No']].values.reshape(-1, 1)
    y = regime_data[col_name].values

    model = LinearRegression()
    model.fit(X, y)

    egim = model.coef_[0]
    current_parca = df['Parca_No'].max()
    son_deger = valid_data['Smoothed'].iloc[-1]

    # 5. GEÇMİŞ TOPARLANMA TESPİTİ: kauçuk/metal kalıp çiftinde metal kavite bu kadar
    # hızlı aşınmaz — kısa vadeli düşüşler genellikle ısıl/malzeme kaynaklı GEÇİCİ
    # dalgalanmalardır. find_all_regime_changes ile kalıbın kendi geçmişinde daha önce
    # böyle bir düşüşten kendiliğinden toparlanma olup olmadığına bakılır.
    segments = find_all_regime_changes(valid_data['Parca_No'].values, valid_data[col_name].values)
    seg_slopes = []
    for lo, hi in segments:
        if hi - lo < 2:
            continue
        xs = valid_data['Parca_No'].values[lo:hi].reshape(-1, 1)
        ys = valid_data[col_name].values[lo:hi]
        seg_slopes.append(LinearRegression().fit(xs, ys).coef_[0])

    # mevcut (son) evrenin ZIT yönünde hareket eden GEÇMİŞ evreler = geçmiş toparlanmalar
    recovery_slopes = [s for s in seg_slopes[:-1] if np.sign(s) == -np.sign(egim) and s != 0] if len(seg_slopes) > 1 else []
    # GÜVENLİK EŞİĞİ: tek bir geçmiş evre gürültüden de çıkabilir (gerçekçi R002-ölçekli
    # gürültüyle yapılan simülasyonda tek-evre şartıyla %5, en az 2 bağımsız evre
    # şartıyla %1 yanlış-toparlanma oranı ölçüldü) — bu yüzden en az 2 bağımsız geçmiş
    # toparlanma evresi görülmeden "toparlanma eğilimi" güvenilir kabul edilmez.
    avg_recovery_egim = float(np.mean(recovery_slopes)) if len(recovery_slopes) >= 2 else None

    # FİZİKSEL YÖN KONTROLÜ: mevcut eğilim, bu ölçü için bilinen KALICI aşınma yönüyle
    # aynıysa (örn. Dış Çap'ta USL'ye/büyümeye doğru gidiş), toparlanma varsayılmaz —
    # bu, geri dönüşü olmayan gerçek metal aşınmasıdır, birikinti değildir.
    kalici_asinma_engeli = False
    if kalici_asinma_yonu is not None:
        mevcut_yon = "USL" if egim > 0 else "LSL"
        if mevcut_yon == kalici_asinma_yonu:
            avg_recovery_egim = None
            kalici_asinma_engeli = True

    # 6. NİHAİ EĞİM SEÇİMİ: geçmişte gerçek bir toparlanma emsali varsa, kalıbın kalan
    # ömrü bu daha gerçekçi (toparlanma) eğilimine göre hesaplanır; emsal yoksa mevcut
    # aşınma eğilimi kullanılır (başka veri/emsal olmadan tek elde olan budur).
    if avg_recovery_egim is not None:
        final_egim = avg_recovery_egim
        yontem = f"Toparlanma Eğilimi (geçmişte {len(recovery_slopes)} kez gözlendi)"
    else:
        final_egim = egim
        yontem = "Mevcut Aşınma Eğilimi (kalıcı/geri dönüşsüz aşınma yönü)" if kalici_asinma_engeli else "Mevcut Aşınma Eğilimi"

    # 7. GÜNCEL SEVİYEYE ÇAPALAMA (anchoring): doğru kendi ortalama konumundan değil,
    # kalıbın AN İTİBARİYLE gerçekte ölçülen yerinden başlamalı — aksi halde "Temizlenmiş
    # Trend" çizgisinin bittiği nokta ile "Gelecek Tahmini"nin başladığı nokta arasında
    # gerçek olmayan bir sıçrama oluşur.
    kesim = son_deger - final_egim * current_parca
    model.intercept_ = kesim
    model.coef_[0] = final_egim

    if son_deger >= specs["USL"] or son_deger <= specs["LSL"]:
        return {"Model": model, "RUL": 0, "Limit": "Limit Aşıldı", "ValidData": valid_data, "Yontem": yontem}
    if abs(final_egim) < 0.0000005:
        return {"Model": model, "RUL": 99999999, "Limit": "Stabil", "ValidData": valid_data, "Yontem": yontem}

    kalan_usl = ((specs["USL"] - kesim) / final_egim) - current_parca
    kalan_lsl = ((specs["LSL"] - kesim) / final_egim) - current_parca

    gelecek_kesisimler = []
    if kalan_usl > 0: gelecek_kesisimler.append(("Üst Limit", kalan_usl))
    if kalan_lsl > 0: gelecek_kesisimler.append(("Alt Limit", kalan_lsl))

    if len(gelecek_kesisimler) > 0:
        hedef_limit, kalan_parca = min(gelecek_kesisimler, key=lambda x: x[1])
        kalan_baski = int(kalan_parca / baski_basina_urun)
        return {"Model": model, "RUL": kalan_baski, "Limit": hedef_limit, "ValidData": valid_data, "Yontem": yontem}
    else:
        return {"Model": model, "RUL": 99999999, "Limit": "Stabil", "ValidData": valid_data, "Yontem": yontem}

# --- EKRAN AKIŞI ---
with st.container():
    c1, c2 = st.columns([1, 2])
    with c1:
        if len(secilen_olculer) == 0: st.warning("⚠️ Lütfen yan menüden ölçüm seçin!")
        uploaded_file = st.file_uploader("📂 Veri Yükle (.xlsx / .csv)", type=["xlsx", "csv"])
        
        if uploaded_file is not None and len(secilen_olculer) > 0:
            try:
                if uploaded_file.name.endswith('.csv'): df_upload = pd.read_csv(uploaded_file, header=None, sep=None, engine='python')
                else: df_upload = pd.read_excel(uploaded_file, header=None)
                
                if len(df_upload.columns) >= len(secilen_olculer) + 1:
                    df_clean = df_upload.copy()
                    temp_df = pd.DataFrame()
                    av_cols, av_labels = [], []
                    
                    temp_df['Parca_No'] = clean_turkish_numbers(df_clean.iloc[:, 0])
                    for i, olcu in enumerate(secilen_olculer):
                        col_idx = i + 1
                        col_name = olcu.replace(" ", "_").replace("ç", "c").replace("Ç", "C").replace("ş", "s").replace("ı", "i").replace("ü", "u")
                        temp_df[col_name] = clean_turkish_numbers(df_clean.iloc[:, col_idx])
                        av_cols.append(col_name)
                        av_labels.append(olcu)
                    
                    temp_df = temp_df.dropna()
                    if len(temp_df) > 1:
                        st.session_state['data'] = temp_df
                        st.session_state['available_cols'] = av_cols
                        st.session_state['available_labels'] = av_labels
                        st.success(f"✅ Başarılı: {len(temp_df)} satır yüklendi.")
                    else: st.error("Hata: Geçerli veri bulunamadı.")
                else: st.error("Hata: Sütun sayısı eksik.")
            except Exception as e: st.error(f"Hata: {e}")
            
    with c2:
        saatlik_baski = 3600 / cevrim_suresi
        kacirilan_baski = int(saatlik_baski * kontrol_sikligi)
        st.info(f"**Sistem Mantığı:** Çevrim süresine göre {kontrol_sikligi} saatlik kalite kontrol periyodunda {kacirilan_baski} baskı üretilir — bu, iki kontrol arasında kaçırılabilecek maksimum baskı sayısıdır ('Kör Nokta'). Tahmini **Bakım Zamanı** bu değerin altına düşerse 🚨 ACİL, 10 katına kadarsa ⚠️ PLANLI BAKIM alarmı verilir.")

if 'data' in st.session_state and 'available_cols' in st.session_state:
    df = st.session_state['data']
    cols = st.session_state['available_cols']
    labels = st.session_state['available_labels']
    
    results, ruls = {}, {}
    for col, label in zip(cols, labels):
        stats = calculate_msa_stats(df[col], limits[label])
        trend = analyze_trend_pure(df, col, limits[label], baski_basina_urun, gurultu_filtresi, KALICI_ASINMA_YONU.get(label))
        results[col] = {"stats": stats, "trend": trend}
        ruls[label] = trend["RUL"]

    en_kritik_hat = min(ruls, key=ruls.get)
    min_rul = ruls[en_kritik_hat]
    blind_spot_baski = (3600 / cevrim_suresi) * kontrol_sikligi
    
    gercek_tasarruf, durum_tipi, durum_mesaji = 0, "success", "Süreç Stabil"
    
    if min_rul <= blind_spot_baski:
        gercek_tasarruf = int(blind_spot_baski * baski_basina_urun) * urun_maliyeti
        durum_tipi = "error"
        if min_rul == 0: durum_mesaji = f"🚨 ACİL DURUM: '{en_kritik_hat}' zaten tolerans dışında!"
        else: durum_mesaji = f"⚠️ RİSK: '{en_kritik_hat}' {min_rul} baskı sonra limit dışına çıkacak — bakım/temizlik gerekebilir!"
    elif min_rul < (blind_spot_baski * 10):
        durum_mesaji = f"⚠️ UYARI: '{en_kritik_hat}' bakım sinyali veriyor."
        durum_tipi = "warning"

    st.divider()
    k1, k2, k3 = st.columns(3)
    omur_yazi = "Sonsuz" if min_rul > 1000000 else f"{min_rul} Baskı"
    if min_rul == 0: omur_yazi = "0 Baskı (Limit Dışı)"

    kritik_trend = results[cols[labels.index(en_kritik_hat)]]["trend"]
    kritik_yontem = kritik_trend.get("Yontem", "Mevcut Aşınma Eğilimi")

    k1.metric("Bakım Zamanı (Tahmini)", omur_yazi, f"Hat: {en_kritik_hat}", delta_color="inverse" if min_rul > 0 else "off",
              help=f"Yöntem: {kritik_yontem}. Bu, kalıbın toplam ömrü değil — bir sonraki temizlik/bakımın gerekeceği tahmini noktadır. Kalıp geçmişte benzer bir düşüşten toparlandıysa (ve bu yön kalıcı aşınma olarak işaretli değilse) bu daha gerçekçi toparlanma eğilimine göre hesaplanır; emsal yoksa mevcut aşınma eğilimi kullanılır.")
    k2.metric("Potansiyel Tasarruf", f"{gercek_tasarruf:,.0f} {simge}", "Hurda Önleme")
    
    genel_ikon = "✅ Stabil"
    if durum_tipi == "error": genel_ikon = "🚨 ACİL BAKIM"
    elif durum_tipi == "warning": genel_ikon = "⚠️ PLANLI BAKIM"
    k3.metric("Operasyonel Durum", genel_ikon)
    if durum_tipi != "success":
        if durum_tipi == "error": st.error(durum_mesaji)
        else: st.warning(durum_mesaji)

    st.caption(f"ℹ️ Hesaplama yöntemi ('{en_kritik_hat}'): **{kritik_yontem}**")

    st.markdown("### 📊 Detaylı Analiz")
    tabs = st.tabs([f"📍 {lbl}" for lbl in labels])
    
    for i, tab in enumerate(tabs):
        col_key, label_key = cols[i], labels[i]
        res, trend, specs, stats = results[col_key], results[col_key]["trend"], limits[label_key], results[col_key]["stats"]
        valid_df = trend["ValidData"]
        
        with tab:
            fig = go.Figure()
            # Ham Veri
            fig.add_trace(go.Scatter(x=df['Parca_No'], y=df[col_key], name='Ham Ölçüm', mode='markers', opacity=0.3, marker=dict(color='gray', size=4)))
            # Filtrelenmiş Veri
            fig.add_trace(go.Scatter(x=valid_df['Parca_No'], y=valid_df['Smoothed'], name='Temizlenmiş Trend', mode='lines', line=dict(color='cyan', width=2)))
            
            son, ek_uzunluk = df['Parca_No'].max(), max(len(df) * 0.2, 100)
            gelecek = np.arange(son, son + ek_uzunluk).reshape(-1, 1)
            pred = trend["Model"].predict(gelecek)
            
            fig.add_trace(go.Scatter(x=gelecek.flatten(), y=pred, name='Gelecek Tahmini', line=dict(color='orange', width=3, dash='dash')))

            fig.add_hline(y=specs["USL"], line_color="red", annotation_text="USL")
            fig.add_hline(y=specs["LSL"], line_color="red", annotation_text="LSL")
            fig.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("📊 MSA İstatistikleri ve Dağılım (IATF 16949)")
            col_msa_stats, col_msa_graph = st.columns([1, 2])

            with col_msa_stats:
                m1, m2 = st.columns(2)
                m1.metric("Ortalama", f"{stats['Mean']:.4f}")
                m2.metric("Std. Sapma", f"{stats['Std']:.5f}")

                m3, m4 = st.columns(2)
                m3.metric("Cp", f"{stats['Cp']:.2f}")
                m4.metric("Cpk", f"{stats['Cpk']:.2f}", delta_color="normal" if stats['Cpk'] > 1.33 else "inverse")

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
                fig_hist.add_vline(x=specs["Nom"], line_color="green", line_dash="dot", opacity=0.5, annotation_text="Nom")
                fig_hist.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0), showlegend=False, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist, use_container_width=True)

            with st.expander(f"📋 {label_key} Veri Listesi"):
                st.dataframe(df[['Parca_No', col_key]].sort_values(by='Parca_No', ascending=False), use_container_width=True)
