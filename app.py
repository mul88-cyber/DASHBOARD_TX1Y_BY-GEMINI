# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import library Google
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# 🎨 2) KONFIGURASI DASHBOARD & STYLING
# ==============================================================================
st.set_page_config(
    page_title="📊 Quantum IDX Dashboard | Analytics & Simulator",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS untuk UI Modern ----
st.markdown("""
<style>
    /* Global Styles */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Header dengan gradien profesional */
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Metric cards dengan glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Tab styling modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: white;
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f3f5;
        border-color: #dee2e6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4361ee !important;
        color: white !important;
        border-color: #4361ee !important;
        box-shadow: 0 4px 12px rgba(67, 97, 238, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Badge untuk status */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .status-green {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-red {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .status-yellow {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-blue {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ---- Konfigurasi Warna Konsisten ----
COLORS = {
    'primary': '#4361ee',
    'secondary': '#3f37c9',
    'success': '#4cc9f0',
    'danger': '#f72585',
    'warning': '#f8961e',
    'info': '#7209b7',
    'light': '#f8f9fa',
    'dark': '#212529',
    'buy': '#2ecc71',
    'sell': '#e74c3c',
    'neutral': '#95a5a6'
}

# --- KONFIGURASI G-DRIVE ---
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
FILE_NAME = "Kompilasi_Data_1Tahun.csv"

# Bobot skor (Logic "Raport")
W = dict(
    trend_akum=0.40, trend_ff=0.30, trend_mfv=0.20, trend_mom=0.10,
    mom_price=0.40,  mom_vol=0.25,  mom_akum=0.25,  mom_ff=0.10,
    blend_trend=0.35, blend_mom=0.35, blend_nbsa=0.20, blend_fcontrib=0.05, blend_unusual=0.05
)

# ==============================================================================
# 🛠️ FUNGSI FORMATTING BAHASA INDONESIA
# ==============================================================================
def format_rupiah(value):
    """Format nilai Rupiah dengan label yang benar dalam Bahasa Indonesia"""
    if value is None or pd.isna(value):
        return "Rp 0"
    
    value = float(value)
    
    if abs(value) >= 1e12:  # Triliun
        return f"Rp {value/1e12:,.2f} Triliun"
    elif abs(value) >= 1e9:  # Miliar
        return f"Rp {value/1e9:,.1f} Miliar"
    elif abs(value) >= 1e6:  # Juta
        return f"Rp {value/1e6:,.1f} Juta"
    elif abs(value) >= 1e3:  # Ribu
        return f"Rp {value:,.0f}"
    else:
        return f"Rp {value:,.0f}"

def format_usd(value):
    """Format nilai USD dengan label yang benar dalam Bahasa Indonesia"""
    if value is None or pd.isna(value):
        return "US$ 0"
    
    value = float(value)
    
    if abs(value) >= 1e9:  # Miliar
        return f"US$ {value/1e9:,.2f} Miliar"
    elif abs(value) >= 1e6:  # Juta
        return f"US$ {value/1e6:,.1f} Juta"
    else:
        return f"US$ {value:,.0f}"

def format_volume(value):
    """Format volume dengan label yang benar"""
    if value is None or pd.isna(value):
        return "0"
    
    value = float(value)
    
    if abs(value) >= 1e9:  # Miliar
        return f"{value/1e9:,.1f} Miliar"
    elif abs(value) >= 1e6:  # Juta
        return f"{value/1e6:,.1f} Juta"
    else:
        return f"{value:,.0f}"

def format_percentage(value):
    """Format persentase"""
    if value is None or pd.isna(value):
        return "0.00%"
    return f"{float(value):.2f}%"

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (via SERVICE ACCOUNT)
# ==============================================================================
def get_gdrive_service():
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except KeyError:
        msg = "❌ Gagal otentikasi: 'st.secrets' tidak menemukan key [gcp_service_account]. Pastikan 'secrets.toml' sudah benar."
        return None, msg
    except Exception as e:
        msg = f"❌ Gagal otentikasi Google Drive: {e}."
        return None, msg

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Mencari file transaksi, men-download, membersihkan, dan membacanya ke Pandas."""
    with st.spinner("🔄 Memuat data dari Google Drive..."):
        service, error_msg = get_gdrive_service()
        if error_msg:
            return pd.DataFrame(), error_msg, "error"

        try:
            query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
            results = service.files().list(
                q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1
            ).execute()
            items = results.get('files', [])

            if not items:
                msg = f"❌ File '{FILE_NAME}' tidak ditemukan di folder GDrive."
                return pd.DataFrame(), msg, "error"

            file_id = items[0]['id']
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)

            df = pd.read_csv(fh, dtype=object)

            df.columns = df.columns.str.strip()
            df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')

            # Daftar kolom numerik yang harus dibersihkan
            cols_to_numeric = [
                'High', 'Low', 'Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell',
                'Bid Volume', 'Offer Volume', 'Previous', 'Change', 'Open Price', 'First Trade',
                'Frequency', 'Index Individual', 'Offer', 'Bid', 'Listed Shares', 'Tradeble Shares',
                'Weight For Index', 'Non Regular Volume', 'Change %', 'Typical Price', 'TPxV',
                'VWMA_20D', 'MA20_vol', 'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow',
                'Bid/Offer Imbalance', 'Money Flow Value', 'Free Float', 'Money Flow Ratio (20D)'
            ]

            for col in cols_to_numeric:
                if col in df.columns:
                    cleaned_col = df[col].astype(str).str.strip()
                    cleaned_col = cleaned_col.str.replace(r'[,\sRp\%]', '', regex=True)
                    df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

            if 'Unusual Volume' in df.columns:
                df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true', 'True', 'TRUE'])
                df['Unusual Volume'] = df['Unusual Volume'].astype(bool)

            if 'Final Signal' in df.columns:
                df['Final Signal'] = df['Final Signal'].astype(str).str.strip()

            if 'Sector' in df.columns:
                 df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
            else:
                 df['Sector'] = 'Others'

            df = df.dropna(subset=['Last Trading Date', 'Stock Code'])

            # Hitung NFF (Rp) jika belum ada
            if 'NFF (Rp)' not in df.columns:
                 if 'Typical Price' in df.columns:
                     df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Typical Price']
                 else:
                     df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Close']

            # Hitung Market Cap jika ada data
            if 'Listed Shares' in df.columns:
                df['Market Cap (Rp)'] = df['Close'] * df['Listed Shares']
                df['Market Cap (T)'] = df['Market Cap (Rp)'] / 1e12
            
            # Hitung VWAP harian
            df['VWAP'] = np.where(df['Volume'] > 0, df['Value'] / df['Volume'], df['Close'])
            
            msg = f"✅ Data berhasil dimuat ({len(df):,} baris)"
            return df, msg, "success"

        except Exception as e:
            msg = f"❌ Terjadi error saat memuat data: {e}."
            return pd.DataFrame(), msg, "error"

# ==============================================================================
# 🛠️ 4) FUNGSI KALKULASI UTAMA
# ==============================================================================
def pct_rank(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0) * 100

def to_pct(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1: return pd.Series(50, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx: return pd.Series(50, index=s.index)
    return (s - mn) / (mx - mn) * 100

def calculate_potential_score(df: pd.DataFrame, latest_date: pd.Timestamp):
    """Menjalankan logika scoring 'Raport' pada data cutoff tanggal tertentu."""
    trend_start = latest_date - pd.Timedelta(days=30)
    mom_start = latest_date - pd.Timedelta(days=7)
    
    df_historic = df[df['Last Trading Date'] <= latest_date]
    trend_df = df_historic[df_historic['Last Trading Date'] >= trend_start].copy()
    mom_df = df_historic[df_historic['Last Trading Date'] >= mom_start].copy()
    last_df = df_historic[df_historic['Last Trading Date'] == latest_date].copy()

    if trend_df.empty or mom_df.empty or last_df.empty:
        msg = "Data tidak cukup."
        return pd.DataFrame(), msg, "warning"

    # 1. Trend Score (30 Hari)
    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close', 'last'), last_final_signal=('Final Signal', 'last'),
        total_net_ff_rp=('NFF (Rp)', 'sum'), total_money_flow=('Money Flow Value', 'sum'),
        avg_change_pct=('Change %', 'mean'), sector=('Sector', 'last')
    ).reset_index()
    score_akum = tr['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 75, 'Netral': 30, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(30)
    score_ff = pct_rank(tr['total_net_ff_rp'])
    score_mfv = pct_rank(tr['total_money_flow'])
    score_mom = pct_rank(tr['avg_change_pct'])
    tr['Trend Score'] = (score_akum * W['trend_akum'] + score_ff * W['trend_ff'] + score_mfv * W['trend_mfv'] + score_mom * W['trend_mom'])

    # 2. Momentum Score (7 Hari)
    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %', 'sum'), had_unusual_volume=('Unusual Volume', 'any'),
        last_final_signal=('Final Signal', 'last'), total_net_ff_rp=('NFF (Rp)', 'sum')
    ).reset_index()
    s_price = pct_rank(mo['total_change_pct'])
    s_vol = mo['had_unusual_volume'].map({True: 100, False: 20}).fillna(20)
    s_akum = mo['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 80, 'Netral': 40, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(40)
    s_ff7 = pct_rank(mo['total_net_ff_rp'])
    mo['Momentum Score'] = (s_price * W['mom_price'] + s_vol * W['mom_vol'] + s_akum * W['mom_akum'] + s_ff7 * W['mom_ff'])

    # 3. NBSA
    nbsa = trend_df.groupby('Stock Code').agg(total_net_ff_30d_rp=('NFF (Rp)', 'sum')).reset_index()
    
    # 4. Foreign Contrib
    if {'Foreign Buy', 'Foreign Sell', 'Value'}.issubset(df.columns):
        tmp = trend_df.copy()
        tmp['Foreign Value proxy'] = tmp['NFF (Rp)']
        contrib = tmp.groupby('Stock Code').agg(total_foreign_value_proxy=('Foreign Value proxy', 'sum'), total_value_30d=('Value', 'sum')).reset_index()
        contrib['foreign_contrib_pct'] = np.where(contrib['total_value_30d'] > 0, (contrib['total_foreign_value_proxy'].abs() / contrib['total_value_30d']) * 100, 0)
    else:
        contrib = pd.DataFrame({'Stock Code': [], 'foreign_contrib_pct': []})

    uv = last_df.set_index('Stock Code')['Unusual Volume'].map({True: 1, False: 0})

    # Merge
    rank = tr[['Stock Code', 'Trend Score', 'last_price', 'last_final_signal', 'sector']].merge(
        mo[['Stock Code', 'Momentum Score']], on='Stock Code', how='outer'
    ).merge(nbsa, on='Stock Code', how='left').merge(contrib[['Stock Code', 'foreign_contrib_pct']], on='Stock Code', how='left')
    
    rank['NBSA Score'] = to_pct(rank['total_net_ff_30d_rp'])
    rank['Foreign Contrib Score'] = to_pct(rank['foreign_contrib_pct'])
    unusual_bonus = uv.reindex(rank['Stock Code']).fillna(0) * 5
    
    rank['Potential Score'] = (
        rank['Trend Score'].fillna(0) * W['blend_trend'] +
        rank['Momentum Score'].fillna(0) * W['blend_mom'] +
        rank['NBSA Score'].fillna(50) * W['blend_nbsa'] +
        rank['Foreign Contrib Score'].fillna(50) * W['blend_fcontrib'] +
        unusual_bonus.values * W['blend_unusual']
    )

    top20 = rank.sort_values('Potential Score', ascending=False).head(20).copy()
    top20.insert(0, 'Analysis Date', latest_date.strftime('%Y-%m-%d'))
    return top20, "Skor berhasil dihitung.", "success"

@st.cache_data(ttl=3600)
def calculate_nff_top_stocks(df, max_date):
    periods = {'7D': 7, '30D': 30, '90D': 90, '180D': 180}; results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    
    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        nff_agg = df_period.groupby('Stock Code')['NFF (Rp)'].sum()
        df_agg = pd.DataFrame(nff_agg).join(latest_data.get('Close', pd.Series())).join(latest_data.get('Sector', pd.Series()))
        df_agg.columns = ['Total Net FF (Rp)', 'Harga Terakhir', 'Sector']
        results[name] = df_agg.sort_values(by='Total Net FF (Rp)', ascending=False).reset_index()
    return results['7D'], results['30D'], results['90D'], results['180D']

@st.cache_data(ttl=3600)
def calculate_mfv_top_stocks(df, max_date):
    periods = {'7D': 7, '30D': 30, '90D': 90, '180D': 180}; results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    
    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        mfv_agg = df_period.groupby('Stock Code')['Money Flow Value'].sum()
        df_agg = pd.DataFrame(mfv_agg).join(latest_data.get('Close', pd.Series())).join(latest_data.get('Sector', pd.Series()))
        df_agg.columns = ['Total Money Flow (Rp)', 'Harga Terakhir', 'Sector']
        results[name] = df_agg.sort_values(by='Total Money Flow (Rp)', ascending=False).reset_index()
    return results['7D'], results['30D'], results['90D'], results['180D']

def run_backtest_analysis(df, days_back=90):
    all_dates = sorted(df['Last Trading Date'].unique())
    if len(all_dates) < days_back: days_back = len(all_dates) - 30 
    start_idx = max(30, len(all_dates) - days_back)
    simulation_dates = all_dates[start_idx:]
    latest_prices = df[df['Last Trading Date'] == all_dates[-1]].set_index('Stock Code')['Close']
    
    backtest_log = []
    progress_bar = st.progress(0); status_text = st.empty(); total_steps = len(simulation_dates)
    
    for i, sim_date in enumerate(simulation_dates):
        pct = (i + 1) / total_steps; progress_bar.progress(pct)
        sim_date_ts = pd.Timestamp(sim_date)
        status_text.text(f"⏳ Mengaudit data tanggal: {sim_date_ts.strftime('%d-%m-%Y')}...")
        try:
            top20, _, status = calculate_potential_score(df, sim_date_ts)
            if status == "success" and not top20.empty:
                for idx, row in top20.iterrows():
                    code = row['Stock Code']
                    entry_price = row['last_price']
                    curr_price = latest_prices.get(code, np.nan)
                    ret_pct = ((curr_price - entry_price) / entry_price * 100) if (pd.notna(curr_price) and entry_price > 0) else 0
                    backtest_log.append({'Signal Date': sim_date_ts, 'Stock Code': code, 'Entry Price': entry_price, 'Current Price': curr_price, 'Return to Date (%)': ret_pct, 'Score at Signal': row['Potential Score']})
        except Exception: continue
    progress_bar.empty(); status_text.empty()
    return pd.DataFrame(backtest_log)

def simulate_portfolio_range(df, capital, start_date_ts, end_date_ts):
    top20, _, status = calculate_potential_score(df, start_date_ts)
    if status != "success" or top20.empty: return pd.DataFrame(), None, "Gagal."
    
    df_end = df[df['Last Trading Date'] == end_date_ts]
    if df_end.empty: return pd.DataFrame(), None, "Data End Date kosong."
    
    exit_prices = df_end.set_index('Stock Code')['Close']
    allocation_per_stock = capital / len(top20)
    portfolio_results = []
    
    for idx, row in top20.iterrows():
        code = row['Stock Code']; buy_price = row['last_price']
        exit_price = exit_prices.get(code, np.nan)
        if pd.isna(exit_price) or buy_price <= 0:
            roi_pct = 0; final_val = allocation_per_stock; gain_rp = 0; exit_price_display = 0
        else:
            roi_pct = ((exit_price - buy_price) / buy_price)
            final_val = allocation_per_stock * (1 + roi_pct)
            gain_rp = final_val - allocation_per_stock
            exit_price_display = exit_price
        portfolio_results.append({'Stock Code': code, 'Sector': row['sector'], 'Buy Price': buy_price, 'Sell Price': exit_price_display, 'Gain/Loss (Rp)': gain_rp, 'Final Value': final_val, 'ROI (%)': roi_pct * 100})
        
    df_port = pd.DataFrame(portfolio_results)
    summary = {'Start Date': start_date_ts, 'End Date': end_date_ts, 'Initial Capital': capital, 'Final Portfolio Value': df_port['Final Value'].sum(), 'Net Profit': df_port['Gain/Loss (Rp)'].sum(), 'Total ROI': (df_port['Gain/Loss (Rp)'].sum() / capital) * 100}
    return df_port, summary, "success"

@st.cache_data(ttl=3600)
def calculate_msci_projection_v2(df, latest_date, usd_rate):
    """
    Menghitung metrik proxy MSCI: 
    1. Full & Float Market Cap (IDR & USD)
    2. Liquidity (ATVR) 3-Month & 12-Month
    """
    # Define time windows
    start_date_12m = latest_date - pd.Timedelta(days=365)
    start_date_3m = latest_date - pd.Timedelta(days=90)
    
    # Filter data slices
    df_12m = df[(df['Last Trading Date'] >= start_date_12m) & (df['Last Trading Date'] <= latest_date)]
    df_3m = df[(df['Last Trading Date'] >= start_date_3m) & (df['Last Trading Date'] <= latest_date)]
    df_last = df[df['Last Trading Date'] == latest_date].copy()
    
    results = []
    
    for idx, row in df_last.iterrows():
        code = row['Stock Code']; close = row['Close']
        listed_shares = row.get('Listed Shares', 0)
        free_float_pct = row.get('Free Float', 0)
        
        # --- 1. SIZE (Market Cap) ---
        # IDR Base
        full_mcap_idr_t = (close * listed_shares) / 1e12 # Triliun
        float_mcap_idr_t = full_mcap_idr_t * (free_float_pct / 100)
        
        # USD Conversion (Miliar)
        full_mcap_usd_miliar = (full_mcap_idr_t * 1e12) / usd_rate / 1e9
        float_mcap_usd_miliar = (float_mcap_idr_t * 1e12) / usd_rate / 1e9
        
        # --- 2. LIQUIDITY (ATVR) ---
        # Total Value Transaksi (IDR)
        val_12m = df_12m[df_12m['Stock Code'] == code]['Value'].sum()
        val_3m = df_3m[df_3m['Stock Code'] == code]['Value'].sum()
        
        # Annualized Value
        annualized_val_3m = val_3m * 4 # Simple annualization (3 bulan x 4)
        
        # Float Mcap (Current as proxy for average) -> Dalam Rupiah Penuh
        float_mcap_full = float_mcap_idr_t * 1e12
        
        # ATVR Calculation (%)
        atvr_12m = (val_12m / float_mcap_full * 100) if float_mcap_full > 0 else 0
        atvr_3m = (annualized_val_3m / float_mcap_full * 100) if float_mcap_full > 0 else 0
            
        results.append({
            'Stock Code': code, 
            'Close': close, 
            'Sector': row['Sector'],
            # IDR Metrics
            'Float Cap (IDR Triliun)': float_mcap_idr_t,
            # USD Metrics (BAHASA INDONESIA)
            'Full Cap (US$ Miliar)': full_mcap_usd_miliar,
            'Float Cap (US$ Miliar)': float_mcap_usd_miliar,
            # Liquidity (BAHASA INDONESIA)
            'ATVR 12 Bulan (%)': atvr_12m,
            'ATVR 3 Bulan (%)': atvr_3m
        })
        
    df_msci = pd.DataFrame(results)
    # Default Sort by USD Float Cap
    df_msci = df_msci.sort_values(by='Float Cap (US$ Miliar)', ascending=False).reset_index(drop=True)
    df_msci['Rank'] = df_msci.index + 1
    return df_msci

# ==============================================================================
# 🆕 FUNGSI TAMBAHAN UNTUK FITUR BARU
# ==============================================================================
def create_market_overview(df_day):
    """Membuat ringkasan pasar yang lebih komprehensif"""
    overview = {
        'total_stocks': len(df_day),
        'total_value': df_day['Value'].sum(),
        'avg_change': df_day['Change %'].mean(),
        'stocks_up': len(df_day[df_day['Change %'] > 0]),
        'stocks_down': len(df_day[df_day['Change %'] < 0]),
        'total_volume': df_day['Volume'].sum(),
        'foreign_net': df_day['NFF (Rp)'].sum(),
        'unusual_volume': df_day['Unusual Volume'].sum(),
        'total_mfv': df_day['Money Flow Value'].sum()
    }
    return overview

def calculate_sector_performance(df_day):
    """Analisis performa per sektor"""
    sector_df = df_day.groupby('Sector').agg({
        'Stock Code': 'count',
        'Change %': 'mean',
        'Value': 'sum',
        'NFF (Rp)': 'sum',
        'Money Flow Value': 'sum'
    }).round(2)
    
    sector_df = sector_df.rename(columns={
        'Stock Code': 'Jumlah Saham',
        'Change %': 'Avg Change %',
        'Value': 'Total Value (Rp)',
        'NFF (Rp)': 'Total NFF (Rp)',
        'Money Flow Value': 'Total MFV (Rp)'
    })
    
    return sector_df.sort_values('Avg Change %', ascending=False)

def create_stock_screener(df_day, filters):
    """Stock screener dengan berbagai kriteria"""
    filtered = df_day.copy()
    
    if filters.get('min_price'):
        filtered = filtered[filtered['Close'] >= filters['min_price']]
    if filters.get('max_price'):
        filtered = filtered[filtered['Close'] <= filters['max_price']]
    if filters.get('min_volume'):
        filtered = filtered[filtered['Volume'] >= filters['min_volume']]
    if filters.get('min_mcap'):
        filtered = filtered[filtered.get('Market Cap (Rp)', 0) >= filters['min_mcap']]
    if filters.get('signal'):
        filtered = filtered[filtered['Final Signal'] == filters['signal']]
    if filters.get('sectors'):
        filtered = filtered[filtered['Sector'].isin(filters['sectors'])]
    
    return filtered.sort_values('Change %', ascending=False)

# ==============================================================================
# 💎 5) LAYOUT UTAMA YANG DITINGKATKAN
# ==============================================================================

# ---- HEADER DENGAN GRADIENT ----
st.markdown("""
<div class="header-gradient">
    <h1 style="margin:0; font-size:2.5rem;">📊 Quantum IDX Dashboard</h1>
    <p style="margin:0; opacity:0.9; font-size:1.1rem;">Advanced Analytics, Backtesting, & Portfolio Simulator</p>
    <div style="margin-top:1rem; display:flex; gap:1rem; flex-wrap:wrap;">
        <div style="background:rgba(255,255,255,0.2); padding:0.5rem 1rem; border-radius:20px;">Real-time Data</div>
        <div style="background:rgba(255,255,255,0.2); padding:0.5rem 1rem; border-radius:20px;">MSCI Simulator</div>
        <div style="background:rgba(255,255,255,0.2); padding:0.5rem 1rem; border-radius:20px;">Portfolio Analytics</div>
        <div style="background:rgba(255,255,255,0.2); padding:0.5rem 1rem; border-radius:20px;">Backtesting Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- LOAD DATA DENGAN STATUS BAR ----
df, status_msg, status_level = load_data()

if status_level == "success":
    st.toast("✅ Data berhasil dimuat!", icon="✅")
    st.sidebar.success(f"Data terakhir: {df['Last Trading Date'].max().strftime('%d %b %Y')}")
elif status_level == "error":
    st.error(status_msg)
    st.stop()

# ---- SIDEBAR MODERN ----
with st.sidebar:
    st.markdown("### 🎛️ Kontrol Dashboard")
    
    # Refresh dengan animasi
    col_refresh1, col_refresh2 = st.columns([3, 1])
    with col_refresh1:
        if st.button("🔄 **Refresh Data**", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_refresh2:
        st.markdown("")
    
    st.markdown("---")
    
    # Date picker dengan styling
    st.markdown("#### 📅 Kalender Analisis")
    max_date = df['Last Trading Date'].max().date()
    selected_date = st.date_input(
        "Pilih Tanggal",
        max_date,
        min_value=df['Last Trading Date'].min().date(),
        max_value=max_date,
        format="DD/MM/YYYY",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick filters
    st.markdown("#### 🔍 Filter Cepat")
    
    # Sector performance preview
    sectors = sorted(df.get("Sector", pd.Series(dtype='object')).dropna().unique())
    selected_sectors_filter = st.multiselect(
        "Pilih Sektor",
        sectors,
        default=sectors[:3] if len(sectors) > 3 else sectors,
        help="Filter berdasarkan sektor perusahaan"
    )
    
    # Price range filter
    st.markdown("#### 💰 Rentang Harga")
    min_price, max_price = st.slider(
        "Range Harga (Rp)",
        min_value=float(df['Close'].min()),
        max_value=float(df['Close'].max()),
        value=(float(df['Close'].quantile(0.1)), float(df['Close'].quantile(0.9))),
        step=100.0
    )
    
    st.markdown("---")
    
    # Dashboard info
    st.markdown("#### 📊 Statistik")
    st.info(f"""
    **Data Summary:**
    - {len(df['Stock Code'].unique()):,} Saham
    - Periode: {df['Last Trading Date'].min().strftime('%d %b %Y')} - {df['Last Trading Date'].max().strftime('%d %b %Y')}
    - Total Baris: {len(df):,}
    """)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("#### ⚡ Aksi Cepat")
    if st.button("📥 Ekspor Data Hari Ini", use_container_width=True):
        csv = df[df['Last Trading Date'].dt.date == selected_date].to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"idx_data_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---- MAIN CONTENT ----
# Filter data untuk tanggal yang dipilih
df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()

# ---- TAB NAVIGATION MODERN ----
tab_titles = [
    "🏠 **Dashboard**", 
    "📈 **Analisis Saham**", 
    "🔍 **Stock Screener**",
    "🏆 **Top 20 Potensial**", 
    "🌊 **Foreign Flow**", 
    "💰 **Money Flow**",
    "🧪 **Backtesting**", 
    "💼 **Portfolio Simulator**", 
    "🌏 **MSCI Simulator**",
    "📊 **Analisis Sektor**"
]

tabs = st.tabs(tab_titles)

# ==============================================================================
# TAB 1: DASHBOARD UTAMA YANG DITINGKATKAN
# ==============================================================================
with tabs[0]:
    st.markdown("## 📊 Dashboard Ringkasan Pasar")
    
    # Market Overview Cards
    overview = create_market_overview(df_day)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Total Saham Aktif</div>
            <div style="font-size:1.8rem; font-weight:700; color:#4361ee;">{overview['total_stocks']:,}</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">
                <span style="color:#2ecc71;">↑ {overview['stocks_up']:,}</span> | 
                <span style="color:#e74c3c;">↓ {overview['stocks_down']:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Nilai Transaksi</div>
            <div style="font-size:1.8rem; font-weight:700; color:#4361ee;">{format_rupiah(overview['total_value'])}</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">Volume: {format_volume(overview['total_volume'])}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color_change = "#2ecc71" if overview['avg_change'] > 0 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Rata² Perubahan</div>
            <div style="font-size:1.8rem; font-weight:700; color:{color_change};">{overview['avg_change']:.2f}%</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">Foreign Net: {format_rupiah(overview['foreign_net'])}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Unusual Volume</div>
            <div style="font-size:1.8rem; font-weight:700; color:#f8961e;">{overview['unusual_volume']:,}</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">MFV Total: {format_rupiah(overview['total_mfv'])}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 📈 Top Gainers & Losers")
        
        # Gainers
        gainers = df_day.nlargest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Volume']]
        losers = df_day.nsmallest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Volume']]
        
        tab_gain, tab_loss = st.tabs(["🏆 Gainers", "📉 Losers"])
        
        with tab_gain:
            fig_gain = px.bar(
                gainers, 
                x='Stock Code', 
                y='Change %',
                color='Change %',
                color_continuous_scale=['#2ecc71', '#27ae60'],
                title=""
            )
            fig_gain.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_gain, use_container_width=True)
            
            st.dataframe(
                gainers.assign(
                    Close=gainers['Close'].apply(lambda x: f"Rp {x:,.0f}"),
                    Change=gainers['Change %'].apply(lambda x: f"{x:+.2f}%"),
                    Volume=gainers['Volume'].apply(lambda x: f"{format_volume(x)}")
                )[['Stock Code', 'Close', 'Change', 'Volume']],
                hide_index=True,
                use_container_width=True
            )
        
        with tab_loss:
            fig_loss = px.bar(
                losers, 
                x='Stock Code', 
                y='Change %',
                color='Change %',
                color_continuous_scale=['#e74c3c', '#c0392b'],
                title=""
            )
            fig_loss.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_loss, use_container_width=True)
            
            st.dataframe(
                losers.assign(
                    Close=losers['Close'].apply(lambda x: f"Rp {x:,.0f}"),
                    Change=losers['Change %'].apply(lambda x: f"{x:+.2f}%"),
                    Volume=losers['Volume'].apply(lambda x: f"{format_volume(x)}")
                )[['Stock Code', 'Close', 'Change', 'Volume']],
                hide_index=True,
                use_container_width=True
            )
    
    with col_chart2:
        st.markdown("### 💰 Top Value Transactions")
        
        high_value = df_day.nlargest(10, 'Value')[['Stock Code', 'Close', 'Value', 'NFF (Rp)']]
        
        fig_value = px.scatter(
            high_value,
            x='Stock Code',
            y='Value',
            size='Value',
            color='NFF (Rp)',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            hover_data=['Close'],
            title=""
        )
        fig_value.update_layout(height=400)
        st.plotly_chart(fig_value, use_container_width=True)
        
        # Foreign activity
        st.markdown("#### 🌍 Aktivitas Asing")
        foreign_agg = df_day.groupby('Sector')['NFF (Rp)'].sum().reset_index()
        fig_foreign = px.bar(
            foreign_agg,
            x='Sector',
            y='NFF (Rp)',
            color='NFF (Rp)',
            color_continuous_scale=px.colors.diverging.RdYlGn
        )
        fig_foreign.update_layout(height=250)
        st.plotly_chart(fig_foreign, use_container_width=True)

# ==============================================================================
# TAB 2: ANALISIS INDIVIDUAL YANG DITINGKATKAN
# ==============================================================================
with tabs[1]:
    st.markdown("## 📈 Analisis Saham Individual")
    
    col_sel1, col_sel2 = st.columns([2, 1])
    
    with col_sel1:
        all_stocks = sorted(df["Stock Code"].unique())
        stock = st.selectbox(
            "Pilih Saham untuk Analisis",
            all_stocks,
            index=all_stocks.index("BBRI") if "BBRI" in all_stocks else 0,
            help="Pilih kode saham untuk melihat analisis mendalam"
        )
    
    with col_sel2:
        analysis_period = st.selectbox(
            "Periode Analisis",
            ["30 Hari", "90 Hari", "1 Tahun", "Semua Data"],
            index=1
        )
    
    if stock:
        # Filter data berdasarkan periode
        if analysis_period == "30 Hari":
            days_back = 30
        elif analysis_period == "90 Hari":
            days_back = 90
        elif analysis_period == "1 Tahun":
            days_back = 365
        else:
            days_back = None
        
        cutoff_date = pd.Timestamp(selected_date)
        if days_back:
            start_date = cutoff_date - pd.Timedelta(days=days_back)
            df_stock = df[(df['Stock Code'] == stock) & 
                         (df['Last Trading Date'] >= start_date) & 
                         (df['Last Trading Date'] <= cutoff_date)].sort_values('Last Trading Date')
        else:
            df_stock = df[df['Stock Code'] == stock].sort_values('Last Trading Date')
        
        if not df_stock.empty:
            latest = df_stock.iloc[-1]
            
            # Metrics Row
            st.markdown("### 📊 Metrics Utama")
            m1, m2, m3, m4, m5 = st.columns(5)
            
            price_change = latest['Change %']
            price_color = "#2ecc71" if price_change > 0 else "#e74c3c"
            
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">Harga</div>
                    <div style="font-size:1.5rem; font-weight:700;">Rp {latest['Close']:,.0f}</div>
                    <div style="font-size:0.8rem; color:{price_color};">{price_change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m2:
                nff_value = latest['NFF (Rp)']
                nff_color = "#2ecc71" if nff_value > 0 else "#e74c3c"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">NFF Hari Ini</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{nff_color};">{format_rupiah(nff_value)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m3:
                mfv_value = latest.get('Money Flow Value', 0)
                mfv_color = "#2ecc71" if mfv_value > 0 else "#e74c3c"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">Money Flow</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{mfv_color};">{format_rupiah(mfv_value)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m4:
                signal = latest.get('Final Signal', 'Netral')
                signal_color = {
                    'Strong Akumulasi': '#2ecc71',
                    'Akumulasi': '#27ae60',
                    'Netral': '#95a5a6',
                    'Distribusi': '#e74c3c',
                    'Strong Distribusi': '#c0392b'
                }.get(signal, '#95a5a6')
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">Signal</div>
                    <div style="font-size:1.3rem; font-weight:700; color:{signal_color};">{signal}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m5:
                mcap = latest.get('Market Cap (T)', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">Market Cap</div>
                    <div style="font-size:1.5rem; font-weight:700;">Rp {mcap:.2f} Triliun</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Advanced Chart
            st.markdown("### 📈 Chart Analisis Teknikal")
            
            # Buat subplot yang lebih kompleks
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
            )
            
            # Harga dengan candlestick jika ada data
            if all(col in df_stock.columns for col in ['Open Price', 'High', 'Low', 'Close']):
                fig.add_trace(
                    go.Candlestick(
                        x=df_stock['Last Trading Date'],
                        open=df_stock['Open Price'],
                        high=df_stock['High'],
                        low=df_stock['Low'],
                        close=df_stock['Close'],
                        name='OHLC'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_stock['Last Trading Date'],
                        y=df_stock['Close'],
                        name='Close Price',
                        line=dict(color='#4361ee', width=2)
                    ),
                    row=1, col=1
                )
            
            # Volume dengan warna berdasarkan NFF
            colors_volume = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_stock['NFF (Rp)']]
            fig.add_trace(
                go.Bar(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['Volume'],
                    name='Volume',
                    marker_color=colors_volume,
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # NFF
            fig.add_trace(
                go.Bar(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['NFF (Rp)'],
                    name='Net Foreign Flow',
                    marker_color=colors_volume
                ),
                row=3, col=1
            )
            
            # Money Flow Value
            colors_mfv = ['#3498db' if x > 0 else '#e74c3c' for x in df_stock.get('Money Flow Value', 0)]
            fig.add_trace(
                go.Bar(
                    x=df_stock['Last Trading Date'],
                    y=df_stock.get('Money Flow Value', 0),
                    name='Money Flow Value',
                    marker_color=colors_mfv
                ),
                row=4, col=1
            )
            
            fig.update_layout(
                height=800,
                title=f"Analisis Komprehensif: {stock}",
                hovermode="x unified",
                showlegend=True,
                template="plotly_white"
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistik tambahan
            st.markdown("### 📊 Statistik Performa")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Avg Daily Change", f"{df_stock['Change %'].mean():.2f}%")
                st.metric("Max Daily Gain", f"{df_stock['Change %'].max():.2f}%")
            
            with col_stat2:
                st.metric("Total NFF Period", format_rupiah(df_stock['NFF (Rp)'].sum()))
                st.metric("Avg Daily Volume", format_volume(df_stock['Volume'].mean()))
            
            with col_stat3:
                st.metric("Volatility (Std Dev)", f"{df_stock['Change %'].std():.2f}%")
                st.metric("Days Analyzed", len(df_stock))
            
        else:
            st.warning(f"Tidak ada data untuk {stock} pada periode yang dipilih.")

# ==============================================================================
# TAB 3: STOCK SCREENER BARU
# ==============================================================================
with tabs[2]:
    st.markdown("## 🔍 Advanced Stock Screener")
    
    # Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        price_min = st.number_input("Harga Minimum (Rp)", 
                                   min_value=0, 
                                   max_value=int(df_day['Close'].max()), 
                                   value=0)
        price_max = st.number_input("Harga Maksimum (Rp)", 
                                   min_value=0, 
                                   max_value=int(df_day['Close'].max()), 
                                   value=int(df_day['Close'].max()))
    
    with col_f2:
        volume_min = st.number_input("Volume Minimum", 
                                    min_value=0, 
                                    max_value=int(df_day['Volume'].max()), 
                                    value=0,
                                    step=1000000)
        change_min = st.number_input("Perubahan Minimum (%)", 
                                    min_value=-100.0, 
                                    max_value=100.0, 
                                    value=-100.0,
                                    step=0.5)
    
    with col_f3:
        change_max = st.number_input("Perubahan Maksimum (%)", 
                                    min_value=-100.0, 
                                    max_value=100.0, 
                                    value=100.0,
                                    step=0.5)
        signal_filter = st.selectbox("Signal", 
                                    ['Semua', 'Strong Akumulasi', 'Akumulasi', 'Netral', 'Distribusi', 'Strong Distribusi'])
    
    # Sektor filter
    selected_sectors = st.multiselect(
        "Filter Sektor",
        options=sorted(df_day['Sector'].unique()),
        default=[]
    )
    
    # Apply filters
    filtered_stocks = df_day.copy()
    
    if price_min > 0:
        filtered_stocks = filtered_stocks[filtered_stocks['Close'] >= price_min]
    if price_max < df_day['Close'].max():
        filtered_stocks = filtered_stocks[filtered_stocks['Close'] <= price_max]
    if volume_min > 0:
        filtered_stocks = filtered_stocks[filtered_stocks['Volume'] >= volume_min]
    if change_min > -100:
        filtered_stocks = filtered_stocks[filtered_stocks['Change %'] >= change_min]
    if change_max < 100:
        filtered_stocks = filtered_stocks[filtered_stocks['Change %'] <= change_max]
    if signal_filter != 'Semua':
        filtered_stocks = filtered_stocks[filtered_stocks['Final Signal'] == signal_filter]
    if selected_sectors:
        filtered_stocks = filtered_stocks[filtered_stocks['Sector'].isin(selected_sectors)]
    
    # Display results
    st.markdown(f"### 📋 Hasil Screener: {len(filtered_stocks)} Saham")
    
    if not filtered_stocks.empty:
        # Tampilkan dengan format yang lebih baik
        display_cols = ['Stock Code', 'Sector', 'Close', 'Change %', 'Volume', 'Value', 'NFF (Rp)', 'Final Signal']
        
        formatted_df = filtered_stocks[display_cols].copy()
        formatted_df['Close'] = formatted_df['Close'].apply(lambda x: f"Rp {x:,.0f}")
        formatted_df['Change %'] = formatted_df['Change %'].apply(lambda x: f"{x:+.2f}%")
        formatted_df['Volume'] = formatted_df['Volume'].apply(lambda x: f"{format_volume(x)}")
        formatted_df['Value'] = formatted_df['Value'].apply(lambda x: f"{format_rupiah(x)}")
        formatted_df['NFF (Rp)'] = formatted_df['NFF (Rp)'].apply(lambda x: f"{format_rupiah(x)}")
        
        st.dataframe(
            formatted_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Stock Code": st.column_config.TextColumn("Kode", width="small"),
                "Sector": st.column_config.TextColumn("Sektor"),
                "Close": st.column_config.TextColumn("Harga"),
                "Change %": st.column_config.TextColumn("Perubahan"),
                "Volume": st.column_config.TextColumn("Volume"),
                "Value": st.column_config.TextColumn("Nilai"),
                "NFF (Rp)": st.column_config.TextColumn("NFF"),
                "Final Signal": st.column_config.TextColumn("Signal")
            }
        )
        
        # Export option
        csv = filtered_stocks.to_csv(index=False)
        st.download_button(
            label="📥 Ekspor Hasil Screener",
            data=csv,
            file_name=f"stock_screener_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Tidak ada saham yang memenuhi kriteria filter.")

# ==============================================================================
# TAB 4: TOP 20 POTENSIAL (DIPERBAIKI)
# ==============================================================================
with tabs[3]:
    st.markdown("## 🏆 Top 20 Saham Potensial")
    st.info("Berdasarkan algoritma scoring yang menggabungkan trend, momentum, dan faktor fundamental.")
    
    @st.cache_data(ttl=3600)
    def get_cached_top20(dframe, tgl):
        return calculate_potential_score(dframe, tgl)
    
    df_top20, msg, status = get_cached_top20(df, pd.Timestamp(selected_date))
    
    if status == "success":
        # Visualisasi distribusi skor
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            fig_dist = px.histogram(
                df_top20,
                x='Potential Score',
                nbins=10,
                title='Distribusi Skor Top 20',
                color_discrete_sequence=['#4361ee']
            )
            fig_dist.update_layout(height=300)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col_viz2:
            sector_counts = df_top20['sector'].value_counts()
            fig_sector = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                title='Komposisi Sektor',
                hole=0.4
            )
            fig_sector.update_layout(height=300)
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # Tabel dengan formatting yang lebih baik
        st.markdown("### 📋 Ranking Saham")
        
        formatted_top20 = df_top20.copy()
        formatted_top20['last_price'] = formatted_top20['last_price'].apply(lambda x: f"Rp {x:,.0f}")
        formatted_top20['Trend Score'] = formatted_top20['Trend Score'].round(1)
        formatted_top20['Momentum Score'] = formatted_top20['Momentum Score'].round(1)
        formatted_top20['Potential Score'] = formatted_top20['Potential Score'].round(1)
        
        st.dataframe(
            formatted_top20[['Stock Code', 'sector', 'last_price', 'Trend Score', 'Momentum Score', 'Potential Score']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Stock Code": st.column_config.TextColumn("Kode", width="small"),
                "sector": st.column_config.TextColumn("Sektor"),
                "last_price": st.column_config.TextColumn("Harga"),
                "Trend Score": st.column_config.ProgressColumn(
                    "Trend Score",
                    format="%.1f",
                    min_value=0,
                    max_value=100
                ),
                "Momentum Score": st.column_config.ProgressColumn(
                    "Momentum Score",
                    format="%.1f",
                    min_value=0,
                    max_value=100
                ),
                "Potential Score": st.column_config.ProgressColumn(
                    "Skor Akhir",
                    format="%.1f",
                    min_value=0,
                    max_value=100
                )
            }
        )
        
        # Detail scoring
        with st.expander("📊 Detail Metrik Scoring"):
            st.markdown("""
            **Komponen Skor:**
            
            **Trend Score (35%):**
            - Akumulasi/Distribusi Signal: 40%
            - Net Foreign Flow 30-hari: 30%
            - Money Flow Value 30-hari: 20%
            - Momentum Harga 30-hari: 10%
            
            **Momentum Score (35%):**
            - Perubahan Harga 7-hari: 40%
            - Unusual Volume: 25%
            - Signal Akumulasi: 25%
            - Foreign Flow 7-hari: 10%
            
            **Additional Factors (30%):**
            - NBSA (Net Buy Sell Analysis): 20%
            - Foreign Contribution: 5%
            - Unusual Volume Bonus: 5%
            """)
            
            # Show raw data
            st.dataframe(df_top20, use_container_width=True)
    
    else:
        st.warning(msg)

# ==============================================================================
# TAB 5: FOREIGN FLOW (DIPERBAIKI - ALL TIME DATA + PERSENTASE DOMINASI)
# ==============================================================================
with tabs[4]:
    st.markdown("## 🌊 Analisis Net Foreign Flow Komprehensif")
    
    # ---- 1. TENTUKAN PERIODE DATA ----
    min_date = df['Last Trading Date'].min()
    max_date = df['Last Trading Date'].max()
    
    st.markdown(f"### 📅 Periode Data: {min_date.strftime('%d %b %Y')} - {max_date.strftime('%d %b %Y')}")
    
    # ---- 2. GRAFIK KUMULATIF FOREIGN FLOW (ALL TIME) ----
    st.markdown("### 📈 Grafik Kumulatif Foreign Flow (All Time)")
    
    # Hitung kumulatif harian
    df_daily_nff = df.groupby('Last Trading Date').agg({
        'NFF (Rp)': 'sum',
        'Foreign Buy': 'sum',
        'Foreign Sell': 'sum',
        'Value': 'sum'  # Untuk menghitung persentase
    }).reset_index().sort_values('Last Trading Date')
    
    # Hitung kumulatif
    df_daily_nff['NFF Kumulatif'] = df_daily_nff['NFF (Rp)'].cumsum()
    df_daily_nff['Foreign Buy Kumulatif'] = df_daily_nff['Foreign Buy'].cumsum()
    df_daily_nff['Foreign Sell Kumulatif'] = df_daily_nff['Foreign Sell'].cumsum()
    
    # Hitung persentase foreign flow terhadap total value
    df_daily_nff['NFF % of Total Value'] = np.where(
        df_daily_nff['Value'] > 0,
        (df_daily_nff['NFF (Rp)'].abs() / df_daily_nff['Value']) * 100,
        0
    )
    
    # Buat chart dengan 2 subplot
    fig_cumulative = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Kumulatif Net Foreign Flow (Rp)', 'Flow Harian (Rp)', 'Persentase NFF vs Total Value'),
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Line chart kumulatif
    fig_cumulative.add_trace(
        go.Scatter(
            x=df_daily_nff['Last Trading Date'],
            y=df_daily_nff['NFF Kumulatif'],
            mode='lines',
            name='NFF Kumulatif',
            line=dict(color='#2ecc71', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.2)',
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Kumulatif: %{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Bar chart harian dengan warna berdasarkan positif/negatif
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df_daily_nff['NFF (Rp)']]
    
    fig_cumulative.add_trace(
        go.Bar(
            x=df_daily_nff['Last Trading Date'],
            y=df_daily_nff['NFF (Rp)'],
            name='NFF Harian',
            marker_color=colors,
            opacity=0.7,
            hovertemplate='<b>%{x|%d %b %Y}</b><br>NFF Harian: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Line chart persentase
    fig_cumulative.add_trace(
        go.Scatter(
            x=df_daily_nff['Last Trading Date'],
            y=df_daily_nff['NFF % of Total Value'],
            mode='lines',
            name='% NFF/Total Value',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)',
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Persentase: %{y:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig_cumulative.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        title_font_size=16
    )
    
    fig_cumulative.update_xaxes(title_text="Tanggal", row=3, col=1)
    fig_cumulative.update_yaxes(title_text="Kumulatif (Rp)", row=1, col=1)
    fig_cumulative.update_yaxes(title_text="Harian (Rp)", row=2, col=1)
    fig_cumulative.update_yaxes(title_text="Persentase (%)", row=3, col=1)
    
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # ---- 3. METRIK SUMMARY ----
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    
    total_nff = df_daily_nff['NFF (Rp)'].sum()
    avg_daily_nff = df_daily_nff['NFF (Rp)'].mean()
    positive_days = len(df_daily_nff[df_daily_nff['NFF (Rp)'] > 0])
    total_days = len(df_daily_nff)
    positive_percentage = (positive_days / total_days * 100) if total_days > 0 else 0
    
    # Hitung total value untuk persentase
    total_value_all_time = df['Value'].sum()
    nff_percentage_of_total = (abs(total_nff) / total_value_all_time * 100) if total_value_all_time > 0 else 0
    
    with col_sum1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.8rem; color:#6c757d;">Total NFF All Time</div>
            <div style="font-size:1.5rem; font-weight:700; color:{'#2ecc71' if total_nff > 0 else '#e74c3c'};">{format_rupiah(total_nff)}</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">{nff_percentage_of_total:.2f}% dari total value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sum2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.8rem; color:#6c757d;">Rata² Harian</div>
            <div style="font-size:1.5rem; font-weight:700;">{format_rupiah(avg_daily_nff)}</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">{total_days} hari trading</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sum3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.8rem; color:#6c757d;">Hari Positif</div>
            <div style="font-size:1.5rem; font-weight:700; color:#2ecc71;">{positive_days}/{total_days}</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">{positive_percentage:.1f}% hari positif</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sum4:
        current_cumulative = df_daily_nff['NFF Kumulatif'].iloc[-1] if not df_daily_nff.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.8rem; color:#6c757d;">Kumulatif Saat Ini</div>
            <div style="font-size:1.5rem; font-weight:700; color={'#2ecc71' if current_cumulative > 0 else '#e74c3c'};">{format_rupiah(current_cumulative)}</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">Sampai {max_date.strftime('%d %b %Y')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---- 4. ANALISIS DOMINASI TRANSAKSI ASING PER SAHAM ----
    st.markdown("### 📊 Dominasi Transaksi Asing per Saham (All Time)")
    st.info("Analisis ini menunjukkan seberapa besar pengaruh asing terhadap aktivitas trading suatu saham.")
    
    # Agregasi per saham
    df_stock_nff = df.groupby('Stock Code').agg({
        'NFF (Rp)': 'sum',
        'Foreign Buy': 'sum',
        'Foreign Sell': 'sum',
        'Value': 'sum',
        'Volume': 'sum',
        'Close': 'last',
        'Sector': 'last'
    }).reset_index()
    
    # Hitung metrik-metrik penting
    df_stock_nff['Total Foreign Activity (Rp)'] = df_stock_nff['Foreign Buy'] + df_stock_nff['Foreign Sell']
    df_stock_nff['Persentase Foreign vs Total Value'] = np.where(
        df_stock_nff['Value'] > 0,
        (df_stock_nff['Total Foreign Activity (Rp)'] / df_stock_nff['Value']) * 100,
        0
    )
    df_stock_nff['Persentase NFF vs Total Value'] = np.where(
        df_stock_nff['Value'] > 0,
        (df_stock_nff['NFF (Rp)'].abs() / df_stock_nff['Value']) * 100,
        0
    )
    
    # Filter hanya saham dengan aktivitas asing yang signifikan
    df_stock_nff = df_stock_nff[df_stock_nff['Total Foreign Activity (Rp)'] > 0]
    
    if not df_stock_nff.empty:
        # ---- TAB UNTUK BERBAGAI VIEW ----
        dominance_tabs = st.tabs([
            "🏆 Top 25 by Value", 
            "📊 Top 25 by Percentage", 
            "🔍 Combined Analysis",
            "📈 Daily Trend"
        ])
        
        with dominance_tabs[0]:
            st.markdown("#### 🥇 25 Saham dengan Aktivitas Asing Terbesar (Value)")
            
            top25_value = df_stock_nff.sort_values('Total Foreign Activity (Rp)', ascending=False).head(25).copy()
            top25_value['Rank'] = range(1, len(top25_value) + 1)
            
            # Visualisasi: Bar chart + scatter
            col_viz1, col_viz2 = st.columns([2, 1])
            
            with col_viz1:
                fig_value = px.bar(
                    top25_value,
                    x='Stock Code',
                    y='Total Foreign Activity (Rp)',
                    color='Persentase Foreign vs Total Value',
                    color_continuous_scale='Viridis',
                    title='Aktivitas Asing (Value)',
                    hover_data=['Sector', 'Close', 'NFF (Rp)']
                )
                fig_value.update_layout(height=500)
                st.plotly_chart(fig_value, use_container_width=True)
            
            with col_viz2:
                # Scatter: Value vs Percentage
                fig_scatter = px.scatter(
                    top25_value,
                    x='Total Foreign Activity (Rp)',
                    y='Persentase Foreign vs Total Value',
                    size='Total Foreign Activity (Rp)',
                    color='Sector',
                    hover_name='Stock Code',
                    title='Value vs Dominasi %',
                    log_x=True
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Tabel detail
            st.markdown("##### 📋 Detail Data")
            display_value = top25_value.copy()
            display_value['Total Foreign Activity (Rp)'] = display_value['Total Foreign Activity (Rp)'].apply(lambda x: format_rupiah(x))
            display_value['NFF (Rp)'] = display_value['NFF (Rp)'].apply(lambda x: format_rupiah(x))
            display_value['Value'] = display_value['Value'].apply(lambda x: format_rupiah(x))
            display_value['Close'] = display_value['Close'].apply(lambda x: f"Rp {x:,.0f}")
            display_value['Persentase Foreign vs Total Value'] = display_value['Persentase Foreign vs Total Value'].apply(lambda x: f"{x:.2f}%")
            display_value['Persentase NFF vs Total Value'] = display_value['Persentase NFF vs Total Value'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(
                display_value[['Rank', 'Stock Code', 'Sector', 'Total Foreign Activity (Rp)', 
                              'Persentase Foreign vs Total Value', 'NFF (Rp)', 'Persentase NFF vs Total Value', 'Close']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Stock Code": st.column_config.TextColumn("Kode", width="small"),
                    "Sector": st.column_config.TextColumn("Sektor"),
                    "Total Foreign Activity (Rp)": st.column_config.TextColumn("Aktivitas Asing"),
                    "Persentase Foreign vs Total Value": st.column_config.ProgressColumn(
                        "Dominasi Asing %",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "NFF (Rp)": st.column_config.TextColumn("Net Flow"),
                    "Persentase NFF vs Total Value": st.column_config.NumberColumn("NFF %", format="%.2f%%"),
                    "Close": st.column_config.TextColumn("Harga")
                }
            )
        
        with dominance_tabs[1]:
            st.markdown("#### 📊 25 Saham dengan Dominasi Asing Tertinggi (Persentase)")
            st.info("Saham dengan persentase tinggi menunjukkan pengaruh asing yang signifikan terhadap aktivitas trading.")
            
            top25_pct = df_stock_nff.sort_values('Persentase Foreign vs Total Value', ascending=False).head(25).copy()
            top25_pct['Rank'] = range(1, len(top25_pct) + 1)
            
            # Filter untuk bubble chart yang lebih baik
            min_activity = top25_pct['Total Foreign Activity (Rp)'].quantile(0.3)
            df_bubble = top25_pct[top25_pct['Total Foreign Activity (Rp)'] >= min_activity]
            
            col_viz3, col_viz4 = st.columns([2, 1])
            
            with col_viz3:
                # Bubble chart: Size = Value, Color = Percentage
                fig_bubble = px.scatter(
                    df_bubble,
                    x='Stock Code',
                    y='Persentase Foreign vs Total Value',
                    size='Total Foreign Activity (Rp)',
                    color='Persentase Foreign vs Total Value',
                    color_continuous_scale='RdYlGn',
                    size_max=50,
                    hover_data=['Sector', 'Total Foreign Activity (Rp)', 'NFF (Rp)'],
                    title='Dominasi Asing (%) vs Aktivitas (Size)'
                )
                fig_bubble.update_layout(height=500, xaxis_tickangle=45)
                st.plotly_chart(fig_bubble, use_container_width=True)
            
            with col_viz4:
                # Histogram persentase
                fig_hist = px.histogram(
                    top25_pct,
                    x='Persentase Foreign vs Total Value',
                    nbins=15,
                    title='Distribusi Dominasi Asing',
                    color_discrete_sequence=['#3498db']
                )
                fig_hist.update_layout(height=500)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Tabel detail - PERBAIKAN DI SINI
            st.markdown("##### 📋 Saham dengan Dominasi Asing Tertinggi")
            
            # Buat DataFrame untuk display dengan data asli untuk styling
            display_pct_original = top25_pct.copy()
            
            # Fungsi untuk highlight berdasarkan nilai numerik asli
            def highlight_high_percentage(row):
                # Gunakan nilai asli dari DataFrame original
                pct_value = display_pct_original.loc[row.name, 'Persentase Foreign vs Total Value']
                if pct_value > 50:
                    return ['background-color: #d4edda'] * len(row)
                elif pct_value > 30:
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return [''] * len(row)
            
            # Buat DataFrame untuk display yang diformat
            display_pct = top25_pct.copy()
            display_pct['Total Foreign Activity (Rp)'] = display_pct['Total Foreign Activity (Rp)'].apply(lambda x: format_rupiah(x))
            display_pct['Value'] = display_pct['Value'].apply(lambda x: format_rupiah(x))
            display_pct['NFF (Rp)'] = display_pct['NFF (Rp)'].apply(lambda x: format_rupiah(x))
            display_pct['Persentase Foreign vs Total Value'] = display_pct['Persentase Foreign vs Total Value'].apply(lambda x: f"{x:.2f}%")
            display_pct['Persentase NFF vs Total Value'] = display_pct['Persentase NFF vs Total Value'].apply(lambda x: f"{x:.2f}%")
            
            # Pastikan index sama untuk mapping
            display_pct_original = display_pct_original.set_index(top25_pct.index)
            display_pct = display_pct.set_index(top25_pct.index)
            
            # Create styled dataframe
            styled_df = display_pct[['Rank', 'Stock Code', 'Sector', 'Persentase Foreign vs Total Value', 
                                    'Total Foreign Activity (Rp)', 'NFF (Rp)', 'Value']].style.apply(
                highlight_high_percentage, axis=1
            )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Stock Code": st.column_config.TextColumn("Kode", width="small"),
                    "Sector": st.column_config.TextColumn("Sektor"),
                    "Persentase Foreign vs Total Value": st.column_config.NumberColumn(
                        "Dominasi Asing %",
                        format="%.2f",
                        help="Persentase aktivitas asing dari total value"
                    ),
                    "Total Foreign Activity (Rp)": st.column_config.TextColumn("Aktivitas Asing"),
                    "NFF (Rp)": st.column_config.TextColumn("Net Flow"),
                    "Value": st.column_config.TextColumn("Total Value")
                }
            )
            
            # Insight box - PERBAIKAN: gunakan nilai asli dari top25_pct
            high_dominance = top25_pct[top25_pct['Persentase Foreign vs Total Value'] > 50]
            if not high_dominance.empty:
                st.success(f"🎯 **Insight**: {len(high_dominance)} saham memiliki dominasi asing >50%. "
                          f"Ini berarti aktivitas trading saham tersebut sangat dipengaruhi oleh asing.")
        
        with dominance_tabs[2]:
            st.markdown("#### 🔍 Analisis Kombinasi: Value vs Persentase")
            st.info("Scatter plot ini membantu mengidentifikasi saham yang menarik dari kedua perspektif.")
            
            # Filter untuk visualisasi yang lebih baik
            min_value_for_viz = df_stock_nff['Total Foreign Activity (Rp)'].quantile(0.1)
            df_viz = df_stock_nff[df_stock_nff['Total Foreign Activity (Rp)'] >= min_value_for_viz].copy()
            
            fig_combined = px.scatter(
                df_viz,
                x='Total Foreign Activity (Rp)',
                y='Persentase Foreign vs Total Value',
                size='Total Foreign Activity (Rp)',
                color='Sector',
                hover_name='Stock Code',
                hover_data=['NFF (Rp)', 'Value', 'Close'],
                title='Value vs Persentase Dominasi Asing',
                log_x=True,
                size_max=40
            )
            
            # Tambahkan quadran lines
            median_value = df_viz['Total Foreign Activity (Rp)'].median()
            median_pct = df_viz['Persentase Foreign vs Total Value'].median()
            
            fig_combined.add_hline(
                y=median_pct,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Median: {median_pct:.1f}%",
                annotation_position="bottom right"
            )
            
            fig_combined.add_vline(
                x=median_value,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Median: {format_rupiah(median_value)}",
                annotation_position="top left"
            )
            
            # Annotate top performers in each quadrant
            # Quadran 1: High Value, High Percentage (Most Interesting)
            q1 = df_viz[(df_viz['Total Foreign Activity (Rp)'] > median_value) & 
                       (df_viz['Persentase Foreign vs Total Value'] > median_pct)].nlargest(5, 'Total Foreign Activity (Rp)')
            
            for idx, row in q1.iterrows():
                fig_combined.add_annotation(
                    x=row['Total Foreign Activity (Rp)'],
                    y=row['Persentase Foreign vs Total Value'],
                    text=row['Stock Code'],
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="green",
                    font=dict(color="green", size=10)
                )
            
            fig_combined.update_layout(
                height=600,
                xaxis_title="Total Aktivitas Asing (Rp, log scale)",
                yaxis_title="Persentase Dominasi Asing (%)",
                showlegend=True
            )
            
            st.plotly_chart(fig_combined, use_container_width=True)
            
            # Quadran Analysis
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            
            q1_count = len(df_viz[(df_viz['Total Foreign Activity (Rp)'] > median_value) & 
                                 (df_viz['Persentase Foreign vs Total Value'] > median_pct)])
            q2_count = len(df_viz[(df_viz['Total Foreign Activity (Rp)'] <= median_value) & 
                                 (df_viz['Persentase Foreign vs Total Value'] > median_pct)])
            q3_count = len(df_viz[(df_viz['Total Foreign Activity (Rp)'] <= median_value) & 
                                 (df_viz['Persentase Foreign vs Total Value'] <= median_pct)])
            q4_count = len(df_viz[(df_viz['Total Foreign Activity (Rp)'] > median_value) & 
                                 (df_viz['Persentase Foreign vs Total Value'] <= median_pct)])
            
            with col_q1:
                st.metric("🏆 High Value & High %", q1_count, 
                         help="Saham dengan aktivitas tinggi DAN dominasi asing tinggi")
            
            with col_q2:
                st.metric("📊 Low Value & High %", q2_count,
                         help="Saham dengan aktivitas rendah TAPI dominasi asing tinggi")
            
            with col_q3:
                st.metric("⚪ Low Value & Low %", q3_count,
                         help="Saham dengan aktivitas rendah DAN dominasi asing rendah")
            
            with col_q4:
                st.metric("💰 High Value & Low %", q4_count,
                         help="Saham dengan aktivitas tinggi TAPI dominasi asing rendah")
            
            # Recommendation
            st.markdown("##### 🎯 Rekomendasi Analisis:")
            st.info("""
            1. **Quadran 1 (High Value & High %)**: Saham yang paling menarik - aktivitas tinggi dan dominasi asing tinggi
            2. **Quadran 2 (Low Value & High %)**: Saham 'hidden gems' - dominasi asing tinggi meski aktivitas rendah
            3. **Quadran 4 (High Value & Low %)**: Saham bluechip - aktivitas tinggi tapi pengaruh asing tidak dominan
            4. **Quadran 3 (Low Value & Low %)**: Saham dengan sedikit pengaruh asing
            """)
        
        with dominance_tabs[3]:
            st.markdown("#### 📈 Trend Harian Dominasi Asing")
            
            # Pilih saham untuk analisis trend
            selected_for_trend = st.selectbox(
                "Pilih Saham untuk Analisis Trend Harian",
                df_stock_nff['Stock Code'].tolist(),
                index=0,
                help="Pilih saham untuk melihat trend harian aktivitas asing"
            )
            
            if selected_for_trend:
                # Filter data harian untuk saham yang dipilih
                df_stock_daily = df[df['Stock Code'] == selected_for_trend].sort_values('Last Trading Date')
                
                if not df_stock_daily.empty:
                    # Hitung persentase harian
                    df_stock_daily['Foreign Activity Daily (Rp)'] = df_stock_daily['Foreign Buy'] + df_stock_daily['Foreign Sell']
                    df_stock_daily['Foreign % Daily'] = np.where(
                        df_stock_daily['Value'] > 0,
                        (df_stock_daily['Foreign Activity Daily (Rp)'] / df_stock_daily['Value']) * 100,
                        0
                    )
                    
                    # Create trend chart
                    fig_trend = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(
                            f'Aktivitas Asing Harian - {selected_for_trend}',
                            'Persentase Dominasi Harian',
                            'Kumulatif NFF'
                        ),
                        row_heights=[0.35, 0.35, 0.3]
                    )
                    
                    # Bar chart aktivitas harian
                    colors_daily = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df_stock_daily['NFF (Rp)']]
                    
                    fig_trend.add_trace(
                        go.Bar(
                            x=df_stock_daily['Last Trading Date'],
                            y=df_stock_daily['Foreign Activity Daily (Rp)'],
                            name='Aktivitas Asing Harian',
                            marker_color='#3498db',
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
                    
                    # Line chart persentase
                    fig_trend.add_trace(
                        go.Scatter(
                            x=df_stock_daily['Last Trading Date'],
                            y=df_stock_daily['Foreign % Daily'],
                            mode='lines+markers',
                            name='% Dominasi Harian',
                            line=dict(color='#9b59b6', width=2),
                            marker=dict(size=4)
                        ),
                        row=2, col=1
                    )
                    
                    # Line chart kumulatif NFF
                    df_stock_daily['NFF Kumulatif'] = df_stock_daily['NFF (Rp)'].cumsum()
                    
                    fig_trend.add_trace(
                        go.Scatter(
                            x=df_stock_daily['Last Trading Date'],
                            y=df_stock_daily['NFF Kumulatif'],
                            mode='lines',
                            name='NFF Kumulatif',
                            line=dict(color='#2ecc71', width=3)
                        ),
                        row=3, col=1
                    )
                    
                    fig_trend.update_layout(
                        height=700,
                        showlegend=True,
                        hovermode='x unified',
                        title=f"Analisis Trend Harian: {selected_for_trend}"
                    )
                    
                    fig_trend.update_yaxes(title_text="Aktivitas (Rp)", row=1, col=1)
                    fig_trend.update_yaxes(title_text="Persentase (%)", row=2, col=1)
                    fig_trend.update_yaxes(title_text="Kumulatif NFF (Rp)", row=3, col=1)
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        avg_pct = df_stock_daily['Foreign % Daily'].mean()
                        st.metric("Rata² Dominasi Harian", f"{avg_pct:.2f}%")
                    
                    with col_stat2:
                        max_pct = df_stock_daily['Foreign % Daily'].max()
                        st.metric("Dominasi Tertinggi Harian", f"{max_pct:.2f}%")
                    
                    with col_stat3:
                        total_activity = df_stock_daily['Foreign Activity Daily (Rp)'].sum()
                        st.metric("Total Aktivitas", format_rupiah(total_activity))
                    
                    with col_stat4:
                        total_nff_stock = df_stock_daily['NFF (Rp)'].sum()
                        st.metric("Total NFF", format_rupiah(total_nff_stock))
        
        # ---- 5. EXPORT DATA ----
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            csv_daily = df_daily_nff.to_csv(index=False)
            st.download_button(
                label="📊 Data Kumulatif Harian",
                data=csv_daily,
                file_name=f"foreign_flow_all_time_{min_date.date()}_to_{max_date.date()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            csv_stocks = df_stock_nff.to_csv(index=False)
            st.download_button(
                label="📈 Data per Saham",
                data=csv_stocks,
                file_name=f"foreign_dominance_by_stock_{max_date.date()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp3:
            # Quick summary
            summary_stats = {
                'total_days': total_days,
                'total_nff': total_nff,
                'avg_daily_nff': avg_daily_nff,
                'positive_days': positive_days,
                'positive_percentage': positive_percentage,
                'total_value': total_value_all_time,
                'nff_percentage_of_total': nff_percentage_of_total,
                'stocks_with_foreign_activity': len(df_stock_nff),
                'avg_foreign_dominance': df_stock_nff['Persentase Foreign vs Total Value'].mean()
            }
            
            import json
            summary_json = json.dumps(summary_stats, indent=2, default=str)
            
            st.download_button(
                label="📋 Summary Stats",
                data=summary_json,
                file_name=f"foreign_flow_summary_{max_date.date()}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # ---- 6. TABEL PERIODE LAIN (Tetap Pertahankan) ----
    st.markdown("---")
    st.markdown("### 📅 Perbandingan Periode Lain")
    
    nff7, nff30, nff90, nff180 = calculate_nff_top_stocks(df, pd.Timestamp(selected_date))
    
    nff_tab1, nff_tab2, nff_tab3, nff_tab4 = st.tabs(["7 Hari", "30 Hari", "90 Hari", "180 Hari"])
    
    with nff_tab1:
        st.dataframe(
            nff7.head(20).assign(
                **{'Total Net FF (Rp)': lambda x: x['Total Net FF (Rp)'].apply(lambda y: format_rupiah(y)),
                  'Harga Terakhir': lambda x: x['Harga Terakhir'].apply(lambda y: f"Rp {y:,.0f}")}
            ),
            hide_index=True,
            use_container_width=True
        )
    
    with nff_tab2:
        st.dataframe(
            nff30.head(20).assign(
                **{'Total Net FF (Rp)': lambda x: x['Total Net FF (Rp)'].apply(lambda y: format_rupiah(y)),
                  'Harga Terakhir': lambda x: x['Harga Terakhir'].apply(lambda y: f"Rp {y:,.0f}")}
            ),
            hide_index=True,
            use_container_width=True
        )
    
    with nff_tab3:
        st.dataframe(
            nff90.head(20).assign(
                **{'Total Net FF (Rp)': lambda x: x['Total Net FF (Rp)'].apply(lambda y: format_rupiah(y)),
                  'Harga Terakhir': lambda x: x['Harga Terakhir'].apply(lambda y: f"Rp {y:,.0f}")}
            ),
            hide_index=True,
            use_container_width=True
        )
    
    with nff_tab4:
        st.dataframe(
            nff180.head(20).assign(
                **{'Total Net FF (Rp)': lambda x: x['Total Net FF (Rp)'].apply(lambda y: format_rupiah(y)),
                  'Harga Terakhir': lambda x: x['Harga Terakhir'].apply(lambda y: f"Rp {y:,.0f}")}
            ),
            hide_index=True,
            use_container_width=True
        )

# ==============================================================================
# TAB 6: MONEY FLOW (BAHASA INDONESIA)
# ==============================================================================
with tabs[5]:
    st.markdown("## 💰 Analisis Money Flow Value")
    
    mfv7, mfv30, mfv90, mfv180 = calculate_mfv_top_stocks(df, pd.Timestamp(selected_date))
    
    # Layout dua kolom
    mfv_col1, mfv_col2 = st.columns(2)
    
    with mfv_col1:
        st.markdown("#### 🥇 Top 10 - 7 Hari")
        st.dataframe(
            mfv7.head(10).assign(
                **{'Total Money Flow (Rp)': lambda x: x['Total Money Flow (Rp)'].apply(lambda y: format_rupiah(y)),
                  'Harga Terakhir': lambda x: x['Harga Terakhir'].apply(lambda y: f"Rp {y:,.0f}")}
            ),
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("#### 🥇 Top 10 - 90 Hari")
        st.dataframe(
            mfv90.head(10).assign(
                **{'Total Money Flow (Rp)': lambda x: x['Total Money Flow (Rp)'].apply(lambda y: format_rupiah(y)),
                  'Harga Terakhir': lambda x: x['Harga Terakhir'].apply(lambda y: f"Rp {y:,.0f}")}
            ),
            hide_index=True,
            use_container_width=True
        )
    
    with mfv_col2:
        st.markdown("#### 🥈 Top 10 - 30 Hari")
        st.dataframe(
            mfv30.head(10).assign(
                **{'Total Money Flow (Rp)': lambda x: x['Total Money Flow (Rp)'].apply(lambda y: format_rupiah(y)),
                  'Harga Terakhir': lambda x: x['Harga Terakhir'].apply(lambda y: f"Rp {y:,.0f}")}
            ),
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("#### 🥈 Top 10 - 180 Hari")
        st.dataframe(
            mfv180.head(10).assign(
                **{'Total Money Flow (Rp)': lambda x: x['Total Money Flow (Rp)'].apply(lambda y: format_rupiah(y)),
                  'Harga Terakhir': lambda x: x['Harga Terakhir'].apply(lambda y: f"Rp {y:,.0f}")}
            ),
            hide_index=True,
            use_container_width=True
        )

# ==============================================================================
# TAB 7: BACKTESTING ENGINE (FIXED)
# ==============================================================================
with tabs[6]:
    st.markdown("## 🧪 Backtesting Engine")
    st.info("Simulasi performa sistem scoring dalam periode historis.")
    
    col_bt1, col_bt2, col_bt3 = st.columns(3)
    
    with col_bt1:
        days_to_test = st.slider("Hari Backtest", 7, 180, 90, 7)
    
    with col_bt2:
        initial_capital = st.number_input("Modal Awal (Rp)", 
                                         min_value=1000000, 
                                         max_value=1000000000, 
                                         value=10000000, 
                                         step=1000000)
    
    with col_bt3:
        st.markdown("")
        st.markdown("")
        run_btn = st.button("🚀 Jalankan Backtest", use_container_width=True)
    
    if run_btn:
        with st.spinner("Menjalankan simulasi..."):
            df_backtest = run_backtest_analysis(df, days_back=days_to_test)
        
        if not df_backtest.empty:
            # Ringkasan performa
            total_signals = len(df_backtest)
            winning_signals = len(df_backtest[df_backtest['Return to Date (%)'] > 0])
            win_rate = (winning_signals / total_signals * 100) if total_signals > 0 else 0
            avg_return = df_backtest['Return to Date (%)'].mean()
            total_return = df_backtest['Return to Date (%)'].sum()
            
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            
            with col_sum1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">Total Sinyal</div>
                    <div style="font-size:1.5rem; font-weight:700;">{total_signals:,}x</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_sum2:
                win_color = "#2ecc71" if win_rate > 50 else "#e74c3c"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">Win Rate</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{win_color};">{win_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_sum3:
                return_color = "#2ecc71" if avg_return > 0 else "#e74c3c"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">Avg Return</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{return_color};">{avg_return:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_sum4:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#6c757d;">Total Return</div>
                    <div style="font-size:1.5rem; font-weight:700;">{total_return:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualisasi distribusi return
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                fig_return_dist = px.histogram(
                    df_backtest,
                    x='Return to Date (%)',
                    nbins=30,
                    title='Distribusi Return',
                    color_discrete_sequence=['#4361ee']
                )
                fig_return_dist.update_layout(height=400)
                st.plotly_chart(fig_return_dist, use_container_width=True)
            
            with col_viz2:
                # Scatter plot return vs score (TANPA TRENDLINE)
                fig_scatter = px.scatter(
                    df_backtest,
                    x='Score at Signal',
                    y='Return to Date (%)',
                    color='Return to Date (%)',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
                    title='Hubungan Skor vs Return'
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("### 📊 Leaderboard Saham")
            
            freq_stats = df_backtest.groupby('Stock Code').agg(
                Frekuensi=('Signal Date', 'count'),
                Avg_Return=('Return to Date (%)', 'mean'),
                Win_Rate=('Return to Date (%)', lambda x: (x > 0).mean() * 100),
                Total_Return=('Return to Date (%)', 'sum')
            ).reset_index().sort_values(['Frekuensi', 'Avg_Return'], ascending=[False, False])
            
            # Konversi Frekuensi ke int untuk ProgressColumn
            max_freq = int(freq_stats['Frekuensi'].max()) if not freq_stats.empty else 1
            
            st.dataframe(
                freq_stats,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Stock Code": st.column_config.TextColumn("Kode Saham"),
                    "Frekuensi": st.column_config.ProgressColumn(
                        "Frekuensi Sinyal",
                        format="%d x",
                        max_value=max_freq
                    ),
                    "Avg_Return": st.column_config.NumberColumn("Avg Return (%)", format="%.2f"),
                    "Win_Rate": st.column_config.NumberColumn("Win Rate (%)", format="%.1f"),
                    "Total_Return": st.column_config.NumberColumn("Total Return (%)", format="%.2f")
                }
            )
            
            with st.expander("📋 Detail Log Backtest"):
                st.dataframe(
                    df_backtest.sort_values('Signal Date', ascending=False),
                    use_container_width=True
                )
        else:
            st.warning("Tidak ada data backtest yang dihasilkan.")

# ==============================================================================
# TAB 8: PORTFOLIO SIMULATOR (FIXED)
# ==============================================================================
with tabs[7]:
    st.markdown("## 💼 Portfolio Simulator")
    st.info("Simulasi investasi berdasarkan top 20 saham potensial pada tanggal tertentu.")
    
    # Get available dates
    avail_dates = sorted(df['Last Trading Date'].unique())
    
    # Input controls
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    with col_p1:
        start_d = st.selectbox(
            "📅 Tanggal Beli",
            avail_dates,
            index=max(0, len(avail_dates)-31),
            format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y')
        )
    
    with col_p2:
        end_d = st.selectbox(
            "📅 Tanggal Jual",
            avail_dates,
            index=len(avail_dates)-1,
            format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y')
        )
    
    with col_p3:
        capital = st.number_input(
            "💰 Modal (Rp)",
            min_value=1_000_000,
            max_value=1_000_000_000,
            value=20_000_000,
            step=1_000_000
        )
    
    with col_p4:
        st.write("")  # Spacer
        st.write("")  # Spacer
        btn_calc = st.button("🚀 Hitung Portfolio", use_container_width=True)
    
    if btn_calc:
        if pd.Timestamp(start_d) >= pd.Timestamp(end_d):
            st.error("❌ Tanggal Jual harus setelah Tanggal Beli.")
        else:
            with st.spinner("📈 Menghitung portfolio..."):
                df_port, sum_port, msg = simulate_portfolio_range(
                    df, capital, pd.Timestamp(start_d), pd.Timestamp(end_d)
                )
            
            if msg == "success" and df_port is not None and sum_port is not None:
                # Display summary metrics
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                
                with col_s1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#6c757d;">Modal Awal</div>
                        <div style="font-size:1.5rem; font-weight:700;">{format_rupiah(sum_port['Initial Capital'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_s2:
                    final_val = sum_port['Final Portfolio Value']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#6c757d;">Saldo Akhir</div>
                        <div style="font-size:1.5rem; font-weight:700;">{format_rupiah(final_val)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_s3:
                    net_profit = sum_port['Net Profit']
                    profit_color = "#2ecc71" if net_profit >= 0 else "#e74c3c"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#6c757d;">Net Profit/Loss</div>
                        <div style="font-size:1.5rem; font-weight:700; color:{profit_color};">{format_rupiah(net_profit)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_s4:
                    total_roi = sum_port['Total ROI']
                    roi_color = "#2ecc71" if total_roi >= 0 else "#e74c3c"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#6c757d;">Total ROI</div>
                        <div style="font-size:1.5rem; font-weight:700; color:{roi_color};">{total_roi:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Charts
                col_ch1, col_ch2 = st.columns(2)
                
                with col_ch1:
                    # PnL per saham
                    df_pnl = df_port.sort_values('Gain/Loss (Rp)', ascending=False)
                    fig_pnl = px.bar(
                        df_pnl,
                        x='Stock Code',
                        y='Gain/Loss (Rp)',
                        color='Gain/Loss (Rp)',
                        color_continuous_scale=['#e74c3c', '#2ecc71'],
                        title="Profit/Loss per Saham"
                    )
                    fig_pnl.update_layout(height=400)
                    st.plotly_chart(fig_pnl, use_container_width=True)
                
                with col_ch2:
                    # Sector distribution
                    fig_sector = px.pie(
                        df_port,
                        names='Sector',
                        values='Final Value',
                        title="Distribusi Portfolio per Sektor",
                        hole=0.4
                    )
                    fig_sector.update_layout(height=400)
                    st.plotly_chart(fig_sector, use_container_width=True)
                
                # Detailed results table
                st.markdown("### 📋 Detail Portfolio")
                
                # Format table
                df_display = df_port.copy()
                df_display['Buy Price'] = df_display['Buy Price'].apply(lambda x: f"Rp {x:,.0f}")
                df_display['Sell Price'] = df_display['Sell Price'].apply(lambda x: f"Rp {x:,.0f}")
                df_display['Gain/Loss (Rp)'] = df_display['Gain/Loss (Rp)'].apply(lambda x: format_rupiah(x))
                df_display['ROI (%)'] = df_display['ROI (%)'].apply(lambda x: f"{x:.2f}%")
                df_display['Final Value'] = df_display['Final Value'].apply(lambda x: format_rupiah(x))
                
                st.dataframe(
                    df_display[['Stock Code', 'Sector', 'Buy Price', 'Sell Price', 'Gain/Loss (Rp)', 'ROI (%)', 'Final Value']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export option
                csv = df_port.to_csv(index=False)
                st.download_button(
                    label="📥 Download Portfolio Results",
                    data=csv,
                    file_name=f"portfolio_{start_d}_to_{end_d}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error(f"❌ {msg}")

# ==============================================================================
# TAB 9: MSCI SIMULATOR (BAHASA INDONESIA LENGKAP)
# ==============================================================================
with tabs[8]:
    st.markdown("## 🌏 MSCI Indonesia Index Simulator")
    st.info("""
    Simulator untuk memperkirakan kandidat MSCI Standard Index berdasarkan:
    - Market Capitalization (US$)
    - Liquidity (ATVR - Annualized Traded Value Ratio)
    - Free Float Percentage
    """)
    
    # Check required columns
    required_cols = ['Listed Shares', 'Free Float']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Kolom yang diperlukan tidak ditemukan: {', '.join(missing_cols)}")
        st.info("Pastikan data Anda memiliki kolom 'Listed Shares' dan 'Free Float'")
    else:
        # Input parameters
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            usd_idr = st.number_input(
                "💵 Kurs US$/IDR Saat Ini",
                value=16500,
                min_value=10000,
                max_value=20000,
                step=50
            )
        
        with col_m2:
            min_float_mcap_usd_miliar = st.number_input(
                "🎯 Minimal Float Market Cap (US$ Miliar)",
                value=1.5,
                min_value=0.1,
                max_value=10.0,
                step=0.1
            )
        
        with col_m3:
            min_atvr = st.number_input(
                "📈 Minimal Liquidity (ATVR %)",
                value=15.0,
                min_value=0.0,
                max_value=100.0,
                step=1.0
            )
        
        # Calculate button
        if st.button("🚀 Hitung MSCI Projection", use_container_width=True):
            with st.spinner("📊 Menghitung proyeksi MSCI..."):
                # Get latest date data
                latest_date = df['Last Trading Date'].max()
                
                # Calculate MSCI metrics
                df_msci = calculate_msci_projection_v2(df, latest_date, usd_idr)
                
                if not df_msci.empty:
                    # Categorization function
                    def categorize_msci(row):
                        # Check size criteria
                        size_ok = row['Float Cap (US$ Miliar)'] >= min_float_mcap_usd_miliar
                        
                        # Check liquidity criteria (both 3M and 12M)
                        liquidity_ok = (row['ATVR 3 Bulan (%)'] >= min_atvr) and (row['ATVR 12 Bulan (%)'] >= min_atvr)
                        
                        if size_ok and liquidity_ok:
                            return "✅ Potential Standard"
                        elif size_ok and not liquidity_ok:
                            return "⚠️ Risk (Low Liquidity)"
                        elif row['Float Cap (US$ Miliar)'] >= (min_float_mcap_usd_miliar * 0.3):
                            return "🔹 Small Cap"
                        else:
                            return "🔻 Micro Cap"
                    
                    # Apply categorization
                    df_msci['MSCI Status'] = df_msci.apply(categorize_msci, axis=1)
                    
                    # Display summary
                    st.markdown("### 📊 Summary")
                    
                    status_counts = df_msci['MSCI Status'].value_counts()
                    col_sm1, col_sm2, col_sm3, col_sm4 = st.columns(4)
                    
                    with col_sm1:
                        potential = status_counts.get('✅ Potential Standard', 0)
                        st.metric("Potential Standard", potential)
                    
                    with col_sm2:
                        risk = status_counts.get('⚠️ Risk (Low Liquidity)', 0)
                        st.metric("Risk (Low Liquidity)", risk)
                    
                    with col_sm3:
                        small = status_counts.get('🔹 Small Cap', 0)
                        st.metric("Small Cap", small)
                    
                    with col_sm4:
                        micro = status_counts.get('🔻 Micro Cap', 0)
                        st.metric("Micro Cap", micro)
                    
                    st.markdown("---")
                    
                    # Visualization
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Scatter plot: Float Cap vs ATVR
                        fig_msci = px.scatter(
                            df_msci.head(50),  # Limit to top 50 for better visibility
                            x='ATVR 12 Bulan (%)',
                            y='Float Cap (US$ Miliar)',
                            color='MSCI Status',
                            size='Full Cap (US$ Miliar)',
                            hover_data=['Stock Code', 'Sector', 'ATVR 3 Bulan (%)'],
                            title="MSCI Qualification Map (Top 50)",
                            color_discrete_map={
                                '✅ Potential Standard': '#2ecc71',
                                '⚠️ Risk (Low Liquidity)': '#e74c3c',
                                '🔹 Small Cap': '#3498db',
                                '🔻 Micro Cap': '#95a5a6'
                            }
                        )
                        # Add threshold lines
                        fig_msci.add_hline(
                            y=min_float_mcap_usd_miliar,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Min Float: US$ {min_float_mcap_usd_miliar} Miliar"
                        )
                        fig_msci.add_vline(
                            x=min_atvr,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text=f"Min ATVR: {min_atvr}%"
                        )
                        fig_msci.update_layout(height=500)
                        st.plotly_chart(fig_msci, use_container_width=True)
                    
                    with col_viz2:
                        # Top candidates table
                        st.markdown("#### 🏆 Top Candidates (Standard Index)")
                        
                        potential_df = df_msci[df_msci['MSCI Status'] == '✅ Potential Standard'].copy()
                        potential_df = potential_df.sort_values('Float Cap (US$ Miliar)', ascending=False)
                        
                        if not potential_df.empty:
                            # Format display
                            display_df = potential_df[['Stock Code', 'Sector', 'Float Cap (US$ Miliar)', 'ATVR 12 Bulan (%)', 'ATVR 3 Bulan (%)']].copy()
                            display_df['Float Cap (US$ Miliar)'] = display_df['Float Cap (US$ Miliar)'].apply(lambda x: format_usd(x * 1e9))  # Convert back to full USD
                            display_df['ATVR 12 Bulan (%)'] = display_df['ATVR 12 Bulan (%)'].apply(lambda x: f"{x:.1f}%")
                            display_df['ATVR 3 Bulan (%)'] = display_df['ATVR 3 Bulan (%)'].apply(lambda x: f"{x:.1f}%")
                            
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("Tidak ada saham yang memenuhi kriteria Standard Index")
                    
                    st.markdown("---")
                    
                    # Detailed results with tabs
                    st.markdown("### 📋 Detailed Results")
                    
                    msci_tabs = st.tabs(["All Stocks", "By Status", "Export"])
                    
                    with msci_tabs[0]:
                        # All stocks with formatted values
                        display_all = df_msci.copy()
                        display_all['Float Cap (US$ Miliar)'] = display_all['Float Cap (US$ Miliar)'].apply(lambda x: format_usd(x * 1e9))
                        display_all['Full Cap (US$ Miliar)'] = display_all['Full Cap (US$ Miliar)'].apply(lambda x: format_usd(x * 1e9))
                        display_all['ATVR 12 Bulan (%)'] = display_all['ATVR 12 Bulan (%)'].apply(lambda x: f"{x:.1f}%")
                        display_all['ATVR 3 Bulan (%)'] = display_all['ATVR 3 Bulan (%)'].apply(lambda x: f"{x:.1f}%")
                        display_all['Float Cap (IDR Triliun)'] = display_all['Float Cap (IDR Triliun)'].apply(lambda x: f"Rp {x:.2f} Triliun")
                        
                        st.dataframe(
                            display_all.sort_values('Float Cap (US$ Miliar)', ascending=False),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with msci_tabs[1]:
                        # Filter by status
                        selected_status = st.selectbox(
                            "Filter by Status",
                            df_msci['MSCI Status'].unique()
                        )
                        
                        filtered_df = df_msci[df_msci['MSCI Status'] == selected_status]
                        st.dataframe(
                            filtered_df.sort_values('Float Cap (US$ Miliar)', ascending=False),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with msci_tabs[2]:
                        # Export options
                        st.markdown("#### 📥 Export Data")
                        
                        csv = df_msci.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"msci_simulation_{latest_date.date()}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Summary stats
                        st.markdown("#### 📊 Summary Statistics")
                        st.json({
                            "simulation_date": str(latest_date.date()),
                            "usd_rate": usd_idr,
                            "min_float_mcap_usd_miliar": min_float_mcap_usd_miliar,
                            "min_atvr": min_atvr,
                            "total_stocks_analyzed": len(df_msci),
                            "status_distribution": status_counts.to_dict()
                        })
                else:
                    st.error("❌ Gagal menghitung proyeksi MSCI. Data tidak tersedia.")
        else:
            # Show preview before calculation
            st.markdown("#### ℹ️ Preview Criteria")
            st.info(f"""
            **Kriteria saat ini:**
            - Kurs US$/IDR: Rp {usd_idr:,}
            - Minimal Float Market Cap: US$ {min_float_mcap_usd_miliar} Miliar
            - Minimal Liquidity (ATVR): {min_atvr}%
            
            **Klik 'Hitung MSCI Projection' untuk memulai simulasi.**
            """)

# ==============================================================================
# TAB 10: ANALISIS SEKTOR (BAHASA INDONESIA)
# ==============================================================================
with tabs[9]:
    st.markdown("## 📊 Analisis Sektor")
    
    # Filter by date range
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        sector_start_date = st.date_input(
            "Tanggal Mulai",
            value=selected_date - timedelta(days=30),
            max_value=selected_date
        )
    
    with col_s2:
        sector_end_date = st.date_input(
            "Tanggal Akhir",
            value=selected_date,
            min_value=sector_start_date
        )
    
    # Filter data by date range
    df_sector = df[
        (df['Last Trading Date'].dt.date >= sector_start_date) &
        (df['Last Trading Date'].dt.date <= sector_end_date)
    ].copy()
    
    if not df_sector.empty:
        # Sector performance analysis
        sector_stats = df_sector.groupby('Sector').agg({
            'Stock Code': 'nunique',
            'Close': 'mean',
            'Change %': 'mean',
            'Value': 'sum',
            'NFF (Rp)': 'sum',
            'Money Flow Value': 'sum',
            'Volume': 'sum'
        }).round(2)
        
        sector_stats = sector_stats.rename(columns={
            'Stock Code': 'Jumlah Saham',
            'Close': 'Avg Harga',
            'Change %': 'Avg Change %',
            'Value': 'Total Value (Rp)',
            'NFF (Rp)': 'Total NFF (Rp)',
            'Money Flow Value': 'Total MFV (Rp)',
            'Volume': 'Total Volume'
        })
        
        # Display metrics
        st.markdown("### 📈 Performa Sektor")
        
        # Top performing sectors
        top_sectors = sector_stats.sort_values('Avg Change %', ascending=False)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_sector_perf = px.bar(
                top_sectors.head(10),
                x=top_sectors.head(10).index,
                y='Avg Change %',
                color='Avg Change %',
                color_continuous_scale=px.colors.diverging.RdYlGn,
                title="Top 10 Sektor (Avg Change %)"
            )
            fig_sector_perf.update_layout(height=400)
            st.plotly_chart(fig_sector_perf, use_container_width=True)
        
        with col_chart2:
            # Sector by value traded
            fig_sector_value = px.pie(
                sector_stats,
                values='Total Value (Rp)',
                names=sector_stats.index,
                title="Distribusi Nilai Transaksi per Sektor",
                hole=0.4
            )
            fig_sector_value.update_layout(height=400)
            st.plotly_chart(fig_sector_value, use_container_width=True)
        
        # Detailed sector table
        st.markdown("### 📋 Detail Statistik Sektor")
        
        # Format table for display
        display_stats = sector_stats.copy()
        display_stats['Avg Harga'] = display_stats['Avg Harga'].apply(lambda x: f"Rp {x:,.0f}")
        display_stats['Avg Change %'] = display_stats['Avg Change %'].apply(lambda x: f"{x:.2f}%")
        display_stats['Total Value (Rp)'] = display_stats['Total Value (Rp)'].apply(lambda x: format_rupiah(x))
        display_stats['Total NFF (Rp)'] = display_stats['Total NFF (Rp)'].apply(lambda x: format_rupiah(x))
        display_stats['Total MFV (Rp)'] = display_stats['Total MFV (Rp)'].apply(lambda x: format_rupiah(x))
        display_stats['Total Volume'] = display_stats['Total Volume'].apply(lambda x: format_volume(x))
        
        st.dataframe(
            display_stats,
            use_container_width=True
        )
        
        # Export option
        csv = sector_stats.to_csv()
        st.download_button(
            label="📥 Ekspor Data Sektor",
            data=csv,
            file_name=f"sector_analysis_{sector_start_date}_to_{sector_end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("Tidak ada data untuk periode yang dipilih.")

# ==============================================================================
# 🎯 FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <p>📊 <strong>Quantum IDX Dashboard</strong> | Advanced Stock Analytics Platform</p>
    <p style="font-size: 0.9rem;">© 2024 | Real-time Data Analytics & Simulation</p>
</div>
""", unsafe_allow_html=True)
