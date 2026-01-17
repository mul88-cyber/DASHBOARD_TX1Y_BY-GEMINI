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
from datetime import datetime

# Google Drive imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ 2) CONFIG & MODERN UI STYLING
# ==============================================================================
st.set_page_config(
    page_title="IDX Quant Pro",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- MDM STYLE CSS (Light, Clean, Purple Gradient) ---
st.markdown("""
<style>
    /* 1. Global Variables */
    :root {
        --primary-color: #4318FF;
        --background-light: #F4F7FE;
        --card-white: #FFFFFF;
        --text-dark: #2B3674;
        --text-grey: #A3AED0;
    }

    /* 2. Main Background */
    .stApp {
        background-color: var(--background-light);
        color: var(--text-dark);
        font-family: 'DM Sans', sans-serif;
    }

    /* 3. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--card-white);
        box-shadow: 14px 14px 40px rgba(112, 144, 176, 0.08);
        border-right: none;
    }
    
    /* 4. Card Container Styling */
    .css-card {
        background-color: var(--card-white);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
        margin-bottom: 24px;
    }
    
    /* 5. Header Gradient Banner */
    .header-banner {
        background: linear-gradient(86.88deg, #4318FF 0%, #868CFF 100%);
        border-radius: 20px;
        padding: 30px 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.2);
    }
    .header-title { font-size: 32px; font-weight: 700; margin-bottom: 8px; }
    .header-subtitle { font-size: 16px; font-weight: 500; opacity: 0.9; }

    /* 6. Metrics Styling */
    div[data-testid="stMetricValue"] {
        color: var(--text-dark) !important;
        font-weight: 700;
        font-size: 28px;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-grey) !important;
        font-size: 14px;
        font-weight: 500;
    }

    /* 7. Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4318FF 0%, #868CFF 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(67, 24, 255, 0.2);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 15px rgba(67, 24, 255, 0.3);
        color: white;
    }

    /* 8. Text & Headers */
    h1, h2, h3, h4, h5 {
        color: var(--text-dark) !important;
    }
    
    /* Custom Card Title Helper */
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & LOGIC WEIGHTS ---
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
FILE_NAME = "Kompilasi_Data_1Tahun.csv"

# ==============================================================================
# 📦 3) DATA LOADER
# ==============================================================================
def get_gdrive_service():
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except KeyError:
        return None, "❌ Key [gcp_service_account] missing in secrets."
    except Exception as e:
        return None, f"❌ Auth Error: {e}"

@st.cache_data(ttl=3600, show_spinner="🔄 Fetching Market Data...")
def load_data():
    service, error_msg = get_gdrive_service()
    if error_msg: return pd.DataFrame(), error_msg, "error"

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1).execute()
        items = results.get('files', [])

        if not items: return pd.DataFrame(), f"❌ File '{FILE_NAME}' not found.", "error"

        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: status, done = downloader.next_chunk()
        fh.seek(0)

        df = pd.read_csv(fh, dtype=object)
        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')

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
                cleaned = df[col].astype(str).str.strip().str.replace(r'[,\sRp\%]', '', regex=True)
                df[col] = pd.to_numeric(cleaned, errors='coerce').fillna(0)

        if 'Unusual Volume' in df.columns:
            df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true', 'True'])
        
        if 'Sector' in df.columns:
             df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
             df['Sector'] = 'Others'

        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])

        if 'NFF (Rp)' not in df.columns:
             if 'Typical Price' in df.columns: df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Typical Price']
             else: df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Close']

        return df, "✅ Data loaded successfully.", "success"

    except Exception as e:
        return pd.DataFrame(), f"❌ Data Load Error: {e}", "error"

# ==============================================================================
# 🧠 4) ANALYTICS ENGINE
# ==============================================================================
def pct_rank(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0) * 100

def calculate_potential_score(df, latest_date):
    trend_start = latest_date - pd.Timedelta(days=30)
    mom_start = latest_date - pd.Timedelta(days=7)
    
    df_historic = df[df['Last Trading Date'] <= latest_date]
    trend_df = df_historic[df_historic['Last Trading Date'] >= trend_start]
    mom_df = df_historic[df_historic['Last Trading Date'] >= mom_start]

    if trend_df.empty: return pd.DataFrame(), "Insufficient Data", "warning"

    # 1. Trend Score
    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close', 'last'), 
        total_net_ff_rp=('NFF (Rp)', 'sum'), 
        total_money_flow=('Money Flow Value', 'sum'),
        avg_change_pct=('Change %', 'mean'), 
        sector=('Sector', 'last')
    ).reset_index()
    
    tr['Trend Score'] = (pct_rank(tr['total_net_ff_rp']) * 0.4 + 
                         pct_rank(tr['total_money_flow']) * 0.3 + 
                         pct_rank(tr['avg_change_pct']) * 0.3)

    # 2. Momentum
    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %', 'sum'),
        total_net_ff_rp=('NFF (Rp)', 'sum'),
        had_unusual_volume=('Unusual Volume', 'any')
    ).reset_index()
    
    mo['Momentum Score'] = (pct_rank(mo['total_change_pct']) * 0.5 + 
                            pct_rank(mo['total_net_ff_rp']) * 0.3 + 
                            mo['had_unusual_volume'].astype(int) * 20)

    # Merge
    rank = tr.merge(mo[['Stock Code', 'Momentum Score']], on='Stock Code', how='outer')
    rank['Potential Score'] = (rank['Trend Score'].fillna(0)*0.5 + rank['Momentum Score'].fillna(0)*0.5)
    
    # Calculate NBSA & Foreign score for Radar
    rank['NBSA Score'] = pct_rank(rank['total_net_ff_rp'])
    rank['Foreign Contrib Score'] = pct_rank(rank['total_net_ff_rp']) # Proxy
    
    top20 = rank.sort_values('Potential Score', ascending=False).head(20).copy()
    top20.insert(0, 'Rank', range(1, len(top20)+1))
    return top20, "Scored", "success"

@st.cache_data(ttl=3600)
def calculate_msci_projection_v2(df, latest_date, usd_rate):
    start_12m = latest_date - pd.Timedelta(days=365)
    start_3m = latest_date - pd.Timedelta(days=90)
    
    df_12m = df[(df['Last Trading Date'] >= start_12m) & (df['Last Trading Date'] <= latest_date)]
    df_3m = df[(df['Last Trading Date'] >= start_3m) & (df['Last Trading Date'] <= latest_date)]
    df_last = df[df['Last Trading Date'] == latest_date].copy()
    
    results = []
    for _, row in df_last.iterrows():
        code = row['Stock Code']
        val_12m = df_12m[df_12m['Stock Code'] == code]['Value'].sum()
        val_3m = df_3m[df_3m['Stock Code'] == code]['Value'].sum()
        
        float_mcap_idr_t = (row['Close'] * row.get('Listed Shares', 0) * row.get('Free Float', 0)/100) / 1e12
        full_mcap_usd_b = (row['Close'] * row.get('Listed Shares', 0)) / usd_rate / 1e9
        float_mcap_usd_b = (float_mcap_idr_t * 1e12) / usd_rate / 1e9
        
        float_mcap_full = float_mcap_idr_t * 1e12
        atvr_12m = (val_12m / float_mcap_full * 100) if float_mcap_full > 0 else 0
        atvr_3m = ((val_3m * 4) / float_mcap_full * 100) if float_mcap_full > 0 else 0
            
        results.append({
            'Stock Code': code, 'Sector': row['Sector'],
            'Float Cap (IDR T)': float_mcap_idr_t,
            'Full Cap ($B)': full_mcap_usd_b,
            'Float Cap ($B)': float_mcap_usd_b,
            'ATVR 12M (%)': atvr_12m, 'ATVR 3M (%)': atvr_3m
        })
        
    df_msci = pd.DataFrame(results).sort_values(by='Float Cap ($B)', ascending=False)
    return df_msci

# Simulasi Portfolio
def simulate_portfolio_range(df, capital, start, end):
    top20, _, _ = calculate_potential_score(df, start)
    if top20.empty: return pd.DataFrame(), {}, "error"
    
    df_end = df[df['Last Trading Date'] == end]
    if df_end.empty: return pd.DataFrame(), {}, "No end data"
    
    results = []
    alloc = capital / len(top20)
    
    for _, row in top20.iterrows():
        code = row['Stock Code']
        buy = row['last_price']
        end_row = df_end[df_end['Stock Code'] == code]
        
        sell = end_row.iloc[0]['Close'] if not end_row.empty else buy
        gain = (sell - buy) / buy
        val = alloc * (1 + gain)
        
        results.append({
            'Stock Code': code, 'Sector': row['sector'],
            'Buy Price': buy, 'Sell Price': sell,
            'Gain/Loss (Rp)': val - alloc, 'ROI (%)': gain*100,
            'Final Value': val
        })
        
    df_res = pd.DataFrame(results)
    summary = {
        'Final Portfolio Value': df_res['Final Value'].sum(),
        'Net Profit': df_res['Final Value'].sum() - capital,
        'Total ROI': (df_res['Final Value'].sum() - capital)/capital * 100
    }
    return df_res, summary, "success"

# ==============================================================================
# 🎨 5) MAIN LAYOUT & UI
# ==============================================================================

# --- LOAD DATA ---
df, msg, status = load_data()
if status == "error": st.error(msg); st.stop()

# --- SIDEBAR (Clean White) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=50)
    st.markdown("<h3 style='color:#2B3674; margin-top:0;'>IDX Quant Pro</h3>", unsafe_allow_html=True)
    st.markdown("<div style='color:#A3AED0; font-size:12px; margin-bottom:20px;'>Advanced Stock Analytics</div>", unsafe_allow_html=True)
    
    menu = st.radio("MAIN MENU", [
        "📊 Market Dashboard", 
        "🔍 Deep Dive Analysis", 
        "🏆 Algorithmic Picks", 
        "💼 Portfolio Simulator", 
        "🌏 MSCI Projector"
    ])
    
    st.divider()
    
    # Global Date Filter
    st.markdown("<b>📅 Data Filter</b>", unsafe_allow_html=True)
    max_date = df['Last Trading Date'].max().date()
    sel_date = st.date_input("Select Date", max_date, min_value=df['Last Trading Date'].min().date(), max_value=max_date)
    
    st.divider()
    if st.button("🔄 Refresh System"):
        st.cache_data.clear()
        st.rerun()

# --- MAIN CONTENT AREA ---

# PAGE 1: DASHBOARD
if menu == "📊 Market Dashboard":
    # 1. Header Banner
    st.markdown(f"""
    <div class="header-banner">
        <div class="header-title">Welcome, Stock Analyzers! 👋</div>
        <div class="header-subtitle">Market Overview for {sel_date.strftime('%d %B %Y')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    df_day = df[df['Last Trading Date'].dt.date == sel_date]
    
    # 2. Key Metrics Row (Card Style)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""<div class="css-card">""", unsafe_allow_html=True)
        st.metric("Total Transaksi", f"Rp {df_day['Value'].sum()/1e12:,.2f} T")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_m2:
        st.markdown("""<div class="css-card">""", unsafe_allow_html=True)
        st.metric("Saham Aktif", f"{len(df_day):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_m3:
        st.markdown("""<div class="css-card">""", unsafe_allow_html=True)
        st.metric("Unusual Vol", f"{df_day['Unusual Volume'].sum():,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_m4:
        st.markdown("""<div class="css-card">""", unsafe_allow_html=True)
        net_foreign = df_day['NFF (Rp)'].sum()
        st.metric("Net Foreign", f"Rp {net_foreign/1e9:,.1f} M", delta_color="normal" if net_foreign > 0 else "inverse")
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Market Map & Tables
    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗺️ Sector Market Map</div>', unsafe_allow_html=True)
        if 'Sector' in df_day.columns:
            sec_agg = df_day.groupby('Sector')['Value'].sum().reset_index()
            fig = px.treemap(sec_agg, path=['Sector'], values='Value', color='Value', color_continuous_scale='Blues')
            fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_c2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🚀 Top Gainers</div>', unsafe_allow_html=True)
        gainers = df_day.nlargest(10, 'Change %')[['Stock Code', 'Close', 'Change %']]
        st.dataframe(
            gainers, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Close": st.column_config.NumberColumn(format="Rp %,d"),
                "Change %": st.column_config.NumberColumn(format="%.2f %%")
            },
            height=400
        )
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 2: DEEP DIVE
elif menu == "🔍 Deep Dive Analysis":
    st.markdown('<h2 style="color:#2B3674;">🔍 Individual Stock Scanner</h2>', unsafe_allow_html=True)
    
    # Filter Bar (White Card)
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    stocks = sorted(df['Stock Code'].unique())
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        sel_stock = st.selectbox("Cari Kode Saham:", stocks, index=stocks.index('BBRI') if 'BBRI' in stocks else 0)
    with col_s2:
        st.write("") # Spacer
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process Data
    stock_df = df[df['Stock Code'] == sel_stock].sort_values('Last Trading Date')
    if not stock_df.empty:
        last = stock_df.iloc[-1]
        
        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"""<div class="css-card" style="text-align:center;">
            <div style="font-size:14px; color:#A3AED0;">Harga Terakhir</div>
            <div style="font-size:24px; font-weight:700; color:#2B3674;">Rp {last['Close']:,.0f}</div>
            <div style="color:{'green' if last['Change %']>0 else 'red'}; font-weight:bold;">{last['Change %']:.2f}%</div>
        </div>""", unsafe_allow_html=True)
        
        k2.markdown(f"""<div class="css-card" style="text-align:center;">
            <div style="font-size:14px; color:#A3AED0;">Net Foreign (Daily)</div>
            <div style="font-size:24px; font-weight:700; color:#2B3674;">Rp {last['NFF (Rp)']/1e9:,.1f} M</div>
        </div>""", unsafe_allow_html=True)
        
        k3.markdown(f"""<div class="css-card" style="text-align:center;">
            <div style="font-size:14px; color:#A3AED0;">Money Flow Value</div>
            <div style="font-size:24px; font-weight:700; color:#2B3674;">Rp {last['Money Flow Value']/1e9:,.1f} M</div>
        </div>""", unsafe_allow_html=True)
        
        k4.markdown(f"""<div class="css-card" style="text-align:center;">
            <div style="font-size:14px; color:#A3AED0;">Transaction Value</div>
            <div style="font-size:24px; font-weight:700; color:#2B3674;">Rp {last['Value']/1e9:,.1f} M</div>
        </div>""", unsafe_allow_html=True)
        
        # Advanced Charts in Tabs
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        tab_c1, tab_c2 = st.tabs(["📉 Price & Flow Action", "🌊 Cumulative Foreign"])
        
        with tab_c1:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05)
            # Price
            fig.add_trace(go.Scatter(x=stock_df['Last Trading Date'], y=stock_df['Close'], name='Price', line=dict(color='#4318FF', width=2)), row=1, col=1)
            # NFF
            colors_nff = ['#00C853' if v >= 0 else '#FF3D00' for v in stock_df['NFF (Rp)']]
            fig.add_trace(go.Bar(x=stock_df['Last Trading Date'], y=stock_df['NFF (Rp)'], name='Net Foreign', marker_color=colors_nff), row=2, col=1)
            # Vol
            fig.add_trace(go.Bar(x=stock_df['Last Trading Date'], y=stock_df['Volume'], name='Volume', marker_color='#A3AED0'), row=3, col=1)
            
            fig.update_layout(height=600, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='white', plot_bgcolor='white')
            fig.update_xaxes(showgrid=True, gridcolor='#F4F7FE')
            fig.update_yaxes(showgrid=True, gridcolor='#F4F7FE')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab_c2:
            stock_df['Cum NFF'] = stock_df['NFF (Rp)'].cumsum()
            fig2 = px.area(stock_df, x='Last Trading Date', y='Cum NFF', title='Cumulative Foreign Flow (YTD)')
            fig2.update_layout(plot_bgcolor='white', font=dict(color='#2B3674'))
            # Fix: use 'fillcolor' instead of 'fill_color' to avoid error
            fig2.update_traces(line_color='#4318FF', fillcolor='rgba(67, 24, 255, 0.1)')
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 3: TOP PICKS
elif menu == "🏆 Algorithmic Picks":
    st.markdown(f"""
    <div class="header-banner">
        <div class="header-title">🏆 Top 20 Potential Stocks</div>
        <div class="header-subtitle">Scoring based on Trend, Momentum, and Big Money Flow</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_run1, col_run2 = st.columns([1, 4])
    with col_run1:
        if st.button("🚀 Run Scoring Algorithm", type="primary"):
            st.session_state['run_score'] = True
    
    if st.session_state.get('run_score'):
        top20, msg, status = calculate_potential_score(df, pd.Timestamp(sel_date))
        
        if not top20.empty:
            # Layout: Leaderboard left, Details right
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                st.markdown('<div class="css-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📋 Leaderboard</div>', unsafe_allow_html=True)
                
                # --- FIX: SELECT & RENAME COLUMNS FOR CLEAN DISPLAY ---
                display_cols = ['Rank', 'Stock Code', 'last_price', 'total_net_ff_rp', 'total_money_flow', 'avg_change_pct', 'sector', 'Trend Score', 'Momentum Score', 'Potential Score']
                
                st.dataframe(
                    top20[display_cols],
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                        "Stock Code": "Stock",
                        "last_price": st.column_config.NumberColumn("Close Price", format="Rp %,d"),
                        "total_net_ff_rp": st.column_config.NumberColumn("Net Foreign", format="Rp %,d"),
                        "total_money_flow": st.column_config.NumberColumn("Money Flow", format="Rp %,d"),
                        "avg_change_pct": st.column_config.NumberColumn("Avg Chg", format="%.2f %%"),
                        "sector": "Sector",
                        "Trend Score": st.column_config.NumberColumn("Trend", format="%.1f"),
                        "Momentum Score": st.column_config.NumberColumn("Momentum", format="%.1f"),
                        "Potential Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=100, format="%.1f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=600
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_res2:
                # Top 1 Highlight
                best = top20.iloc[0]
                st.markdown('<div class="css-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">🥇 Top Pick: {best["Stock Code"]}</div>', unsafe_allow_html=True)
                
                # Radar Chart for Score Breakdown
                categories = ['Trend', 'Momentum', 'NBSA', 'Foreign']
                values = [best['Trend Score'], best['Momentum Score'], best['NBSA Score'], best['Foreign Contrib Score']]
                
                fig = px.line_polar(r=values, theta=categories, line_close=True)
                fig.update_traces(fill='toself', line_color='#4318FF')
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Total Score", f"{best['Potential Score']:.1f}")
                st.metric("Last Price", f"Rp {best['last_price']:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)

# PAGE 4: PORTFOLIO
elif menu == "💼 Portfolio Simulator":
    st.markdown('<h2 style="color:#2B3674;">💼 Portfolio Backtest Simulator</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    avail_dates = sorted(df['Last Trading Date'].unique())
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: start_d = st.selectbox("📅 Tanggal Beli", avail_dates, index=max(0, len(avail_dates)-31), format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y'))
    with c2: end_d = st.selectbox("📅 Tanggal Jual", avail_dates, index=len(avail_dates)-1, format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y'))
    with c3: cap = st.number_input("💰 Modal Awal (Rp)", 1_000_000, 10_000_000_000, 100_000_000, 1_000_000)
    with c4: 
        st.write("") 
        st.write("")
        sim_btn = st.button("🚀 Mulai Simulasi", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if sim_btn:
        if pd.Timestamp(start_d) >= pd.Timestamp(end_d): st.error("Tanggal Jual harus setelah Beli.")
        else:
            with st.spinner("Menghitung PnL..."):
                df_port, sum_port, msg = simulate_portfolio_range(df, cap, pd.Timestamp(start_d), pd.Timestamp(end_d))
                
                if msg == "success":
                    # Result Cards
                    r1, r2, r3 = st.columns(3)
                    r1.markdown(f"""<div class="css-card"><div style="color:#A3AED0;">Final Value</div><div style="font-size:24px; font-weight:bold; color:#2B3674;">Rp {sum_port['Final Portfolio Value']:,.0f}</div></div>""", unsafe_allow_html=True)
                    r2.markdown(f"""<div class="css-card"><div style="color:#A3AED0;">Net Profit</div><div style="font-size:24px; font-weight:bold; color:{'green' if sum_port['Net Profit']>0 else 'red'};">Rp {sum_port['Net Profit']:,.0f}</div></div>""", unsafe_allow_html=True)
                    r3.markdown(f"""<div class="css-card"><div style="color:#A3AED0;">ROI</div><div style="font-size:24px; font-weight:bold; color:#4318FF;">{sum_port['Total ROI']:.2f}%</div></div>""", unsafe_allow_html=True)
                    
                    st.markdown('<div class="css-card">', unsafe_allow_html=True)
                    st.dataframe(
                        df_port[['Stock Code', 'Sector', 'Buy Price', 'Sell Price', 'Gain/Loss (Rp)', 'ROI (%)']], 
                        use_container_width=True,
                        column_config={
                            "Buy Price": st.column_config.NumberColumn(format="Rp %,d"),
                            "Sell Price": st.column_config.NumberColumn(format="Rp %,d"),
                            "Gain/Loss (Rp)": st.column_config.NumberColumn(format="Rp %,d"),
                            "ROI (%)": st.column_config.NumberColumn(format="%.2f %%")
                        }
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

# PAGE 5: MSCI
elif menu == "🌏 MSCI Projector":
    st.markdown('<h2 style="color:#2B3674;">🌏 MSCI Indonesia Proxy</h2>', unsafe_allow_html=True)
    st.info("Prediksi kandidat masuk indeks MSCI berdasarkan Float Market Cap & Likuiditas (ATVR).")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    col_in1, col_in2 = st.columns(2)
    with col_in1: usd = st.number_input("Kurs USD/IDR", value=16200, step=50)
    with col_in2: 
        st.write("")
        calc_btn = st.button("Hitung Proyeksi", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if calc_btn:
        df_msci = calculate_msci_projection_v2(df, df['Last Trading Date'].max(), usd)
        
        # Categorization logic inline
        def cat_msci(r):
            if r['Float Cap ($B)'] >= 1.5 and r['ATVR 12M (%)'] >= 15: return "✅ Potential Standard"
            elif r['Float Cap ($B)'] >= 1.5: return "⚠️ Illiquid (Risk)"
            elif r['Float Cap ($B)'] >= 0.8: return "🔹 Small Cap"
            else: return "🔻 Micro"
        
        df_msci['Status'] = df_msci.apply(cat_msci, axis=1)
        
        # Scatter Plot
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        fig = px.scatter(
            df_msci.head(100), 
            x="ATVR 12M (%)", y="Float Cap ($B)", 
            color="Status", size="Full Cap ($B)", hover_name="Stock Code",
            color_discrete_map={"✅ Potential Standard": "#00C853", "⚠️ Illiquid (Risk)": "#FF3D00", "🔹 Small Cap": "#2962FF", "🔻 Micro": "#B0BEC5"}
        )
        fig.add_hline(y=1.5, line_dash="dash", annotation_text="Standard Threshold ($1.5B)")
        fig.add_vline(x=15, line_dash="dash", annotation_text="Liquidity (15%)")
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Candidate List")
        st.dataframe(
            df_msci[df_msci['Status'] == "✅ Potential Standard"][['Stock Code', 'Status', 'Float Cap ($B)', 'ATVR 12M (%)']], 
            use_container_width=True,
            column_config={
                "Float Cap ($B)": st.column_config.NumberColumn(format="$ %.2f B"), 
                "ATVR 12M (%)": st.column_config.NumberColumn(format="%.2f %%")
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<center style='color:#A3AED0;'> IDX Quant Pro v2.0 • Powered by Streamlit </center>", unsafe_allow_html=True)
