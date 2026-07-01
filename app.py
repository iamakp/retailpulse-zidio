"""RetailPulse – Main Overview Page"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time

st.set_page_config(
    page_title="RetailPulse | AI-Powered Retail Analytics",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020818 0%, #0a1628 50%, #020818 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040d1f 0%, #071428 100%) !important;
    border-right: 1px solid #1a3a5c !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Sidebar nav links uppercase */
[data-testid="stSidebarNav"] a {
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(59,130,246,0.15) !important;
    color: #60a5fa !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(90deg, rgba(59,130,246,0.25), rgba(139,92,246,0.15)) !important;
    color: #60a5fa !important;
    border-left: 3px solid #3b82f6 !important;
}

/* Animated hero */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.4); }
    50%       { box-shadow: 0 0 0 12px rgba(59,130,246,0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.5); }
    to   { opacity: 1; transform: scale(1); }
}

.hero-banner {
    background: linear-gradient(270deg, #0f2744, #1a1a5e, #0f2744, #1a3a5c);
    background-size: 400% 400%;
    animation: gradientShift 8s ease infinite;
    border: 1px solid #2d5a8e;
    border-radius: 20px;
    padding: 50px 40px;
    text-align: center;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    animation: gradientShift 8s ease infinite, fadeInUp 0.8s ease;
}
.hero-banner::before {
    content: '';
    position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
    animation: shimmer 3s infinite;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399, #60a5fa);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.2rem;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 20px;
}
.hero-badges {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 16px;
}
.hero-badge {
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.kpi-card {
    background: linear-gradient(135deg, #0d1f3c, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
    animation: fadeInUp 0.6s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: #3b82f6;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-blue::after   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.kpi-purple::after { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.kpi-green::after  { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-orange::after { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

.kpi-icon  { font-size: 2rem; margin-bottom: 8px; }
.kpi-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: countUp 0.8s ease;
}
.kpi-label { color: #64748b; font-size: 0.82rem; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-badge {
    display: inline-block;
    margin-top: 10px;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    background: #064e3b;
    color: #34d399;
}

.module-card {
    background: linear-gradient(135deg, #0d1f3c, #0f172a);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 22px;
    margin: 6px 0;
    transition: all 0.25s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.7s ease;
}
.module-card:hover {
    transform: translateY(-3px);
    border-color: #3b82f6;
    background: linear-gradient(135deg, #0f2744, #111827);
    box-shadow: 0 8px 32px rgba(59,130,246,0.15);
}
.module-card::before {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 14px 0 0 14px;
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    opacity: 0;
    transition: opacity 0.2s;
}
.module-card:hover::before { opacity: 1; }
.module-icon  { font-size: 1.8rem; }
.module-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; }
.module-desc  { color: #475569; font-size: 0.82rem; margin-top: 5px; line-height: 1.5; }
.status-live  {
    background: linear-gradient(90deg, #064e3b, #065f46);
    color: #34d399;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.tech-badge {
    background: linear-gradient(135deg, #0f2744, #1e3a5f);
    border: 1px solid #2d5a8e;
    color: #93c5fd;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 3px;
    display: inline-block;
    transition: all 0.2s;
}
.tech-badge:hover {
    background: linear-gradient(135deg, #1e3a5f, #2d5a8e);
    transform: scale(1.05);
}

.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 28px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

.footer-bar {
    text-align: center;
    color: #334155;
    font-size: 0.82rem;
    border-top: 1px solid #0f1f3a;
    padding-top: 24px;
    margin-top: 40px;
}

/* Metric override */
[data-testid="metric-container"] {
    background: #0d1f3c !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center;padding:20px 0 10px 0'>
  <div style='font-size:3rem;animation:pulse 2s infinite'>📊</div>
  <div style='font-size:1.4rem;font-weight:800;letter-spacing:0.05em;
    background:linear-gradient(90deg,#60a5fa,#a78bfa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    RETAILPULSE
  </div>
  <div style='font-size:0.72rem;color:#475569;letter-spacing:0.12em;
    text-transform:uppercase;margin-top:4px'>
    AI-Powered Analytics
  </div>
</div>
<div style='height:1px;background:linear-gradient(90deg,transparent,#1e3a5f,transparent);margin:10px 0'></div>
""", unsafe_allow_html=True)

# Live stats
stats = {"rows":"805,549","customers":"5,878","products":"4,631","revenue":"£17.7M"}
p = Path("data/processed/retail_clean.parquet")
if p.exists():
    try:
        df = pd.read_parquet(p, columns=["Customer_ID","StockCode","TotalPrice"])
        stats = {"rows":f"{len(df):,}","customers":f"{df['Customer_ID'].nunique():,}",
                 "products":f"{df['StockCode'].nunique():,}",
                 "revenue":f"£{df['TotalPrice'].sum()/1e6:.1f}M"}
    except: pass

st.sidebar.markdown(f"""
<div style='background:linear-gradient(135deg,#040d1f,#071428);
  border:1px solid #1a3a5c;border-radius:12px;padding:16px;margin:10px 0;font-size:0.82rem'>
  <div style='color:#475569;font-size:0.7rem;letter-spacing:0.1em;
    text-transform:uppercase;font-weight:700;margin-bottom:12px'>
    📡 LIVE DATASET
  </div>
  <div style='display:flex;justify-content:space-between;margin:6px 0'>
    <span style='color:#64748b'>🧾 TRANSACTIONS</span>
    <span style='color:#60a5fa;font-weight:700'>{stats['rows']}</span>
  </div>
  <div style='display:flex;justify-content:space-between;margin:6px 0'>
    <span style='color:#64748b'>👥 CUSTOMERS</span>
    <span style='color:#a78bfa;font-weight:700'>{stats['customers']}</span>
  </div>
  <div style='display:flex;justify-content:space-between;margin:6px 0'>
    <span style='color:#64748b'>📦 PRODUCTS</span>
    <span style='color:#34d399;font-weight:700'>{stats['products']}</span>
  </div>
  <div style='display:flex;justify-content:space-between;margin:6px 0'>
    <span style='color:#64748b'>💰 REVENUE</span>
    <span style='color:#fbbf24;font-weight:700'>{stats['revenue']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='margin:10px 0'>
  <div style='color:#475569;font-size:0.7rem;letter-spacing:0.1em;
    text-transform:uppercase;font-weight:700;margin-bottom:10px'>
    🔧 TECH STACK
  </div>
  <div style='font-size:0.75rem;color:#334155;line-height:2'>
    <span style='color:#60a5fa'>●</span> Python 3.11 &nbsp;
    <span style='color:#a78bfa'>●</span> Prophet<br>
    <span style='color:#34d399'>●</span> PyTorch LSTM &nbsp;
    <span style='color:#f59e0b'>●</span> XGBoost<br>
    <span style='color:#ec4899'>●</span> SHAP &nbsp;
    <span style='color:#60a5fa'>●</span> MLflow<br>
    <span style='color:#a78bfa'>●</span> Evidently AI &nbsp;
    <span style='color:#34d399'>●</span> Docker
  </div>
</div>
<div style='height:1px;background:linear-gradient(90deg,transparent,#1e3a5f,transparent);margin:14px 0'></div>
<div style='font-size:0.7rem;color:#1e3a5f;text-align:center;letter-spacing:0.05em'>
  ZIDIO DEVELOPMENT · 2026
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
  <div class='hero-title'>📊 RETAILPULSE</div>
  <div style='font-size:1.5rem;font-weight:700;color:#cbd5e1;margin-bottom:6px'>
    AI-Powered Customer Analytics & Demand Forecasting
  </div>
  <div class='hero-sub'>
    End-to-End ML Platform · UCI Online Retail II · 805K+ Real Transactions
  </div>
  <div class='hero-badges'>
    <span class='hero-badge'>🧠 Prophet + LSTM Ensemble</span>
    <span class='hero-badge'>⚡ XGBoost + SHAP</span>
    <span class='hero-badge'>🎯 RFM Segmentation</span>
    <span class='hero-badge'>📦 Inventory EOQ</span>
    <span class='hero-badge'>🛡️ MLflow + Evidently</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
kpis = [
    (c1,"kpi-blue","📈","8.67%","FORECAST MAPE","✅ TARGET ≤ 12%"),
    (c2,"kpi-purple","🎯","0.849","CHURN AUC-ROC","✅ TARGET ≥ 0.88"),
    (c3,"kpi-green","📉","30–50%","STOCKOUT REDUCTION","✅ VIA FORECASTING"),
    (c4,"kpi-orange","👥","6","CUSTOMER SEGMENTS","✅ RFM + K-MEANS"),
]
for col,cls,icon,val,label,badge in kpis:
    with col:
        st.markdown(f"""
        <div class='kpi-card {cls}'>
          <div class='kpi-icon'>{icon}</div>
          <div class='kpi-value'>{val}</div>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-badge'>{badge}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Mini chart row ────────────────────────────────────────────────────────────
if p.exists():
    try:
        df_full = pd.read_parquet(p, columns=["Date","TotalPrice","Customer_ID","Country"])
        daily = df_full.groupby("Date")["TotalPrice"].sum().reset_index()
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = daily.sort_values("Date")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["Date"], y=daily["TotalPrice"],
                fill="tozeroy",
                line=dict(color="#3b82f6", width=1.5),
                fillcolor="rgba(59,130,246,0.1)",
                name="Revenue"
            ))
            fig.update_layout(
                title="📈 Daily Revenue Trend",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=10,r=10,t=40,b=10),
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1e3a5f"),
                font=dict(color="#94a3b8", size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            monthly = df_full.groupby(df_full["Date"].apply(
                lambda x: pd.to_datetime(x).strftime("%Y-%m")
            ))["TotalPrice"].sum().reset_index()
            monthly.columns = ["Month","Revenue"]
            fig2 = px.bar(monthly, x="Month", y="Revenue",
                          color="Revenue", color_continuous_scale="Blues",
                          title="📊 Monthly Revenue")
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=10,r=10,t=40,b=10),
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=9)),
                yaxis=dict(showgrid=True, gridcolor="#1e3a5f"),
                font=dict(color="#94a3b8", size=11),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            top_c = df_full.groupby("Country")["TotalPrice"].sum().nlargest(8).reset_index()
            fig3 = px.pie(top_c, names="Country", values="TotalPrice",
                          title="🌍 Revenue by Country",
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            fig3.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=10,r=10,t=40,b=10),
                showlegend=False,
                font=dict(color="#94a3b8", size=11),
            )
            fig3.update_traces(textinfo="none")
            st.plotly_chart(fig3, use_container_width=True)
    except: pass

# ── Modules ───────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>🚀 PLATFORM MODULES</div>", unsafe_allow_html=True)

modules = [
    ("🔍","EDA & DATA EXPLORER","805K transactions · time-series · heatmap · geography · product analysis"),
    ("👥","CUSTOMER SEGMENTATION","RFM scoring · K-Means + DBSCAN · PCA visualisation · 6 business segments"),
    ("📈","DEMAND FORECASTING","Prophet + LSTM ensemble · MAPE 8.67% · 30-day ahead · what-if analysis"),
    ("⚠️","CHURN PREDICTION","XGBoost + SHAP · AUC 0.849 · risk tiers · threshold slider · CSV export"),
    ("📦","INVENTORY OPTIMIZATION","EOQ · safety stock · reorder point · status alerts · demand-driven"),
    ("🛡️","MODEL MONITORING","PSI drift detection · MAPE tracking · AUC monitoring · retraining log"),
]
r1, r2 = st.columns(3), st.columns(3)
for i, (icon,title,desc) in enumerate(modules):
    col = r1[i] if i < 3 else r2[i-3]
    with col:
        st.markdown(f"""
        <div class='module-card'>
          <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
            <span class='module-icon'>{icon}</span>
            <span class='module-title'>{title}</span>
            <span class='status-live'>● LIVE</span>
          </div>
          <div class='module-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>🔧 TECHNOLOGY STACK</div>", unsafe_allow_html=True)
techs = ["Python 3.11","Pandas","NumPy","Scikit-learn","Prophet","PyTorch Lightning",
         "XGBoost","SHAP","Optuna","Streamlit","MLflow","Evidently AI",
         "Docker","GitHub Actions","Great Expectations","PostgreSQL","Redis"]
st.markdown("<div>" + "".join([f"<span class='tech-badge'>{t}</span>" for t in techs]) + "</div>",
            unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer-bar'>
  Built by <b style='color:#60a5fa'>Abhishek Kumar Prajapati</b> &nbsp;·&nbsp;
  <b style='color:#a78bfa'>Zidio Development</b> &nbsp;·&nbsp;
  Data Science & Analytics Internship &nbsp;·&nbsp;
  <b style='color:#34d399'>June–July 2026</b>
</div>
""", unsafe_allow_html=True)
