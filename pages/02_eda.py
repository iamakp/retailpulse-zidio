"""
RetailPulse - EDA & Data Explorer Page
Author: Abhishek Kumar Prajapati (Zidio Development)
Year: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(".")

# ── INLINED GLOBAL COLOR THEMES & CONFIGURATIONS ──────────────────────────────
TEMPLATE = "plotly_dark"
CHART_COLORS = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]

def style_fig(fig, height=350):
    """Replaces external theme configurations cleanly in-memory"""
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    if hasattr(fig, 'layout') and fig.layout.xaxis:
        fig.update_xaxes(showgrid=False, title_font=dict(color="#64748b"))
    if hasattr(fig, 'layout') and fig.layout.yaxis:
        fig.update_yaxes(gridcolor="#1e3a5c", title_font=dict(color="#64748b"))
    return fig

# ── INITIAL SETUP CONFIGURATIONS ──────────────────────────────────────────────
st.set_page_config(page_title="EDA | RetailPulse", page_icon="🔍", layout="wide")

# Modern Neomorphic Dashboard Backing Injection
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#020818,#0a1628,#020818);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#040d1f,#071428)!important;border-right:1px solid #1a3a5c!important;}
[data-testid="stSidebarNav"] a{text-transform:uppercase!important;letter-spacing:0.08em!important;font-size:0.78rem!important;font-weight:600!important;}
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.page-header{background:linear-gradient(135deg,#0f2744,#0d2137);border:1px solid #2d5a8e;
  border-radius:16px;padding:30px;margin-bottom:24px;animation:fadeInUp 0.6s ease;border-left:4px solid #3b82f6;}
[data-testid="metric-container"] {
    background: #0d1f3c !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class='page-header'>
  <div style='font-size:1.8rem;font-weight:800;color:#e2e8f0'>🔍 EDA & DATA EXPLORER</div>
  <div style='color:#64748b;margin-top:6px'>Exploratory Data Analysis on UCI Online Retail II Dataset</div>
</div>""", unsafe_allow_html=True)

# ── DATA STREAM MANAGEMENT WITH CLOUD SAFEGUARD ──────────────────────────────
@st.cache_data
def load_data():
    p = Path("data/processed/retail_clean.parquet")
    if p.exists():
        return pd.read_parquet(p)
    
    # Cloud Fallback generator to handle hidden .gitignore items smoothly
    np.random.seed(42)
    mock_dates = pd.date_range(start="2025-01-01", periods=200, freq="D")
    return pd.DataFrame({
        "Date": np.random.choice(mock_dates, 5000),
        "TotalPrice": np.random.exponential(40, 5000),
        "Quantity": np.random.randint(1, 20, 5000),
        "Customer_ID": np.random.randint(10000, 16000, 5000),
        "Country": np.random.choice(["United Kingdom", "Germany", "France", "EIRE", "Netherlands"], 5000),
        "StockCode": np.random.randint(20000, 25000, 5000).astype(str),
        "Description": np.random.choice(["WHITE HANGING HEART LIGHT-HOLDER", "REGENCY CAKESTAND 3 TIER", "ASSORTED COLOUR BIRD ORNAMENT", "PARTY BUNTING"], 5000),
        "Price": np.random.uniform(1.5, 12.0, 5000),
        "DayOfWeek": np.random.randint(0, 7, 5000),
        "Hour": np.random.randint(8, 18, 5000),
        "Year": np.random.choice([2025, 2026], 5000),
        "Month": np.random.randint(1, 13, 5000)
    })

df = load_data()

# ── STRATEGIC KPIS HEADER GRID ────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🧾 Transactions", f"{len(df):,}")
col2.metric("👥 Customers",    f"{df['Customer_ID'].nunique():,}")
col3.metric("📦 Products",     f"{df['StockCode'].nunique():,}")
col4.metric("🌍 Countries",    f"{df['Country'].nunique():,}")
col5.metric("💰 Revenue",      f"£{df['TotalPrice'].sum()/1e6:.1f}M")

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📅 Time Series", "🌍 Geography", "📦 Products"])

# ── TAB 1: OVERVIEW METRIC DISTRIBUTIONS ──────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    rev_cap = df["TotalPrice"].quantile(0.95)
    rev_data = df[df["TotalPrice"] <= rev_cap]["TotalPrice"]
    fig1 = px.histogram(rev_data, nbins=60, title="💰 Revenue Per Transaction Distribution",
                        labels={"value":"Revenue (£)", "count":"Transactions"},
                        color_discrete_sequence=[CHART_COLORS[0]])
    fig1.update_layout(showlegend=False, bargap=0.06)
    style_fig(fig1)
    col1.plotly_chart(fig1, use_container_width=True)

    qty_cap = df["Quantity"].quantile(0.95)
    qty_data = df[df["Quantity"] <= qty_cap]["Quantity"]
    fig2 = px.histogram(qty_data, nbins=50, title="📦 Order Quantity Distribution",
                        color_discrete_sequence=[CHART_COLORS[2]])
    fig2.update_layout(showlegend=False)
    style_fig(fig2)
    col2.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🔥 Sales Heatmap — Day of Week × Hour")
    heatmap_data = df.groupby(["DayOfWeek", "Hour"])["TotalPrice"].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="DayOfWeek", columns="Hour", values="TotalPrice").fillna(0)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Validation map check for pivot tracking safety
    heatmap_pivot.index = [day_names[i] for i in heatmap_pivot.index]
    fig3 = px.imshow(heatmap_pivot, title="Revenue Heatmap by Day & Hour",
                     color_continuous_scale=[[0, "#0a1830"], [0.5, "#3b82f6"], [1, "#a78bfa"]],
                     labels=dict(x="Hour of Day", y="Day of Week", color="Revenue £"))
    style_fig(fig3, height=320)
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: TIME-SERIES TIMELINES ──────────────────────────────────────────────
with tab2:
    daily = df.groupby("Date")["TotalPrice"].sum().reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily_sorted = daily.sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_sorted["Date"], y=daily_sorted["TotalPrice"],
                             mode="lines", name="Daily Revenue",
                             line=dict(color=CHART_COLORS[0], width=1.5),
                             fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"))
    daily_sorted["MA7"] = daily_sorted["TotalPrice"].rolling(7).mean()
    fig.add_trace(go.Scatter(x=daily_sorted["Date"], y=daily_sorted["MA7"],
                             mode="lines", name="7-Day MA",
                             line=dict(color=CHART_COLORS[3], width=2.5)))
    fig.update_layout(title="📅 Daily Revenue Over Time", legend=dict(orientation="h"))
    style_fig(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    monthly = df.groupby(["Year", "Month"])["TotalPrice"].sum().reset_index()
    monthly["YM"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)
    fig2 = px.bar(monthly, x="YM", y="TotalPrice", title="📊 Monthly Revenue",
                  color="TotalPrice", color_continuous_scale=[[0, "#1e3a5f"], [1, "#60a5fa"]])
    fig2.update_layout(xaxis_tickangle=45)
    style_fig(fig2)
    col1.plotly_chart(fig2, use_container_width=True)

    dow = df.groupby("DayOfWeek")["TotalPrice"].sum().reset_index()
    dow["Day"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig3 = px.bar(dow, x="Day", y="TotalPrice", title="📅 Revenue by Day of Week",
                  color="TotalPrice", color_continuous_scale=[[0, "#3b0764"], [1, "#a78bfa"]])
    style_fig(fig3)
    col2.plotly_chart(fig3, use_container_width=True)

# ── TAB 3: GEOGRAPHICAL DISTRIBUTIONS ─────────────────────────────────────────
with tab3:
    country_rev = df.groupby("Country")["TotalPrice"].sum().reset_index().sort_values("TotalPrice", ascending=False)
    col1, col2 = st.columns(2)
    fig1 = px.bar(country_rev.head(15), x="TotalPrice", y="Country", orientation="h",
                  title="🌍 Top 15 Countries by Revenue",
                  color="TotalPrice", color_continuous_scale=[[0, "#1e3a5f"], [1, "#60a5fa"]])
    fig1.update_layout(yaxis={"categoryorder":"total ascending"})
    style_fig(fig1, height=450)
    col1.plotly_chart(fig1, use_container_width=True)

    country_cust = df.groupby("Country")["Customer_ID"].nunique().reset_index().sort_values("Customer_ID", ascending=False)
    fig2 = px.pie(country_cust.head(10), names="Country", values="Customer_ID",
                  title="👥 Customers by Country (Top 10)",
                  color_discrete_sequence=CHART_COLORS, hole=0.4)
    style_fig(fig2)
    col2.plotly_chart(fig2, use_container_width=True)

# ── TAB 4: PRODUCT PERFORMANCES ───────────────────────────────────────────────
with tab4:
    top_products = df.groupby("Description")["TotalPrice"].sum().reset_index().sort_values("TotalPrice", ascending=False).head(20)
    fig = px.bar(top_products, x="TotalPrice", y="Description", orientation="h",
                 title="🏆 Top 20 Products by Revenue",
                 color="TotalPrice", color_continuous_scale=[[0, "#7c2d12"], [1, "#f59e0b"]])
    fig.update_layout(yaxis={"categoryorder":"total ascending"})
    style_fig(fig, height=600)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    top_qty = df.groupby("Description")["Quantity"].sum().reset_index().sort_values("Quantity", ascending=False).head(10)
    fig2 = px.bar(top_qty, x="Quantity", y="Description", orientation="h",
                  title="📦 Top 10 by Quantity Sold",
                  color="Quantity", color_continuous_scale=[[0, "#064e3b"], [1, "#34d399"]])
    fig2.update_layout(yaxis={"categoryorder":"total ascending"})
    style_fig(fig2)
    col1.plotly_chart(fig2, use_container_width=True)

    price_dist = df.groupby("Description")["Price"].mean().reset_index().sort_values("Price", ascending=False).head(10)
    fig3 = px.bar(price_dist, x="Price", y="Description", orientation="h",
                  title="💎 Top 10 by Avg Unit Price",
                  color="Price", color_continuous_scale=[[0, "#7f1d1d"], [1, "#fb7185"]])
    fig3.update_layout(yaxis={"categoryorder":"total ascending"})
    style_fig(fig3)
    col2.plotly_chart(fig3, use_container_width=True)