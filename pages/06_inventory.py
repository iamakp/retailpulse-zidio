"""Page 6 – Inventory Optimization"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
sys.path.append(".")

st.title("📦 Inventory Optimization")
st.markdown("EOQ · Safety Stock · Reorder Point | Target: Stockout Reduction 30–50%")

@st.cache_data
def load_inventory():
    p = Path("data/processed/inventory_recs.parquet")
    if p.exists():
        return pd.read_parquet(p)
    # Demo fallback
    np.random.seed(42)
    n = 100
    statuses = np.random.choice(["🔴 Reorder Now","🟢 OK","🟡 Overstock Risk"], n, p=[0.2,0.6,0.2])
    return pd.DataFrame({
        "StockCode": [f"SC{i:04d}" for i in range(n)],
        "Description": [f"Product {i}" for i in range(n)],
        "AvgDailyDemand": np.random.exponential(5, n).round(2),
        "DemandStd": np.random.exponential(2, n).round(2),
        "SafetyStock": np.random.randint(5, 50, n).astype(float),
        "ReorderPoint": np.random.randint(20, 150, n).astype(float),
        "EOQ": np.random.randint(50, 500, n).astype(float),
        "Forecasted30d": np.random.randint(50, 500, n).astype(float),
        "EstimatedCurrentStock": np.random.randint(10, 300, n).astype(float),
        "Status": statuses,
    })

df = load_inventory()

# KPIs
reorder = (df["Status"] == "🔴 Reorder Now").sum()
overstock = (df["Status"] == "🟡 Overstock Risk").sum()
ok = (df["Status"] == "🟢 OK").sum()

col1,col2,col3,col4 = st.columns(4)
col1.metric("🔴 Reorder Alerts", reorder)
col2.metric("🟡 Overstock Risks", overstock)
col3.metric("🟢 Stock OK", ok)
col4.metric("📦 Total Products", len(df))

st.markdown("---")
tab1,tab2,tab3 = st.tabs(["🚨 Alerts","📊 Analytics","📋 Full Table"])

with tab1:
    reorder_df = df[df["Status"]=="🔴 Reorder Now"].sort_values("AvgDailyDemand", ascending=False)
    st.subheader(f"🔴 {len(reorder_df)} Products Need Reordering")
    if len(reorder_df):
        fig = px.bar(reorder_df.head(20), x="Description", y="ReorderPoint",
                     title="Top 20 Products by Reorder Point",
                     color="AvgDailyDemand", color_continuous_scale="Reds")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(reorder_df[["StockCode","Description","AvgDailyDemand",
                                  "SafetyStock","ReorderPoint","EOQ","Status"]].head(50),
                     use_container_width=True)

with tab2:
    col1,col2 = st.columns(2)
    status_counts = df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status","Count"]
    fig1 = px.pie(status_counts, names="Status", values="Count", title="Inventory Status Distribution")
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="AvgDailyDemand", y="SafetyStock",
                      title="Daily Demand vs Safety Stock",
                      color="Status", size="EOQ",
                      color_discrete_map={"🔴 Reorder Now":"red","🟢 OK":"green","🟡 Overstock Risk":"orange"})
    col2.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df, x="EOQ", nbins=30, title="EOQ Distribution",
                        color_discrete_sequence=["#3b82f6"])
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    status_filter = st.selectbox("Filter by status", ["All","🔴 Reorder Now","🟢 OK","🟡 Overstock Risk"])
    filtered = df if status_filter == "All" else df[df["Status"]==status_filter]
    st.dataframe(filtered, use_container_width=True)
    st.download_button("⬇️ Export Inventory Report", filtered.to_csv(index=False),
                       "inventory_report.csv", "text/csv")
