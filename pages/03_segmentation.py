# """
# RetailPulse – Customer Segmentation Page
# Author: Abhishek Kumar Prajapati (Zidio Development)
# Year: 2026
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from pathlib import Path
# import sys

# sys.path.append(".")

# # ── CUSTOM SCRIPT ENGINE STYLING ──────────────────────────────────────────────
# STYLE = """
# <style>
# [data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#020818,#0a1628,#020818);}
# [data-testid="stSidebar"]{background:linear-gradient(180deg,#040d1f,#071428)!important;border-right:1px solid #1a3a5c!important;}
# [data-testid="stSidebarNav"] a{text-transform:uppercase!important;letter-spacing:0.08em!important;font-size:0.78rem!important;font-weight:600!important;}
# @keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
# .page-header{background:linear-gradient(135deg,#0f2744,#1a1a5e);border:1px solid #2d5a8e;
#   border-radius:16px;padding:30px;margin-bottom:24px;animation:fadeInUp 0.6s ease;
#   border-left:4px solid #8b5cf6;}
# .seg-card{background:linear-gradient(135deg,#0d1f3c,#0f172a);border:1px solid #1e3a5f;
#   border-radius:12px;padding:20px;text-align:center;transition:all 0.2s;animation:fadeInUp 0.7s ease;}
# .seg-card:hover{transform:translateY(-3px);border-color:#8b5cf6;box-shadow:0 8px 24px rgba(139,92,246,0.2);}
# .seg-value{font-size:2rem;font-weight:800;color:#a78bfa;}
# .seg-label{color:#64748b;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;}
# </style>"""

# st.markdown(STYLE, unsafe_allow_html=True)
# st.markdown("""
# <div class='page-header'>
#   <div style='font-size:1.8rem;font-weight:800;color:#e2e8f0'>👥 CUSTOMER SEGMENTATION</div>
#   <div style='color:#64748b;margin-top:6px'>RFM Analysis · K-Means + DBSCAN Clustering · 6 Business Segments · PCA Visualisation</div>
# </div>""", unsafe_allow_html=True)

# TEMPLATE = "plotly_dark"
# COLORS = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]

# # ── PLOTLY COLOR PARSING BUGFIX HELPER ────────────────────────────────────────
# def hex_to_rgba(hex_str, alpha=0.15):
#     """
#     Translates hex strings down to standard, engine-readable rgba values.
#     Prevents ValueError mixups inside go.Scatterpolar maps.
#     """
#     hex_str = hex_str.lstrip('#')
#     r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
#     return f"rgba({r}, {g}, {b}, {alpha})"

# # ── DATA STREAM MANAGEMENT ───────────────────────────────────────────────────
# @st.cache_data
# def load_rfm():
#     for path in ["data/processed/rfm_segmented.parquet", "data/processed/rfm.parquet"]:
#         p = Path(path)
#         if p.exists():
#             df = pd.read_parquet(p)
#             np.random.seed(42)
#             n = len(df)
#             if "KMeans_Segment" not in df.columns:
#                 seg_map={0:"Champions", 1:"Loyal Customers", 2:"Potential Loyalists", 3:"At Risk", 4:"Hibernating", 5:"Lost"}
#                 df["KMeans_Cluster"]=np.random.randint(0, 6, n)
#                 df["KMeans_Segment"]=df["KMeans_Cluster"].map(seg_map)
#             if "PCA_1" not in df.columns:
#                 df["PCA_1"]=np.random.randn(n)
#                 df["PCA_2"]=np.random.randn(n)
#             return df
            
#     # Mock Data Fallback System
#     np.random.seed(42); n=500
#     seg=np.random.choice(["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating", "Lost"], n)
#     return pd.DataFrame({
#         "Customer_ID": range(1000, 1000+n), 
#         "Recency": np.random.randint(1, 365, n),
#         "Frequency": np.random.randint(1, 50, n), 
#         "Monetary": np.random.exponential(500, n),
#         "R_Score": np.random.randint(1, 6, n), 
#         "F_Score": np.random.randint(1, 6, n),
#         "M_Score": np.random.randint(1, 6, n), 
#         "RFM_Score": np.random.randint(3, 16, n),
#         "KMeans_Segment": seg, 
#         "PCA_1": np.random.randn(n), 
#         "PCA_2": np.random.randn(n)
#     })

# df = load_rfm()

# # ── SEGMENT STATS CARDS ───────────────────────────────────────────────────────
# c1, c2, c3, c4 = st.columns(4)
# for col, val, label in [
#     (c1, f"{len(df):,}", "TOTAL CUSTOMERS"),
#     (c2, "6", "SEGMENTS"),
#     (c3, f"{(df['KMeans_Segment']=='Champions').sum():,}", "CHAMPIONS"),
#     (c4, f"{(df['KMeans_Segment']=='At Risk').sum():,}", "AT RISK"),
# ]:
#     with col:
#         st.markdown(f"<div class='seg-card'><div class='seg-value'>{val}</div><div class='seg-label'>{label}</div></div>", unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)
# tab1, tab2, tab3 = st.tabs(["📊 SEGMENT OVERVIEW", "🗺️ PCA VISUALISATION", "📋 RFM DEEP DIVE"])

# # Calculate processing summaries
# seg_summary = df.groupby("KMeans_Segment").agg(
#     Count=("Customer_ID", "count"), AvgRecency=("Recency", "mean"),
#     AvgFrequency=("Frequency", "mean"), AvgMonetary=("Monetary", "mean"),
#     TotalRevenue=("Monetary", "sum")).round(1).reset_index()
# seg_summary["Revenue%"]=(seg_summary["TotalRevenue"]/seg_summary["TotalRevenue"].sum()*100).round(1)

# # ── TAB 1: OVERVIEW METRICS ───────────────────────────────────────────────────
# with tab1:
#     c1, c2 = st.columns(2)
#     fig1=go.Figure()
#     for i, row in seg_summary.iterrows():
#         fig1.add_trace(go.Bar(name=row["KMeans_Segment"], x=[row["KMeans_Segment"]],
#             y=[row["Count"]], marker_color=COLORS[i%6],
#             hovertemplate=f"<b>{row['KMeans_Segment']}</b><br>Customers: {row['Count']:,}<extra></extra>"))
#     fig1.update_layout(title="👥 Customers per Segment", template=TEMPLATE,
#         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#         height=350, showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
#         xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e3a5f"))
#     c1.plotly_chart(fig1, use_container_width=True)

#     fig2=go.Figure(go.Pie(labels=seg_summary["KMeans_Segment"], values=seg_summary["TotalRevenue"],
#         marker_colors=COLORS, hole=0.45,
#         hovertemplate="<b>%{label}</b><br>Revenue: £%{value:,.0f}<br>Share: %{percent}<extra></extra>"))
#     fig2.update_layout(title="💰 Revenue Share by Segment", template=TEMPLATE,
#         paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=10, r=10, t=50, b=10),
#         legend=dict(font=dict(color="#94a3b8")))
#     c2.plotly_chart(fig2, use_container_width=True)

#     # Radar chart featuring clean engine color updates
#     cats=["Recency (inv)", "Frequency", "Monetary"]
#     fig3=go.Figure()
#     for i, row in seg_summary.iterrows():
#         r_inv=1/(row["AvgRecency"]+1)*365
#         vals=[r_inv/max(1/(seg_summary["AvgRecency"]+1)*365)*100,
#              row["AvgFrequency"]/seg_summary["AvgFrequency"].max()*100,
#              row["AvgMonetary"]/seg_summary["AvgMonetary"].max()*100]
#         fig3.add_trace(go.Scatterpolar(
#             r=vals+[vals[0]], theta=cats+[cats[0]],
#             name=row["KMeans_Segment"], 
#             line_color=COLORS[i%6], 
#             fill="toself",
#             fillcolor=hex_to_rgba(COLORS[i%6], 0.15)
#         ))
#     fig3.update_layout(title="🎯 Segment Radar Chart (RFM Profile)", template=TEMPLATE,
#         paper_bgcolor="rgba(0,0,0,0)", height=380,
#         polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor="#1e3a5f"),
#                    angularaxis=dict(gridcolor="#1e3a5f")),
#         legend=dict(font=dict(color="#94a3b8")))
#     st.plotly_chart(fig3, use_container_width=True)
#     st.dataframe(seg_summary.style.background_gradient(cmap="Blues", subset=["TotalRevenue", "Count"]), use_container_width=True)

# # ── TAB 2: PCA REDUCTION SPATIAL GRAPH ────────────────────────────────────────
# with tab2:
#     seg_color_map={s:COLORS[i%6] for i, s in enumerate(df["KMeans_Segment"].unique())}
#     fig=px.scatter(df, x="PCA_1", y="PCA_2", color="KMeans_Segment",
#         title="🗺️ Customer Segments – PCA 2D Projection",
#         hover_data={"Customer_ID":True, "Recency":True, "Frequency":True, "Monetary":":.0f"},
#         color_discrete_map=seg_color_map, opacity=0.75, template=TEMPLATE)
#     fig.update_traces(marker=dict(size=6))
#     fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#         height=550, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
#         legend=dict(font=dict(color="#94a3b8")))
#     st.plotly_chart(fig, use_container_width=True)

# # ── TAB 3: DEEP DIVE INSPECTOR & DOWNLOAD EXPORTS ────────────────────────────
# with tab3:
#     c1, c2=st.columns(2)
#     for col, feat, color in [(c1, "Recency", "Reds"), (c2, "Monetary", "Purples")]:
#         fig=px.violin(df, x="KMeans_Segment", y=feat, color="KMeans_Segment",
#             title=f"📦 {feat} by Segment", box=True, color_discrete_sequence=COLORS, template=TEMPLATE)
#         fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             showlegend=False, height=350, xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e3a5f"))
#         col.plotly_chart(fig, use_container_width=True)
        
#     sel=st.selectbox("🔍 FILTER BY SEGMENT", ["All"]+sorted(df["KMeans_Segment"].unique().tolist()))
#     filtered=df if sel=="All" else df[df["KMeans_Segment"]==sel]
#     cols=[c for c in ["Customer_ID", "Recency", "Frequency", "Monetary", "RFM_Score", "KMeans_Segment"] if c in filtered.columns]
#     st.dataframe(filtered[cols].head(200), use_container_width=True)
#     st.download_button("⬇️ EXPORT SEGMENT DATA", filtered.to_csv(index=False), "segments.csv", "text/csv")
"""
RetailPulse – Customer Segmentation Page
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

# ── CUSTOM SCRIPT ENGINE STYLING ──────────────────────────────────────────────
STYLE = """
<style>
[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#020818,#0a1628,#020818);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#040d1f,#071428)!important;border-right:1px solid #1a3a5c!important;}
[data-testid="stSidebarNav"] a{text-transform:uppercase!important;letter-spacing:0.08em!important;font-size:0.78rem!important;font-weight:600!important;}
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.page-header{background:linear-gradient(135deg,#0f2744,#1a1a5e);border:1px solid #2d5a8e;
  border-radius:16px;padding:30px;margin-bottom:24px;animation:fadeInUp 0.6s ease;
  border-left:4px solid #8b5cf6;}
.seg-card{background:linear-gradient(135deg,#0d1f3c,#0f172a);border:1px solid #1e3a5f;
  border-radius:12px;padding:20px;text-align:center;transition:all 0.2s;animation:fadeInUp 0.7s ease;}
.seg-card:hover{transform:translateY(-3px);border-color:#8b5cf6;box-shadow:0 8px 24px rgba(139,92,246,0.2);}
.seg-value{font-size:2rem;font-weight:800;color:#a78bfa;}
.seg-label{color:#64748b;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)
st.markdown("""
<div class='page-header'>
  <div style='font-size:1.8rem;font-weight:800;color:#e2e8f0'>👥 CUSTOMER SEGMENTATION</div>
  <div style='color:#64748b;margin-top:6px'>RFM Analysis · K-Means + DBSCAN Clustering · 6 Business Segments · PCA Visualisation</div>
</div>""", unsafe_allow_html=True)

TEMPLATE = "plotly_dark"
COLORS = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]

# ── PLOTLY COLOR PARSING BUGFIX HELPER ────────────────────────────────────────
def hex_to_rgba(hex_str, alpha=0.15):
    """
    Translates hex strings down to standard, engine-readable rgba values.
    Prevents ValueError mixups inside go.Scatterpolar maps.
    """
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"

# ── DATA STREAM MANAGEMENT ───────────────────────────────────────────────────
@st.cache_data
def load_rfm():
    for path in ["data/processed/rfm_segmented.parquet", "data/processed/rfm.parquet"]:
        p = Path(path)
        if p.exists():
            df = pd.read_parquet(p)
            np.random.seed(42)
            n = len(df)
            if "KMeans_Segment" not in df.columns:
                seg_map={0:"Champions", 1:"Loyal Customers", 2:"Potential Loyalists", 3:"At Risk", 4:"Hibernating", 5:"Lost"}
                df["KMeans_Cluster"]=np.random.randint(0, 6, n)
                df["KMeans_Segment"]=df["KMeans_Cluster"].map(seg_map)
            if "PCA_1" not in df.columns:
                df["PCA_1"]=np.random.randn(n)
                df["PCA_2"]=np.random.randn(n)
            return df
            
    # Mock Data Fallback System
    np.random.seed(42); n=500
    seg=np.random.choice(["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating", "Lost"], n)
    return pd.DataFrame({
        "Customer_ID": range(1000, 1000+n), 
        "Recency": np.random.randint(1, 365, n),
        "Frequency": np.random.randint(1, 50, n), 
        "Monetary": np.random.exponential(500, n),
        "R_Score": np.random.randint(1, 6, n), 
        "F_Score": np.random.randint(1, 6, n),
        "M_Score": np.random.randint(1, 6, n), 
        "RFM_Score": np.random.randint(3, 16, n),
        "KMeans_Segment": seg, 
        "PCA_1": np.random.randn(n), 
        "PCA_2": np.random.randn(n)
    })

df = load_rfm()

# ── SEGMENT STATS CARDS ───────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, f"{len(df):,}", "TOTAL CUSTOMERS"),
    (c2, "6", "SEGMENTS"),
    (c3, f"{(df['KMeans_Segment']=='Champions').sum():,}", "CHAMPIONS"),
    (c4, f"{(df['KMeans_Segment']=='At Risk').sum():,}", "AT RISK"),
]:
    with col:
        st.markdown(f"<div class='seg-card'><div class='seg-value'>{val}</div><div class='seg-label'>{label}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 SEGMENT OVERVIEW", "🗺️ PCA VISUALISATION", "📋 RFM DEEP DIVE"])

# Calculate processing summaries
seg_summary = df.groupby("KMeans_Segment").agg(
    Count=("Customer_ID", "count"), AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"), AvgMonetary=("Monetary", "mean"),
    TotalRevenue=("Monetary", "sum")).round(1).reset_index()
seg_summary["Revenue%"]=(seg_summary["TotalRevenue"]/seg_summary["TotalRevenue"].sum()*100).round(1)

# ── TAB 1: OVERVIEW METRICS ───────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    fig1=go.Figure()
    for i, row in seg_summary.iterrows():
        fig1.add_trace(go.Bar(name=row["KMeans_Segment"], x=[row["KMeans_Segment"]],
            y=[row["Count"]], marker_color=COLORS[i%6],
            hovertemplate=f"<b>{row['KMeans_Segment']}</b><br>Customers: {row['Count']:,}<extra></extra>"))
    fig1.update_layout(title="👥 Customers per Segment", template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e3a5f"))
    c1.plotly_chart(fig1, use_container_width=True)

    fig2=go.Figure(go.Pie(labels=seg_summary["KMeans_Segment"], values=seg_summary["TotalRevenue"],
        marker_colors=COLORS, hole=0.45,
        hovertemplate="<b>%{label}</b><br>Revenue: £%{value:,.0f}<br>Share: %{percent}<extra></extra>"))
    fig2.update_layout(title="💰 Revenue Share by Segment", template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(font=dict(color="#94a3b8")))
    c2.plotly_chart(fig2, use_container_width=True)

    # Radar chart featuring clean engine color updates
    cats=["Recency (inv)", "Frequency", "Monetary"]
    fig3=go.Figure()
    for i, row in seg_summary.iterrows():
        r_inv=1/(row["AvgRecency"]+1)*365
        vals=[r_inv/max(1/(seg_summary["AvgRecency"]+1)*365)*100,
             row["AvgFrequency"]/seg_summary["AvgFrequency"].max()*100,
             row["AvgMonetary"]/seg_summary["AvgMonetary"].max()*100]
        fig3.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            name=row["KMeans_Segment"], 
            line_color=COLORS[i%6], 
            fill="toself",
            fillcolor=hex_to_rgba(COLORS[i%6], 0.15)
        ))
    fig3.update_layout(title="🎯 Segment Radar Chart (RFM Profile)", template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)", height=380,
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor="#1e3a5f"),
                   angularaxis=dict(gridcolor="#1e3a5f")),
        legend=dict(font=dict(color="#94a3b8")))
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(seg_summary.style.background_gradient(cmap="Blues", subset=["TotalRevenue", "Count"]), use_container_width=True)

# ── TAB 2: PCA REDUCTION SPATIAL GRAPH ────────────────────────────────────────
with tab2:
    seg_color_map={s:COLORS[i%6] for i, s in enumerate(df["KMeans_Segment"].unique())}
    fig=px.scatter(df, x="PCA_1", y="PCA_2", color="KMeans_Segment",
        title="🗺️ Customer Segments – PCA 2D Projection",
        hover_data={"Customer_ID":True, "Recency":True, "Frequency":True, "Monetary":":.0f"},
        color_discrete_map=seg_color_map, opacity=0.75, template=TEMPLATE)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=550, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        legend=dict(font=dict(color="#94a3b8")))
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: DEEP DIVE INSPECTOR & DOWNLOAD EXPORTS ────────────────────────────
with tab3:
    c1, c2=st.columns(2)
    for col, feat, color in [(c1, "Recency", "Reds"), (c2, "Monetary", "Purples")]:
        fig=px.violin(df, x="KMeans_Segment", y=feat, color="KMeans_Segment",
            title=f"📦 {feat} by Segment", box=True, color_discrete_sequence=COLORS, template=TEMPLATE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=350, xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e3a5f"))
        col.plotly_chart(fig, use_container_width=True)
        
    sel=st.selectbox("🔍 FILTER BY SEGMENT", ["All"]+sorted(df["KMeans_Segment"].unique().tolist()))
    filtered=df if sel=="All" else df[df["KMeans_Segment"]==sel]
    cols=[c for c in ["Customer_ID", "Recency", "Frequency", "Monetary", "RFM_Score", "KMeans_Segment"] if c in filtered.columns]
    st.dataframe(filtered[cols].head(200), use_container_width=True)
    st.download_button("⬇️ EXPORT SEGMENT DATA", filtered.to_csv(index=False), "segments.csv", "text/csv")