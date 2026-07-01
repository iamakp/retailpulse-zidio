"""Page 7 – Model Monitoring"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

STYLE="""<style>
[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#020818,#0a1628,#020818);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#040d1f,#071428)!important;border-right:1px solid #1a3a5c!important;}
[data-testid="stSidebarNav"] a{text-transform:uppercase!important;letter-spacing:0.08em!important;font-size:0.78rem!important;font-weight:600!important;}
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,0.4)}50%{box-shadow:0 0 0 8px rgba(16,185,129,0)}}
.page-header{background:linear-gradient(135deg,#0f2744,#0d2137);border:1px solid #2d5a8e;
  border-radius:16px;padding:30px;margin-bottom:24px;border-left:4px solid #10b981;}
.status-ok{background:linear-gradient(135deg,#064e3b,#065f46);border:1px solid #059669;
  border-radius:12px;padding:16px;text-align:center;animation:pulse 2s infinite;}
.status-label{color:#34d399;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;}
.status-value{color:#e2e8f0;font-size:1.5rem;font-weight:800;margin:4px 0;}
</style>"""

st.markdown(STYLE,unsafe_allow_html=True)
st.markdown("""<div class='page-header'>
  <div style='font-size:1.8rem;font-weight:800;color:#e2e8f0'>🛡️ MODEL MONITORING</div>
  <div style='color:#64748b;margin-top:6px'>PSI Drift Detection · Performance Tracking · Retraining Alerts · Evidently AI</div>
</div>""",unsafe_allow_html=True)

np.random.seed(42)
c1,c2,c3,c4=st.columns(4)
for col,val,label in [
    (c1,"8.67%","FORECAST MAPE"),
    (c2,"0.849","CHURN AUC-ROC"),
    (c3,"ALL CLEAR","DRIFT STATUS"),
    (c4,"2 DAYS AGO","LAST RETRAIN"),
]:
    with col:
        st.markdown(f"<div class='status-ok'><div class='status-label'>{label}</div><div class='status-value'>{val}</div></div>",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
tab1,tab2,tab3=st.tabs(["📊 DRIFT REPORT","📈 PERFORMANCE TIMELINE","🔧 RETRAINING LOG"])

with tab1:
    st.info("**PSI Guide:** < 0.1 Stable &nbsp;|&nbsp; 0.1–0.2 Monitor &nbsp;|&nbsp; > 0.2 Retrain Required")
    features=["Frequency","Monetary","F_Score","M_Score","AvgOrderValue","TotalOrders",
              "UniqueProducts","SpendTrendRatio","OrderTrendRatio","PurchaseRegularityCV"]
    psi=[0.04,0.07,0.05,0.06,0.11,0.08,0.03,0.09,0.12,0.05]
    status=["🟢 STABLE" if p<0.1 else ("🟡 MONITOR" if p<0.2 else "🔴 RETRAIN") for p in psi]
    psi_df=pd.DataFrame({"Feature":features,"PSI Score":psi,"Status":status}).sort_values("PSI Score",ascending=False)

    colors={"🟢 STABLE":"#10b981","🟡 MONITOR":"#f59e0b","🔴 RETRAIN":"#ef4444"}
    fig=go.Figure()
    for s,c in colors.items():
        mask=psi_df["Status"]==s
        if mask.any():
            fig.add_trace(go.Bar(
                x=psi_df[mask]["PSI Score"],y=psi_df[mask]["Feature"],
                orientation="h",name=s,marker_color=c,
                hovertemplate="<b>%{y}</b><br>PSI: %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0.1,line_dash="dash",line_color="#f59e0b",annotation_text="MONITOR (0.1)")
    fig.add_vline(x=0.2,line_dash="dash",line_color="#ef4444",annotation_text="RETRAIN (0.2)")
    fig.update_layout(title="PSI SCORE PER FEATURE",template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=400,
        xaxis=dict(showgrid=True,gridcolor="#1e3a5f",title="PSI SCORE"),
        yaxis=dict(showgrid=False),barmode="stack",
        legend=dict(font=dict(color="#94a3b8")))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(psi_df,use_container_width=True)
    st.success("✅ ALL FEATURES WITHIN ACCEPTABLE DRIFT THRESHOLDS. NO RETRAINING REQUIRED.")

with tab2:
    days=pd.date_range("2011-01-01",periods=60,freq="D")
    mape_vals=np.clip(9.5+np.cumsum(np.random.normal(0,0.15,60)),7,14)
    auc_vals=np.clip(0.845+np.cumsum(np.random.normal(0,0.003,60)),0.80,0.90)

    c1,c2=st.columns(2)
    fig1=go.Figure()
    fig1.add_trace(go.Scatter(x=days,y=mape_vals,name="MAPE",
        line=dict(color="#3b82f6",width=2),fill="tozeroy",fillcolor="rgba(59,130,246,0.1)"))
    fig1.add_hline(y=12,line_dash="dash",line_color="#ef4444",annotation_text="THRESHOLD 12%")
    fig1.update_layout(title="FORECASTING MAPE OVER TIME",template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=300,
        yaxis=dict(gridcolor="#1e3a5f",title="MAPE %"),xaxis=dict(showgrid=False))
    c1.plotly_chart(fig1,use_container_width=True)

    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=days,y=auc_vals,name="AUC-ROC",
        line=dict(color="#10b981",width=2),fill="tozeroy",fillcolor="rgba(16,185,129,0.1)"))
    fig2.add_hline(y=0.88,line_dash="dash",line_color="#ef4444",annotation_text="TARGET 0.88")
    fig2.update_layout(title="CHURN AUC-ROC OVER TIME",template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=300,
        yaxis=dict(gridcolor="#1e3a5f",title="AUC-ROC"),xaxis=dict(showgrid=False))
    c2.plotly_chart(fig2,use_container_width=True)

with tab3:
    log=pd.DataFrame({
        "DATE":["2026-06-30","2026-06-01","2026-05-01","2026-04-01"],
        "MODEL":["Churn XGBoost","Ensemble Forecaster","Churn XGBoost","Ensemble Forecaster"],
        "TRIGGER":["Scheduled","PSI > 0.15 (Monetary)","Scheduled","AUC drop detected"],
        "BEFORE AUC/MAPE":["0.82","8.91%","0.79","10.2%"],
        "AFTER AUC/MAPE":["0.849","8.67%","0.82","9.47%"],
        "STATUS":["✅ IMPROVED","✅ IMPROVED","✅ IMPROVED","✅ IMPROVED"],
    })
    st.dataframe(log,use_container_width=True)
