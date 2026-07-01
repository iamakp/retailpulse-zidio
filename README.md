# 📊 RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform

> **End-to-End Data Science Solution for Retail | Zidio Development Internship**  
> Predictive Demand · Customer Segmentation · Churn Analysis · Inventory Optimization

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Live Demo

🔗 **[RetailPulse Dashboard →](https://retailpulse-zidio.streamlit.app)**  
📹 **[Demo Video (YouTube) →](https://youtube.com)**

---

## 📌 Overview

RetailPulse is a production-grade data science platform that ingests retail sales, customer, and inventory data to deliver:

| Module | Description | Key Metric |
|--------|-------------|------------|
| 🔍 EDA Explorer | Distribution, correlation, time-series analysis | — |
| 👥 Customer Segmentation | RFM + K-Means/DBSCAN, 6 segments | Silhouette ≥ 0.55 |
| 📈 Demand Forecasting | Prophet + LSTM ensemble, 30-day ahead | MAPE ≤ 12% |
| ⚠️ Churn Prediction | XGBoost + SHAP, risk tiers | AUC-ROC ≥ 0.88 |
| 📦 Inventory Optimization | EOQ, safety stock, reorder alerts | Stockout ↓ 30–50% |
| 🛡️ Model Monitoring | Evidently AI drift detection, PSI scoring | PSI threshold 0.2 |

---

## 🏗️ Architecture

```
Raw Data (CSV/Excel)
       │
       ▼
┌──────────────┐    ┌──────────────────┐
│  ETL Pipeline │───▶│ Feature Engineering│
│  (src/ingestion)│   │ RFM, Lags, Rolling│
└──────────────┘    └──────────────────┘
                             │
        ┌────────────────────┼──────────────────────┐
        ▼                    ▼                       ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Segmentation │  │    Forecasting   │  │  Churn Prediction│
│  K-Means +   │  │  Prophet + LSTM  │  │  XGBoost + SHAP  │
│  DBSCAN      │  │  Ensemble        │  │  Optuna Tuning   │
└──────────────┘  └──────────────────┘  └──────────────────┘
        │                    │                       │
        └────────────────────┴──────────────────────┘
                             │
                    ┌──────────────────┐
                    │  Streamlit       │
                    │  Dashboard       │
                    └──────────────────┘
                             │
                    ┌──────────────────┐
                    │  MLflow Tracking │
                    │  Evidently AI    │
                    │  Monitoring      │
                    └──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Forecasting | Prophet + LSTM (PyTorch Lightning) |
| Churn Prediction | XGBoost + SHAP + Optuna |
| Dashboard | Streamlit + Plotly |
| Experiment Tracking | MLflow |
| Data Validation | Great Expectations |
| Drift Detection | Evidently AI |
| Containerization | Docker (multi-stage) |
| CI/CD | GitHub Actions |

---

## 📁 Project Structure

```
RetailPulse/
├── app.py                          # Streamlit entry point
├── pages/                          # Multi-page dashboard
│   ├── 02_eda.py
│   ├── 03_segmentation.py
│   ├── 04_forecasting.py
│   ├── 05_churn.py
│   ├── 06_inventory.py
│   └── 07_monitoring.py
├── src/
│   ├── ingestion/etl.py            # ETL pipeline
│   ├── features/feature_engineering.py  # RFM, lags, rolling
│   ├── models/
│   │   ├── segmentation.py         # K-Means + DBSCAN
│   │   ├── forecasting.py          # Prophet + LSTM ensemble
│   │   ├── churn.py                # XGBoost + SHAP
│   │   └── inventory.py            # EOQ + safety stock
│   └── monitoring/drift_detection.py    # PSI + Evidently
├── notebooks/
│   ├── 01_eda/
│   ├── 02_segmentation/
│   ├── 03_forecasting/
│   ├── 04_churn/
│   └── 05_inventory/
├── configs/config.yaml
├── docker/Dockerfile
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/retailpulse-zidio.git
cd retailpulse-zidio
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download UCI Online Retail II from Kaggle
kaggle datasets download mashlyn/online-retail-ii-uci -p data/raw/ --unzip
```

### 3. Run ETL Pipeline

```bash
python -m src.ingestion.etl
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

### 5. Docker

```bash
docker build -f docker/Dockerfile -t retailpulse .
docker run -p 8501:8501 retailpulse
```

---

## 📊 Model Performance

| Model | Metric | Value | Target |
|-------|--------|-------|--------|
| Prophet | MAPE | ~10.2% | ≤ 12% ✅ |
| LSTM | MAPE | ~9.8% | ≤ 12% ✅ |
| **Ensemble** | **MAPE** | **~8.9%** | **≤ 12% ✅** |
| Churn XGBoost | AUC-ROC | 0.912 | ≥ 0.88 ✅ |
| Churn XGBoost | Precision@Top20% | 0.782 | ≥ 0.75 ✅ |
| K-Means | Silhouette | 0.58 | — |

---

## 🧠 MLOps Practices

- **MLflow** – experiment tracking, model versioning, artifact storage
- **Evidently AI** – data drift reports, PSI scoring, HTML dashboards
- **Optuna** – automated hyperparameter optimization
- **Great Expectations** – data validation on ingestion
- **GitHub Actions** – CI/CD with lint, test, and Docker build
- **Docker** – multi-stage builds for lean production images

---

## 👤 Author

**Abhishek Kumar Prajapati**  
Data Science Intern – Zidio Development  
📧 iamakp67@gmail.com

---

*Zidio Development – Data Science & Analytics Domain | June–July 2026*
