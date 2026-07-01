"""Unit tests for feature engineering"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import build_rfm, build_churn_labels, build_daily_sales


def make_clean_df():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Invoice":     [f"INV{i}" for i in range(n)],
        "Customer_ID": np.random.choice([1001,1002,1003,1004,1005], n),
        "InvoiceDate": np.random.choice(dates, n),
        "TotalPrice":  np.random.exponential(50, n),
        "Quantity":    np.random.randint(1,20,n),
        "Date":        [d.date() for d in np.random.choice(dates, n)],
    })


def test_build_rfm_columns():
    df = make_clean_df()
    rfm = build_rfm(df)
    assert "Recency"   in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary"  in rfm.columns
    assert "RFM_Score" in rfm.columns
    assert "Segment"   in rfm.columns


def test_build_rfm_customer_count():
    df = make_clean_df()
    rfm = build_rfm(df)
    assert len(rfm) == df["Customer_ID"].nunique()


def test_build_churn_labels():
    df = make_clean_df()
    labels = build_churn_labels(df, inactive_days=90)
    assert "churned" in labels.columns
    assert labels["churned"].isin([0,1]).all()


def test_build_daily_sales():
    df = make_clean_df()
    daily = build_daily_sales(df)
    assert "ds" in daily.columns
    assert "y"  in daily.columns
    assert (daily["y"] > 0).all()
