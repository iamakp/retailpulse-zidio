"""Unit tests for ETL pipeline"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.etl import normalise_columns, clean_data


def make_sample_df():
    return pd.DataFrame({
        "Invoice":     ["536365","536366","C536367","536368"],
        "StockCode":   ["85123A","71053","84406B","84029G"],
        "Description": ["WHITE HANGING","WHITE METAL","CREAM CUPID","KNITTED UNION"],
        "Quantity":    [6, 6, -1, 8],
        "InvoiceDate": pd.to_datetime(["2010-12-01 08:26","2010-12-01 08:26",
                                       "2010-12-01 08:28","2010-12-01 08:34"]),
        "Price":       [2.55, 3.39, 4.95, 3.39],
        "Customer ID": [17850.0, 17850.0, 17850.0, 13047.0],
        "Country":     ["UK","UK","UK","UK"],
    })


def test_normalise_columns():
    df = make_sample_df()
    df = normalise_columns(df)
    assert "Customer_ID" in df.columns
    assert "Customer ID" not in df.columns


def test_clean_data_removes_cancellations():
    df = make_sample_df()
    df = normalise_columns(df)
    df = clean_data(df)
    assert not df["Invoice"].astype(str).str.startswith("C").any()


def test_clean_data_removes_negative_quantity():
    df = make_sample_df()
    df = normalise_columns(df)
    df = clean_data(df)
    assert (df["Quantity"] > 0).all()


def test_clean_data_adds_total_price():
    df = make_sample_df()
    df = normalise_columns(df)
    df = clean_data(df)
    assert "TotalPrice" in df.columns
    assert (df["TotalPrice"] > 0).all()


def test_clean_data_row_count():
    df = make_sample_df()
    df = normalise_columns(df)
    result = clean_data(df)
    # Should keep only 2 valid rows (remove cancellation + negative qty)
    assert len(result) == 2
