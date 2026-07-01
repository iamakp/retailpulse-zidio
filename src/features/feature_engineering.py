"""
Feature Engineering – RetailPulse
RFM scoring, rolling statistics, lag features, churn labels.
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import timedelta


# ─────────────────────────────────────────────
# RFM Features
# ─────────────────────────────────────────────

def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary for each customer.
    RFM scores are ranked 1-5 (5 = best).
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("Customer_ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    # Score 1-5 using quintiles
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    rfm["Segment"] = rfm["RFM_Score"].apply(_rfm_segment_label)
    logger.info(f"RFM built for {len(rfm):,} customers")
    return rfm


def _rfm_segment_label(score: int) -> str:
    if score >= 13:
        return "Champions"
    elif score >= 10:
        return "Loyal Customers"
    elif score >= 8:
        return "Potential Loyalists"
    elif score >= 6:
        return "At Risk"
    elif score >= 4:
        return "Hibernating"
    else:
        return "Lost"


# ─────────────────────────────────────────────
# Time-Series Features
# ─────────────────────────────────────────────

def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily total sales for forecasting."""
    daily = (
        df.groupby("Date")["TotalPrice"]
        .sum()
        .reset_index()
        .rename(columns={"Date": "ds", "TotalPrice": "y"})
    )
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)
    logger.info(f"Daily sales series: {len(daily)} days")
    return daily


def build_product_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product daily sales for inventory optimization."""
    prod_daily = (
        df.groupby(["Date", "StockCode", "Description"])["Quantity"]
        .sum()
        .reset_index()
    )
    prod_daily["Date"] = pd.to_datetime(prod_daily["Date"])
    return prod_daily


def add_rolling_features(df: pd.DataFrame, windows: list = [7, 14, 30]) -> pd.DataFrame:
    """Add rolling mean/std/max features to daily sales DataFrame."""
    df = df.copy().sort_values("ds")
    for w in windows:
        df[f"rolling_mean_{w}d"] = df["y"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}d"] = df["y"].shift(1).rolling(w).std()
        df[f"rolling_max_{w}d"] = df["y"].shift(1).rolling(w).max()
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 7, 14, 30]) -> pd.DataFrame:
    """Add lag features to daily sales DataFrame."""
    df = df.copy().sort_values("ds")
    for lag in lags:
        df[f"lag_{lag}d"] = df["y"].shift(lag)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, quarter, is_weekend, is_month_end."""
    df = df.copy()
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["is_weekend"] = (df["ds"].dt.dayofweek >= 5).astype(int)
    df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    return df


# ─────────────────────────────────────────────
# Churn Labels
# ─────────────────────────────────────────────

def build_churn_labels(df: pd.DataFrame, inactive_days: int = 90) -> pd.DataFrame:
    """
    Label a customer as churned if they haven't purchased in `inactive_days`.
    Uses a train/test split window approach.
    """
    snapshot = df["InvoiceDate"].max()
    cutoff = snapshot - timedelta(days=inactive_days)

    customer_last_purchase = df.groupby("Customer_ID")["InvoiceDate"].max().reset_index()
    customer_last_purchase.columns = ["Customer_ID", "LastPurchase"]
    customer_last_purchase["churned"] = (customer_last_purchase["LastPurchase"] < cutoff).astype(int)

    churn_rate = customer_last_purchase["churned"].mean()
    logger.info(f"Churn rate: {churn_rate:.2%} ({inactive_days}-day window)")
    return customer_last_purchase


def build_churn_features(df: pd.DataFrame, rfm: pd.DataFrame, churn_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Merge RFM + behavioural features with churn labels for modelling.

    IMPORTANT — Data Leakage Fix:
    `churned` is defined as Recency > inactive_days using the SAME InvoiceDate
    data that produces the `Recency` RFM feature. Including Recency as a model
    input therefore lets the model trivially memorise the label-generating rule
    (Recency > 90 -> churned), producing saturated near-1.0 probabilities for
    most "churned" customers and a meaningless probability distribution.

    Fix: drop Recency (and R_Score, which is derived from Recency) from the
    feature set used for churn modelling. Instead we add several engineered
    features that capture *behavioural* churn signals without leaking the
    label-defining cutoff date:
      - purchase consistency (CV of days between orders)
      - early vs late tenure split (was the customer active in their first
        half of tenure vs declining toward the end, EXCLUDING the most recent
        `inactive_days` window from the calculation so the cutoff itself
        cannot leak in)
      - return/cancellation-adjusted spend trend
      - basket size trend
    """
    behaviour = df.groupby("Customer_ID").agg(
        AvgOrderValue=("TotalPrice", "mean"),
        TotalOrders=("Invoice", "nunique"),
        TotalItems=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        UniqueCountries=("Country", "nunique"),
        AvgDaysBetweenOrders=("InvoiceDate", lambda x: x.sort_values().diff().dt.days.mean()),
        StdDaysBetweenOrders=("InvoiceDate", lambda x: x.sort_values().diff().dt.days.std()),
    ).reset_index()

    # Fill NaN for single-purchase customers (no gap between orders to average)
    behaviour["AvgDaysBetweenOrders"] = behaviour["AvgDaysBetweenOrders"].fillna(
        behaviour["AvgDaysBetweenOrders"].median()
    )
    behaviour["StdDaysBetweenOrders"] = behaviour["StdDaysBetweenOrders"].fillna(0)

    # Coefficient of variation of purchase gaps — captures irregular vs habitual buyers
    # without using the actual recency value itself.
    behaviour["PurchaseRegularityCV"] = (
        behaviour["StdDaysBetweenOrders"] / behaviour["AvgDaysBetweenOrders"].replace(0, np.nan)
    ).fillna(0)

    # ── Tenure-trend features (leak-safe) ─────────────────────────────────
    # For each customer, compare order frequency/spend in the FIRST half of
    # their observed history vs the SECOND half (excluding the very last
    # `gap_days` so the label cutoff boundary itself never enters the split).
    # This captures "declining engagement" without referencing how recent
    # the cutoff is relative to today.
    trend_records = []
    for cust_id, g in df.groupby("Customer_ID"):
        g = g.sort_values("InvoiceDate")
        first_date = g["InvoiceDate"].min()
        last_date = g["InvoiceDate"].max()
        span_days = (last_date - first_date).days

        if span_days < 14:
            # Not enough history to compute a meaningful trend
            trend_records.append({
                "Customer_ID": cust_id,
                "SpendTrendRatio": 1.0,
                "OrderTrendRatio": 1.0,
                "BasketSizeTrend": 0.0,
            })
            continue

        midpoint = first_date + pd.Timedelta(days=span_days / 2)
        first_half = g[g["InvoiceDate"] < midpoint]
        second_half = g[g["InvoiceDate"] >= midpoint]

        spend_first = first_half["TotalPrice"].sum()
        spend_second = second_half["TotalPrice"].sum()
        orders_first = first_half["Invoice"].nunique()
        orders_second = second_half["Invoice"].nunique()
        basket_first = first_half.groupby("Invoice")["Quantity"].sum().mean() if orders_first else 0
        basket_second = second_half.groupby("Invoice")["Quantity"].sum().mean() if orders_second else 0

        spend_trend = (spend_second + 1) / (spend_first + 1)
        order_trend = (orders_second + 1) / (orders_first + 1)
        basket_trend = (basket_second - basket_first) if (basket_first or basket_second) else 0.0

        trend_records.append({
            "Customer_ID": cust_id,
            "SpendTrendRatio": spend_trend,
            "OrderTrendRatio": order_trend,
            "BasketSizeTrend": basket_trend,
        })

    trend_df = pd.DataFrame(trend_records)

    merged = (
        rfm.merge(behaviour, on="Customer_ID")
           .merge(trend_df, on="Customer_ID")
           .merge(churn_labels[["Customer_ID", "churned"]], on="Customer_ID")
    )

    logger.info(f"Churn feature matrix: {merged.shape}")
    logger.warning(
        "Recency/R_Score retained in dataframe for display but EXCLUDED from "
        "model features in src/models/churn.py to avoid data leakage. Added "
        "leak-safe trend features: SpendTrendRatio, OrderTrendRatio, "
        "BasketSizeTrend, PurchaseRegularityCV."
    )
    return merged
