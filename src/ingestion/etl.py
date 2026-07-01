"""
ETL Pipeline – RetailPulse
Handles UCI Online Retail II (.xlsx or .csv).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_raw_file(raw_dir: str = "data/raw") -> Path:
    """Auto-detect the dataset file in data/raw/."""
    raw = Path(raw_dir)
    for pattern in ["*.xlsx", "*.xls", "*.csv"]:
        files = list(raw.glob(pattern))
        if files:
            logger.info(f"Found raw file: {files[0]}")
            return files[0]
    raise FileNotFoundError(
        f"\n❌ No data file found in {raw_dir}/\n"
        "👉 Download UCI Online Retail II from:\n"
        "   https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci\n"
        "   Then place the file inside  data/raw/\n"
    )


def load_raw_data(path=None) -> pd.DataFrame:
    if path is None:
        path = find_raw_file()
    path = Path(path)
    logger.info(f"Loading: {path}")
    if path.suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        df = pd.concat(sheets.values(), ignore_index=True)
        logger.info(f"Loaded {len(sheets)} sheet(s) -> {len(df):,} rows")
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
        logger.info(f"Loaded CSV -> {len(df):,} rows")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Customer ID": "Customer_ID",
        "Invoice No":  "Invoice",
        "InvoiceNo":   "Invoice",
        "Unit Price":  "Price",
        "UnitPrice":   "Price",
        "Invoice Date":"InvoiceDate",
        "Stock Code":  "StockCode",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    initial = len(df)
    df = normalise_columns(df)

    required = ["Invoice", "StockCode", "Description", "Quantity",
                "InvoiceDate", "Price", "Customer_ID", "Country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    df = df.dropna(subset=["Customer_ID", "Description"])
    df["Customer_ID"] = df["Customer_ID"].astype(int)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]  = df["Quantity"] * df["Price"]
    df["Year"]        = df["InvoiceDate"].dt.year
    df["Month"]       = df["InvoiceDate"].dt.month
    df["DayOfWeek"]   = df["InvoiceDate"].dt.dayofweek
    df["Hour"]        = df["InvoiceDate"].dt.hour
    df["Date"]        = df["InvoiceDate"].dt.date

    logger.success(f"Clean: {len(df):,} rows (removed {initial - len(df):,})")
    return df.reset_index(drop=True)


def run_data_quality_checks(df: pd.DataFrame) -> dict:
    checks = {
        "total_rows":       len(df),
        "missing_values":   df.isnull().sum().to_dict(),
        "duplicate_rows":   int(df.duplicated().sum()),
        "date_range":       {"min": str(df["InvoiceDate"].min().date()),
                             "max": str(df["InvoiceDate"].max().date())},
        "unique_customers": int(df["Customer_ID"].nunique()),
        "unique_products":  int(df["StockCode"].nunique()),
        "unique_countries": int(df["Country"].nunique()),
        "total_revenue_GBP":round(float(df["TotalPrice"].sum()), 2),
    }
    for k, v in checks.items():
        if k not in ("missing_values",):
            logger.info(f"  {k}: {v}")
    return checks


def save_processed(df: pd.DataFrame, output_dir: str = "data/processed") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / "retail_clean.parquet"
    df.to_parquet(out, index=False)
    logger.success(f"Saved -> {out}")


def run_etl(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    raw_path = cfg["data"].get("raw_path")
    if raw_path and Path(raw_path).exists():
        df = load_raw_data(raw_path)
    else:
        df = load_raw_data()
    df = clean_data(df)
    quality = run_data_quality_checks(df)
    save_processed(df, cfg["data"]["processed_path"])
    return df, quality


if __name__ == "__main__":
    df, quality = run_etl()
    print(f"\n✅ ETL complete!")
    print(f"   Rows     : {quality['total_rows']:,}")
    print(f"   Customers: {quality['unique_customers']:,}")
    print(f"   Revenue  : £{quality['total_revenue_GBP']:,.2f}")
    print(f"   Date     : {quality['date_range']['min']} -> {quality['date_range']['max']}")
