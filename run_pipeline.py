"""
Pipeline Runner – RetailPulse
Run individual steps or the full pipeline from command line.

Usage:
    python run_pipeline.py --step etl
    python run_pipeline.py --step features
    python run_pipeline.py --step segmentation
    python run_pipeline.py --step forecasting
    python run_pipeline.py --step churn
    python run_pipeline.py --step inventory
    python run_pipeline.py --step all
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# ── make sure src/ is on path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


def step_etl():
    logger.info("━━━ STEP 1: ETL Pipeline ━━━")
    from src.ingestion.etl import run_etl
    result = run_etl()
    df, quality = result[0], result[1]
    logger.success(f"ETL done → {len(df):,} rows")
    return df, quality


def step_features(df=None):
    logger.info("━━━ STEP 2: Feature Engineering ━━━")
    import pandas as pd
    from src.features.feature_engineering import (
        build_rfm, build_daily_sales, build_churn_labels,
        build_churn_features, add_rolling_features,
        add_lag_features, add_calendar_features,
    )

    if df is None:
        df = pd.read_parquet("data/processed/retail_clean.parquet")

    rfm = build_rfm(df)
    daily = build_daily_sales(df)
    daily = add_rolling_features(daily)
    daily = add_lag_features(daily)
    daily = add_calendar_features(daily)
    churn_labels = build_churn_labels(df)
    churn_features = build_churn_features(df, rfm, churn_labels)

    # Save
    rfm.to_parquet("data/processed/rfm.parquet", index=False)
    daily.to_parquet("data/processed/daily_sales.parquet", index=False)
    churn_features.to_parquet("data/processed/churn_features.parquet", index=False)
    logger.success("Features saved to data/processed/")
    return rfm, daily, churn_features


def step_segmentation(rfm=None):
    logger.info("━━━ STEP 3: Customer Segmentation ━━━")
    import pandas as pd
    from src.models.segmentation import (
        train_kmeans, train_dbscan, get_pca_coords, segment_summary
    )

    if rfm is None:
        rfm = pd.read_parquet("data/processed/rfm.parquet")

    rfm_km = train_kmeans(rfm, n_clusters=6, model_dir="models/")
    rfm_km = train_dbscan(rfm_km)
    rfm_km = get_pca_coords(rfm_km)
    summary = segment_summary(rfm_km)

    rfm_km.to_parquet("data/processed/rfm_segmented.parquet", index=False)
    logger.success(f"Segmentation done:\n{summary.to_string()}")
    return rfm_km


def step_forecasting(daily=None):
    logger.info("━━━ STEP 4: Demand Forecasting ━━━")
    import pandas as pd
    import yaml
    from src.models.forecasting import run_forecasting_pipeline

    if daily is None:
        daily = pd.read_parquet("data/processed/daily_sales.parquet")[["ds", "y"]]

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    results = run_forecasting_pipeline(daily, cfg, model_dir="models/")
    logger.success(f"Forecasting done | Ensemble MAPE: {results['ensemble_metrics']['mape']:.2f}%")
    return results


def step_churn(churn_features=None):
    logger.info("━━━ STEP 5: Churn Prediction ━━━")
    import pandas as pd
    from src.models.churn import train_churn_model, predict_churn_risk

    if churn_features is None:
        churn_features = pd.read_parquet("data/processed/churn_features.parquet")

    results = train_churn_model(churn_features, model_dir="models/", n_trials=20)
    scored = predict_churn_risk(results["model"], churn_features)
    scored.to_parquet("data/processed/churn_scored.parquet", index=False)

    logger.success(
        f"Churn done | AUC-ROC: {results['metrics']['auc_roc']:.4f} | "
        f"Precision@Top20%: {results['metrics']['precision_top20pct']:.4f}"
    )
    return results


def step_inventory():
    logger.info("━━━ STEP 6: Inventory Optimization ━━━")
    import pandas as pd
    from src.features.feature_engineering import build_product_daily_sales
    from src.models.inventory import build_inventory_recommendations, inventory_kpi_summary

    df = pd.read_parquet("data/processed/retail_clean.parquet")
    product_daily = build_product_daily_sales(df)
    product_daily["Date"] = product_daily["Date"].astype("datetime64[ns]")

    # Use simple avg forecast per product (replace with LSTM per-product in production)
    recs = build_inventory_recommendations(product_daily, forecast_by_product={})
    recs.to_parquet("data/processed/inventory_recs.parquet", index=False)

    kpis = inventory_kpi_summary(recs)
    logger.success(f"Inventory done | KPIs: {kpis}")
    return recs


def main():
    parser = argparse.ArgumentParser(description="RetailPulse Pipeline Runner")
    parser.add_argument(
        "--step",
        choices=["etl", "features", "segmentation", "forecasting", "churn", "inventory", "all"],
        required=True,
        help="Which pipeline step to run"
    )
    args = parser.parse_args()

    if args.step == "etl":
        step_etl()
    elif args.step == "features":
        step_features()
    elif args.step == "segmentation":
        step_segmentation()
    elif args.step == "forecasting":
        step_forecasting()
    elif args.step == "churn":
        step_churn()
    elif args.step == "inventory":
        step_inventory()
    elif args.step == "all":
        logger.info("🚀 Running FULL pipeline...")
        df, quality = step_etl()
        rfm, daily, churn_features = step_features(df)
        step_segmentation(rfm)
        step_forecasting(daily)
        step_churn(churn_features)
        step_inventory()
        logger.success("🎉 Full pipeline complete! Now run: streamlit run app.py")


if __name__ == "__main__":
    main()
