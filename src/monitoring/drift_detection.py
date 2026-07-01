"""
Model Monitoring & Drift Detection – RetailPulse
Uses Evidently AI to detect data and prediction drift.
Triggers retraining when PSI > 0.2.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently AI not installed. Drift detection will use basic PSI.")


# ─────────────────────────────────────────────
# PSI (Population Stability Index) – fallback
# ─────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    < 0.1: No significant change
    0.1–0.2: Moderate change
    > 0.2: Significant drift → retrain
    """
    eps = 1e-6
    expected_pct, bin_edges = np.histogram(expected, bins=bins, density=False)
    actual_pct, _ = np.histogram(actual, bins=bin_edges, density=False)

    expected_pct = expected_pct / (len(expected) + eps)
    actual_pct = actual_pct / (len(actual) + eps)

    # Avoid log(0)
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def check_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list,
    psi_threshold: float = 0.2,
) -> dict:
    """Compute PSI for each feature and flag drifted features."""
    results = {}
    for col in feature_cols:
        if col in reference_df.columns and col in current_df.columns:
            psi = compute_psi(reference_df[col].dropna().values, current_df[col].dropna().values)
            results[col] = {
                "psi": psi,
                "drifted": psi > psi_threshold,
                "severity": "HIGH" if psi > 0.25 else ("MODERATE" if psi > 0.1 else "LOW"),
            }

    drifted_features = [k for k, v in results.items() if v["drifted"]]
    logger.info(f"Drift check: {len(drifted_features)}/{len(feature_cols)} features drifted")
    if drifted_features:
        logger.warning(f"Drifted features: {drifted_features}")

    return results


def should_retrain(drift_results: dict, psi_threshold: float = 0.2) -> bool:
    """Return True if any feature exceeds the PSI threshold."""
    return any(v["psi"] > psi_threshold for v in drift_results.values())


# ─────────────────────────────────────────────
# Evidently AI Reports (if available)
# ─────────────────────────────────────────────

def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = "reports/",
    report_name: str = "drift_report.html",
) -> str | None:
    if not EVIDENTLY_AVAILABLE:
        logger.warning("Evidently AI not available. Skipping HTML report.")
        return None

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference_df, current_data=current_df)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / report_name
    report.save_html(str(out_path))
    logger.info(f"Drift report saved to {out_path}")
    return str(out_path)


# ─────────────────────────────────────────────
# Forecast Performance Monitor
# ─────────────────────────────────────────────

def monitor_forecast_accuracy(
    actuals: pd.Series,
    predictions: pd.Series,
    mape_threshold: float = 12.0,
) -> dict:
    """Check if live MAPE exceeds production threshold."""
    from sklearn.metrics import mean_absolute_percentage_error
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    status = "OK" if mape <= mape_threshold else "DEGRADED"

    result = {
        "mape": round(mape, 2),
        "threshold": mape_threshold,
        "status": status,
        "retrain_recommended": status == "DEGRADED",
    }
    logger.info(f"Forecast monitor: MAPE={mape:.2f}% | Status={status}")
    return result


def monitor_churn_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_threshold: float = 0.88,
) -> dict:
    """Check if live AUC-ROC drops below production threshold."""
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)
    status = "OK" if auc >= auc_threshold else "DEGRADED"

    result = {
        "auc_roc": round(auc, 4),
        "threshold": auc_threshold,
        "status": status,
        "retrain_recommended": status == "DEGRADED",
    }
    logger.info(f"Churn monitor: AUC={auc:.4f} | Status={status}")
    return result
