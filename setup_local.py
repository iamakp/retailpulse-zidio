"""
Local Setup Script – RetailPulse
Run this ONCE to set up your environment.
Usage: python setup_local.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run(cmd, desc=""):
    print(f"\n{'='*50}")
    print(f"▶ {desc or cmd}")
    print('='*50)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {cmd}")
        sys.exit(1)
    print(f"✅ Done")

def main():
    print("\n🚀 RetailPulse – Local Setup\n")

    # 1. Create all directories
    dirs = [
        "data/raw", "data/processed", "data/external",
        "models", "reports", "mlruns",
        "notebooks/01_eda", "notebooks/02_segmentation",
        "notebooks/03_forecasting", "notebooks/04_churn",
        "notebooks/05_inventory",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")

    # 2. Install packages (core only for fast setup)
    run(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    run(
        f"{sys.executable} -m pip install "
        "pandas numpy scikit-learn plotly streamlit mlflow "
        "xgboost shap optuna prophet torch pytorch-lightning "
        "loguru pyyaml joblib tqdm openpyxl pyarrow "
        "evidently great-expectations",
        "Installing core packages (this takes 3-5 min)"
    )

    print("\n" + "="*50)
    print("✅ Setup complete!")
    print("="*50)
    print("\nNext steps:")
    print("  1. Download dataset from Kaggle → put in data/raw/")
    print("  2. Run ETL:        python run_pipeline.py --step etl")
    print("  3. Launch app:     streamlit run app.py")
    print("  4. MLflow UI:      mlflow ui")

if __name__ == "__main__":
    main()
