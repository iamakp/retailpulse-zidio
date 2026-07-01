"""Unit tests for ML models"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.inventory import (
    calculate_safety_stock, calculate_reorder_point, calculate_eoq
)
from src.monitoring.drift_detection import compute_psi, should_retrain


def test_safety_stock_positive():
    ss = calculate_safety_stock(demand_std=5.0, lead_time_days=7)
    assert ss > 0


def test_reorder_point_greater_than_demand():
    ss = calculate_safety_stock(5.0, 7)
    rop = calculate_reorder_point(avg_daily_demand=10.0, lead_time_days=7, safety_stock=ss)
    assert rop > 10.0 * 7


def test_eoq_positive():
    eoq = calculate_eoq(annual_demand=1000, order_cost=50, holding_cost_per_unit=2)
    assert eoq > 0
    assert abs(eoq - np.sqrt(2*1000*50/2)) < 0.01


def test_psi_stable_distributions():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)
    psi = compute_psi(ref, cur)
    assert psi < 0.1  # stable


def test_psi_drifted_distributions():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(3, 1, 1000)   # big shift
    psi = compute_psi(ref, cur)
    assert psi > 0.2  # should flag drift


def test_should_retrain_false():
    results = {"Recency": {"psi":0.05,"drifted":False,"severity":"LOW"}}
    assert should_retrain(results) == False


def test_should_retrain_true():
    results = {"Recency": {"psi":0.25,"drifted":True,"severity":"HIGH"}}
    assert should_retrain(results) == True
