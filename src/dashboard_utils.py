"""Shared paths and strict artifact readers for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = {
    "model_ready": "data/model_ready_data.csv",
    "delay_summary": "outputs/ps2/delay_lens_summary.json",
    "feature_availability": "outputs/ps2/feature_availability_report.csv",
    "eda_summaries": "outputs/eda/eda_group_summaries.csv",
    "key_findings": "outputs/eda/key_findings.json",
    "dbscan_assignments": "outputs/ps6/dbscan_hotspot_assignments.csv",
    "hotspot_summary": "outputs/ps6/hotspot_summary.csv",
    "rider_assignments": "outputs/ps7/rider_segment_assignments.csv",
    "rider_profiles": "outputs/ps7/rider_segment_profiles.csv",
    "recommendations": "outputs/ps8/recommendations.json",
    "regression_metrics": "outputs/ml/regression_metrics.json",
    "classification_metrics": "outputs/ml/classification_metrics.json",
    "optimization_summary": "outputs/model_optimization/optimization_summary.json",
    "v2_threshold": "models/experimental/classification_threshold_v2.json",
    "output_manifest": "outputs/ps6_ps7_ps8_output_manifest.json",
}

EDA_CHARTS = {
    "Delivery time distribution": "outputs/eda/delivery_time_distribution.png",
    "Delivery time by order hour": "outputs/eda/delivery_time_by_order_hour.png",
    "Delivery time by weekday": "outputs/eda/delivery_time_by_weekday.png",
    "Delivery time by order type": "outputs/eda/delivery_time_by_order_type.png",
    "Delivery time by city category": "outputs/eda/delivery_time_by_city.png",
    "Delivery time by festival status": "outputs/eda/delivery_time_by_festival.png",
    "Distance versus delivery time": "outputs/eda/delivery_time_vs_distance.png",
    "Late rate by order hour": "outputs/eda/late_rate_by_order_hour.png",
    "Late rate by weather": "outputs/eda/late_rate_by_weather.png",
    "Late rate by traffic": "outputs/eda/late_rate_by_traffic.png",
}


def project_path(relative_path: str) -> Path:
    """Resolve a saved project artifact without relying on the shell directory."""
    return PROJECT_ROOT / relative_path


def read_csv_artifact(
    relative_path: str,
    required_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    path = project_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required project output was not found: {relative_path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Project output contains no rows: {relative_path}")
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Project output {relative_path} is missing required columns: {missing_columns}"
        )
    return frame


def read_json_artifact(relative_path: str) -> dict[str, Any] | list[Any]:
    path = project_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required project output was not found: {relative_path}")
    with path.open(encoding="utf-8") as file:
        content = json.load(file)
    if not content:
        raise ValueError(f"Project output contains no data: {relative_path}")
    return content

