"""Leakage-conscious feature engineering shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "Order_Date",
    "Time_Orderd",
    "Type_of_order",
    "City",
    "Festival",
    "distance_km",
]

ENGINEERED_NUMERIC_FEATURES = [
    "order_year",
    "order_month",
    "order_day",
    "order_dayofweek",
    "order_dayofyear",
    "order_hour",
    "order_minute",
    "order_hour_sin",
    "order_hour_cos",
    "order_date_missing",
    "order_time_missing",
    "distance_km",
]

CATEGORICAL_FEATURES = ["Type_of_order", "City", "Festival"]
ENGINEERED_FEATURES = ENGINEERED_NUMERIC_FEATURES + CATEGORICAL_FEATURES

_MISSING_TEXT = {"", "nan", "null", "none", "na", "n/a"}


def _clean_text(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    return cleaned.mask(cleaned.str.casefold().isin(_MISSING_TEXT), pd.NA)


def engineer_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """Derive order-time calendar features from the six approved raw inputs.

    Missing timestamps remain missing in their derived values and are represented
    by explicit missingness indicators. Numeric imputation is left to the fitted
    training pipeline.
    """
    missing_columns = sorted(set(FEATURES) - set(input_df.columns))
    if missing_columns:
        raise KeyError(f"Missing approved input columns: {missing_columns}")

    source = input_df.loc[:, FEATURES].copy()
    order_date = pd.to_datetime(_clean_text(source["Order_Date"]), errors="coerce")
    raw_time = _clean_text(source["Time_Orderd"])
    time_delta = pd.to_timedelta(raw_time, errors="coerce")

    # Support common clock strings such as "7:30 PM" without filling missing times.
    fallback_time = pd.to_datetime(raw_time, format="mixed", errors="coerce")
    time_hour = time_delta.dt.components["hours"].astype("float64")
    time_minute = time_delta.dt.components["minutes"].astype("float64")
    fallback_mask = time_delta.isna() & fallback_time.notna()
    time_hour.loc[fallback_mask] = fallback_time.loc[fallback_mask].dt.hour.astype("float64")
    time_minute.loc[fallback_mask] = fallback_time.loc[fallback_mask].dt.minute.astype("float64")

    hour_angle = 2 * np.pi * (time_hour + time_minute / 60) / 24
    result = pd.DataFrame(index=source.index)
    result["order_year"] = order_date.dt.year.astype("float64")
    result["order_month"] = order_date.dt.month.astype("float64")
    result["order_day"] = order_date.dt.day.astype("float64")
    result["order_dayofweek"] = order_date.dt.dayofweek.astype("float64")
    result["order_dayofyear"] = order_date.dt.dayofyear.astype("float64")
    result["order_hour"] = time_hour
    result["order_minute"] = time_minute
    result["order_hour_sin"] = np.sin(hour_angle)
    result["order_hour_cos"] = np.cos(hour_angle)
    result["order_date_missing"] = order_date.isna().astype("int8")
    result["order_time_missing"] = raw_time.isna().astype("int8")
    result["distance_km"] = pd.to_numeric(source["distance_km"], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        cleaned_category = _clean_text(source[column]).astype(object)
        result[column] = cleaned_category.where(cleaned_category.notna(), np.nan)

    return result.loc[:, ENGINEERED_FEATURES]
