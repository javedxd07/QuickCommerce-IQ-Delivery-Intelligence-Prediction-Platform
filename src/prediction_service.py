"""Load trained pipelines and serve delivery-intelligence predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from src.feature_engineering import FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "models"


def load_models(model_directory: str | Path | None = None) -> dict[str, Any]:
    """Load the fitted V2 pipelines and their validation-selected threshold."""
    directory = Path(model_directory) if model_directory is not None else MODEL_DIRECTORY
    experimental_directory = directory / "experimental"
    regression_path = experimental_directory / "regression_optimized_v2.joblib"
    classification_path = experimental_directory / "classification_optimized_v2.joblib"
    threshold_path = experimental_directory / "classification_threshold_v2.json"

    missing = [str(path) for path in (regression_path, classification_path, threshold_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required trained model artifacts are missing: {missing}")
    if sklearn.__version__ != "1.9.0":
        raise RuntimeError(
            "The V2 model artifacts were trained with scikit-learn 1.9.0; "
            f"the active runtime is {sklearn.__version__}. Install the project requirements "
            "before loading these models."
        )

    with threshold_path.open(encoding="utf-8") as file:
        threshold_data = json.load(file)
    threshold = float(threshold_data["threshold"])
    if not 0 < threshold < 1:
        raise ValueError(f"Classification threshold must be between 0 and 1; received {threshold}.")

    regression = joblib.load(regression_path)
    classification = joblib.load(classification_path)
    if not callable(getattr(regression, "predict", None)):
        raise TypeError(f"Saved V2 regression artifact is not a prediction pipeline: {regression_path}")
    if not callable(getattr(classification, "predict_proba", None)):
        raise TypeError(
            f"Saved V2 classification artifact does not provide predict_proba: {classification_path}"
        )

    return {
        "regression": regression,
        "classification": classification,
        "classification_threshold": threshold,
        "model_version": "V2 experimental",
    }


def _approved_inputs(input_df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURES) - set(input_df.columns))
    if missing:
        raise KeyError(f"Missing approved prediction input columns: {missing}")
    return input_df.loc[:, FEATURES].copy()


def predict_delivery_time(
    input_df: pd.DataFrame,
    models: dict[str, Any] | None = None,
) -> pd.Series:
    """Return predicted delivery durations in minutes for each input row."""
    loaded_models = models if models is not None else load_models()
    predictions = loaded_models["regression"].predict(_approved_inputs(input_df))
    return pd.Series(predictions, index=input_df.index, name="predicted_delivery_time")


def predict_late_delivery(
    input_df: pd.DataFrame,
    models: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return estimated late-risk scores and labels using the selected threshold."""
    loaded_models = models if models is not None else load_models()
    approved = _approved_inputs(input_df)
    risk_scores = loaded_models["classification"].predict_proba(approved)[:, 1]
    return pd.DataFrame(
        {
            "estimated_late_risk_score": risk_scores,
            "predicted_is_late": (risk_scores >= loaded_models["classification_threshold"]).astype("int8"),
            "classification_threshold": loaded_models["classification_threshold"],
        },
        index=input_df.index,
    )
