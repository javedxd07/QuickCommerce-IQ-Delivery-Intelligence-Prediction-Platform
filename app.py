"""Premium Streamlit interface for the Delivery Intelligence project."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, time
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_components import (
    insight_panel,
    kpi_grid,
    load_stylesheet,
    page_header,
    recommendation_card,
    render_brand,
    render_sidebar_footer,
    result_panel,
    section_heading,
    status_banner,
    status_pill,
)
from src.dashboard_utils import (
    ARTIFACTS,
    EDA_CHARTS,
    project_path,
    read_csv_artifact,
    read_json_artifact,
)
from src.feature_engineering import FEATURES
from src.prediction_service import (
    load_models,
    predict_delivery_time,
    predict_late_delivery,
)

LOGGER = logging.getLogger(__name__)
PLOT_FONT = "Inter, Segoe UI, sans-serif"
PLOT_LAYOUT = {
    "font": {"family": PLOT_FONT, "color": "#334155", "size": 12},
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
    "margin": {"l": 12, "r": 12, "t": 28, "b": 12},
    "hoverlabel": {
        "bgcolor": "#111827",
        "font": {"family": PLOT_FONT, "color": "#ffffff"},
        "bordercolor": "#111827",
    },
}

st.set_page_config(
    page_title="Delivery Intelligence",
    page_icon=":material/package_2:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_csv(name: str, required_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    return read_csv_artifact(ARTIFACTS.get(name, name), required_columns)


@st.cache_data(show_spinner=False)
def cached_json(name: str) -> dict[str, Any] | list[Any]:
    return read_json_artifact(ARTIFACTS.get(name, name))


@st.cache_resource(show_spinner="Loading saved prediction models...")
def cached_models() -> dict[str, Any]:
    return load_models()


def _csv(name: str, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    try:
        return cached_csv(name, required)
    except (FileNotFoundError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Required project data is unavailable ({name}): {exc}")
        return None


def _json(name: str) -> dict[str, Any] | list[Any] | None:
    try:
        return cached_json(name)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Required project report is unavailable ({name}): {exc}")
        return None


def _page_header(eyebrow: str, title: str, description: str, context: str | None = None) -> None:
    page_header(eyebrow, title, description, context)


def _plot(figure: Any, height: int = 340) -> None:
    figure.update_layout(**PLOT_LAYOUT, height=height)
    figure.update_xaxes(showgrid=False, linecolor="#e2e8f0", zeroline=False)
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#e2e8f0",
        zeroline=False,
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "responsive": True})


def _sla_notice(summary: dict[str, Any] | None = None) -> None:
    sla_minutes = 30
    disclaimer = "30 minutes is an analytical threshold, not an official customer delivery promise."
    if summary and isinstance(summary.get("sla"), dict):
        sla = summary["sla"]
        sla_minutes = int(sla.get("minutes", sla_minutes))
        disclaimer = str(sla.get("statement", disclaimer))
    status_banner(f"{sla_minutes}-minute analytical delay lens. {disclaimer}", "info")


def _format_percent(value: float) -> str:
    return f"{value:.2f}%"


def _as_bool(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.casefold().isin(
        {"true", "1", "yes", "y"}
    )


def _style_category_chart(figure: Any, y_title: str) -> Any:
    figure.update_layout(
        **PLOT_LAYOUT,
        height=330,
        coloraxis_showscale=False,
        showlegend=False,
        yaxis_title=y_title,
        xaxis_title=None,
        bargap=0.34,
    )
    figure.update_xaxes(showgrid=False, linecolor="#e2e8f0", tickangle=0)
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#e2e8f0",
        zeroline=False,
    )
    return figure


def render_executive_overview() -> None:
    _page_header(
        "Delivery Operations",
        "Executive Overview",
        "Monitor delivery performance, demand, and operational risk.",
    )
    data = _csv(
        "model_ready",
        ("delivery_time", "is_late", "delivery_gps_valid", "Delivery_person_ID", "Order_Date"),
    )
    summary = _json("delay_summary")
    if data is None or not isinstance(summary, dict) or data.empty:
        return
    _sla_notice(summary)

    orders = len(data)
    late_count = int(pd.to_numeric(data["is_late"], errors="coerce").fillna(0).sum())
    on_time_count = orders - late_count
    valid_gps_count = int(_as_bool(data["delivery_gps_valid"]).sum())
    average_delivery = float(pd.to_numeric(data["delivery_time"], errors="coerce").mean())
    rider_count = int(data["Delivery_person_ID"].nunique())
    gps_coverage = valid_gps_count / orders * 100 if orders else 0
    kpi_grid(
        [
            {"label": "Total orders", "value": f"{orders:,}", "context": "Observed delivery records"},
            {"label": "Average delivery time", "value": f"{average_delivery:.2f} min", "context": "Across valid delivery targets"},
            {"label": "On-time rate", "value": _format_percent(on_time_count / orders * 100), "context": f"{on_time_count:,} orders"},
            {"label": "Late rate", "value": _format_percent(late_count / orders * 100), "context": f"{late_count:,} orders"},
            {"label": "Valid GPS coverage", "value": _format_percent(gps_coverage), "context": f"{valid_gps_count:,} customer locations"},
            {"label": "Active rider IDs", "value": f"{rider_count:,}", "context": "Unique observed rider identifiers"},
        ]
    )

    section_heading("Delivery performance", "On-time and late shares use the 30-minute analytical threshold.")
    left, right = st.columns(2, gap="large")
    outcome_data = pd.DataFrame(
        {"Outcome": ["On time", "Late"], "Orders": [on_time_count, late_count]}
    )
    with left:
        figure = px.pie(
            outcome_data,
            names="Outcome",
            values="Orders",
            hole=0.72,
            color="Outcome",
            color_discrete_map={"On time": "#0d9488", "Late": "#dc2626"},
            custom_data=["Orders"],
        )
        figure.update_traces(
            textposition="inside",
            texttemplate="%{percent:.1%}",
            hovertemplate="%{label}<br>%{value:,} orders<br>%{percent:.2%}<extra></extra>",
            marker=dict(line=dict(color="#ffffff", width=3)),
        )
        figure.update_layout(
            **PLOT_LAYOUT,
            height=330,
            showlegend=True,
            legend=dict(orientation="h", y=-0.08, x=0),
        )
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "responsive": True})
    with right:
        order_dates = pd.to_datetime(data["Order_Date"], errors="coerce").dropna()
        volume = order_dates.dt.date.value_counts().sort_index()
        trend_data = pd.DataFrame({"Order date": list(volume.index), "Orders": volume.values})
        figure = px.area(
            trend_data,
            x="Order date",
            y="Orders",
            color_discrete_sequence=["#2563eb"],
        )
        figure.update_traces(
            line=dict(width=2, color="#2563eb"),
            fillcolor="rgba(37, 99, 235, 0.10)",
            hovertemplate="%{x|%d %b %Y}<br>%{y:,} orders<extra></extra>",
        )
        _plot(figure)

    section_heading("Validated project insights", "Saved observations from the EDA report.")
    findings = _json("key_findings")
    if isinstance(findings, list):
        for start in range(0, min(len(findings), 3), 2):
            columns = st.columns(2, gap="large")
            for offset, finding in enumerate(findings[start : start + 2]):
                if isinstance(finding, dict):
                    with columns[offset]:
                        insight_panel(
                            str(finding.get("title", "Finding")),
                            str(finding.get("observation", "")),
                        )
    else:
        st.caption("The saved EDA findings report is unavailable.")


def render_delivery_analytics() -> None:
    _page_header(
        "Analytics",
        "Delivery Analytics",
        "Explore observed delivery-time and late-rate patterns using saved Module 02 outputs.",
    )
    summary = _json("delay_summary")
    data = _csv("eda_summaries", ("analysis", "group", "mean", "late_percent", "n"))
    if not isinstance(summary, dict) or data is None:
        return
    _sla_notice(summary)
    comparisons = {
        "Order hour": "by_order_hour",
        "Weekday": "by_weekday",
        "Order type": "by_order_type",
        "City service category": "by_city",
        "Festival status": "by_festival",
        "Distance quartile": "by_distance_quartile",
    }
    choice = st.selectbox("Compare by", list(comparisons), key="analytics_group")
    selected = data.loc[data["analysis"] == comparisons[choice]].copy()
    if selected.empty:
        st.warning(f"No saved summary rows are available for {choice.lower()}.")
        return
    selected["mean"] = pd.to_numeric(selected["mean"], errors="coerce")
    selected["late_percent"] = pd.to_numeric(selected["late_percent"], errors="coerce")
    section_heading(
        f"Performance by {choice.lower()}",
        "Both charts and the table use saved descriptive summaries; sample counts are included for context.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        section_heading("Average delivery time")
        figure = px.bar(
            selected,
            x="group",
            y="mean",
            text=selected["mean"].map(lambda value: f"{value:.1f}" if pd.notna(value) else ""),
            hover_data={"n": ":,", "late_percent": ":.2f"},
            color_discrete_sequence=["#2563eb"],
            labels={"group": choice, "mean": "Minutes", "n": "Orders", "late_percent": "Late rate (%)"},
        )
        figure.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Average: %{y:.2f} min<br>Orders: %{customdata[0]:,}<extra></extra>",
        )
        _plot(_style_category_chart(figure, "Minutes"), 350)
    with right:
        section_heading("Late-delivery rate")
        figure = px.bar(
            selected,
            x="group",
            y="late_percent",
            text=selected["late_percent"].map(lambda value: f"{value:.1f}%" if pd.notna(value) else ""),
            hover_data={"n": ":,", "mean": ":.2f"},
            color_discrete_sequence=["#0d9488"],
            labels={"group": choice, "late_percent": "Late orders (%)", "n": "Orders", "mean": "Average minutes"},
        )
        figure.update_traces(textposition="outside", cliponaxis=False)
        _plot(_style_category_chart(figure, "Late orders (%)"), 350)

    table = selected[["group", "n", "mean", "late_percent"]].rename(
        columns={
            "group": choice,
            "n": "Orders",
            "mean": "Average delivery (min)",
            "late_percent": "Late orders (%)",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Average delivery (min)": st.column_config.NumberColumn(format="%.2f"),
            "Late orders (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Orders": st.column_config.NumberColumn(format="%d"),
        },
    )

    section_heading("Saved EDA chart gallery", "Original charts generated by the validated EDA notebook.")
    chart_labels = list(EDA_CHARTS)
    selected_charts = st.multiselect(
        "Select charts",
        chart_labels,
        default=[
            "Delivery time distribution",
            "Delivery time by order hour",
            "Delivery time by order type",
            "Delivery time by city category",
            "Delivery time by festival status",
            "Distance versus delivery time",
        ],
        key="eda_gallery",
    )
    for start in range(0, len(selected_charts), 2):
        columns = st.columns(2, gap="large")
        for offset, label in enumerate(selected_charts[start : start + 2]):
            path = project_path(EDA_CHARTS[label])
            with columns[offset]:
                if path.is_file():
                    st.image(str(path), caption=label, width="stretch")
                else:
                    st.warning(f"Saved chart is unavailable: {EDA_CHARTS[label]}")
    st.caption("These are descriptive associations from restaurant food delivery; they do not establish causation.")


@st.cache_data(show_spinner=False)
def _demo_scenarios(data: pd.DataFrame) -> dict[str, pd.Series]:
    candidates = data.loc[:, FEATURES].copy()
    candidates["distance_km"] = pd.to_numeric(candidates["distance_km"], errors="coerce")
    parsed_dates = pd.to_datetime(candidates["Order_Date"], errors="coerce")
    parsed_times = pd.to_datetime(
        candidates["Time_Orderd"], format="mixed", errors="coerce"
    )
    valid = candidates.notna().all(axis=1) & parsed_dates.notna() & parsed_times.notna()
    valid &= candidates["distance_km"].gt(0)
    candidates = candidates.loc[valid].copy()
    if candidates.empty:
        raise ValueError("No complete observed rows are available for prediction scenarios.")
    candidates["_order_hour"] = parsed_times.loc[candidates.index].dt.hour

    used_indices: set[Any] = set()

    def choose_nearest(pool: pd.DataFrame, distance: float) -> pd.Series:
        available = pool.loc[~pool.index.isin(used_indices)]
        if available.empty:
            available = pool
        selected = available.loc[
            (available["distance_km"] - distance).abs().idxmin()
        ]
        used_indices.add(selected.name)
        return selected.loc[list(FEATURES)]

    distance_median = float(candidates["distance_km"].median())
    hour_median = float(candidates["_order_hour"].median())
    distance_scale = max(float(candidates["distance_km"].std()), 1.0)
    hour_scale = max(float(candidates["_order_hour"].std()), 1.0)
    typical_scores = (
        ((candidates["distance_km"] - distance_median) / distance_scale) ** 2
        + ((candidates["_order_hour"] - hour_median) / hour_scale) ** 2
    )
    typical = candidates.loc[typical_scores.idxmin()]
    used_indices.add(typical.name)

    busy_hour = int(candidates["_order_hour"].value_counts().idxmax())
    busy_pool = candidates.loc[candidates["_order_hour"] == busy_hour]
    busy = choose_nearest(busy_pool, distance_median)

    long_distance_target = float(candidates["distance_km"].quantile(0.9))
    long_pool = candidates.loc[
        candidates["distance_km"] >= candidates["distance_km"].quantile(0.75)
    ]
    longer_distance = choose_nearest(long_pool, long_distance_target)
    return {
        "Typical order": typical.loc[list(FEATURES)],
        "Busy-hour order": busy,
        "Longer-distance order": longer_distance,
    }


def _prediction_form(data: pd.DataFrame, mode: str) -> None:
    try:
        scenarios = _demo_scenarios(data)
    except ValueError as exc:
        st.error(f"Prediction scenarios are unavailable: {exc}")
        return
    scenario_name = st.selectbox(
        "Demo scenario",
        list(scenarios),
        key="prediction_demo_scenario",
        help="Each scenario uses a complete, observed row from the model-ready dataset.",
    )
    sample = scenarios[scenario_name]
    st.caption(
        "Scenario inputs are copied from one real model-ready row and shared across both prediction pages. "
        "Adjust the fields to explore another input combination."
    )

    valid_dates = pd.to_datetime(data["Order_Date"], errors="coerce").dropna()
    date_value = pd.Timestamp(sample["Order_Date"]).date()
    order_types = sorted(data["Type_of_order"].dropna().astype(str).unique())
    city_categories = sorted(data["City"].dropna().astype(str).unique())
    festivals = sorted(data["Festival"].dropna().astype(str).unique())
    if not order_types or not city_categories or not festivals:
        st.error("Observed category values needed for prediction are unavailable.")
        return

    def selected_index(values: list[str], column: str) -> int:
        value = str(sample[column])
        return values.index(value) if value in values else 0

    scenario_key = scenario_name.lower().replace(" ", "_").replace("-", "_")
    with st.form(f"{mode}_prediction_form_{scenario_key}", border=False):
        section_heading("Order context", "Use values available at order placement.")
        left, right = st.columns(2, gap="large")
        with left:
            order_date = st.date_input(
                "Order date", value=date_value, key=f"{mode}_{scenario_key}_date"
            )
            selected_type = st.selectbox(
                "Order type",
                order_types,
                index=selected_index(order_types, "Type_of_order"),
                key=f"{mode}_{scenario_key}_type",
            )
            selected_city = st.selectbox(
                "City service category",
                city_categories,
                index=selected_index(city_categories, "City"),
                key=f"{mode}_{scenario_key}_city",
            )
        with right:
            parsed_time = pd.to_datetime(
                str(sample["Time_Orderd"]), format="mixed", errors="coerce"
            )
            if pd.isna(parsed_time):
                st.error("The selected observed scenario has an invalid order time.")
                return
            sample_time = parsed_time.time()
            order_time = st.time_input(
                "Order time",
                value=sample_time,
                key=f"{mode}_{scenario_key}_time",
            )
            selected_festival = st.selectbox(
                "Festival status",
                festivals,
                index=selected_index(festivals, "Festival"),
                key=f"{mode}_{scenario_key}_festival",
            )
            observed_distance = pd.to_numeric(data["distance_km"], errors="coerce").dropna()
            default_distance = float(sample["distance_km"])
            distance = st.number_input(
                "Delivery distance (km)",
                min_value=0.1,
                max_value=100.0,
                value=min(100.0, max(0.1, default_distance)),
                step=0.1,
                key=f"{mode}_{scenario_key}_distance",
            )
        submitted = st.form_submit_button(
            "Estimate delivery time" if mode == "time" else "Estimate late-delivery risk",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return
    if not isinstance(order_date, date) or not isinstance(order_time, time):
        st.error("Enter a valid order date and time.")
        return
    if not np.isfinite(distance) or distance <= 0:
        st.error("Delivery distance must be a positive finite number.")
        return
    observed_distance = pd.to_numeric(data["distance_km"], errors="coerce").dropna()
    if not observed_distance.empty and not (
        observed_distance.min() <= distance <= observed_distance.max()
    ):
        status_banner(
            f"Distance is outside the observed range of {observed_distance.min():.2f}–"
            f"{observed_distance.max():.2f} km. Treat the estimate as an extrapolation.",
            "warning",
        )
    if not valid_dates.empty and not (valid_dates.min().date() <= order_date <= valid_dates.max().date()):
        status_banner(
            f"Date is outside the observed project period ({valid_dates.min().date()} to "
            f"{valid_dates.max().date()}). Treat the estimate as an extrapolation.",
            "warning",
        )
    row = pd.DataFrame(
        [
            {
                "Order_Date": order_date.isoformat(),
                "Time_Orderd": order_time.strftime("%I:%M %p"),
                "Type_of_order": selected_type,
                "City": selected_city,
                "Festival": selected_festival,
                "distance_km": float(distance),
            }
        ],
        columns=FEATURES,
    )
    _run_prediction(mode, row)


def _run_prediction(mode: str, input_row: pd.DataFrame) -> None:
    try:
        models = cached_models()
        if mode == "time":
            prediction = float(predict_delivery_time(input_row, models=models).iloc[0])
            if not math.isfinite(prediction) or prediction <= 0:
                raise ValueError(f"The V2 model returned an invalid delivery-time estimate: {prediction}.")
            result_panel(
                "Estimated delivery duration",
                f"{prediction:.1f} min",
                "V2 HistGradientBoosting estimate from the entered order details. "
                "Actual delivery time may differ.",
            )
            return

        result = predict_late_delivery(input_row, models=models).iloc[0]
        score = float(result["estimated_late_risk_score"])
        threshold = float(result["classification_threshold"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"The V2 model returned an invalid late-risk score: {score}.")
        flagged = bool(result["predicted_is_late"])
        label = "Late" if flagged else "On time"
        status = status_pill(label, "medium" if flagged else "neutral")
        result_panel(
            "Estimated late-delivery risk score · not a calibrated probability",
            f"{score:.3f}",
            f"V2 threshold: {threshold:.2f}. Scores at or above the threshold are flagged late; "
            "lower scores are flagged on time. This is a decision-support signal, not a calibrated probability.",
            status=status,
            progress=score,
            threshold=threshold,
        )
        st.caption(
            "False positive: flagged late, but actually on time. False negative: flagged on time, "
            "but actually late. The decision threshold was selected on validation data only."
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"Prediction could not be completed: {exc}")
    except Exception as exc:
        LOGGER.exception("Prediction failed unexpectedly.")
        st.error(
            "Prediction failed unexpectedly. Check the saved model artifacts and input values, "
            f"then retry. Details: {type(exc).__name__}: {exc}"
        )


def render_delivery_time_prediction() -> None:
    _page_header(
        "Prediction",
        "Delivery Time Prediction",
        "Estimate delivery duration using the saved regression pipeline.",
    )
    status_banner(
        "V2 model · only the six approved order-placement features are used. "
        "distance_km availability at prediction time assumes selected-store and customer coordinates "
        "are already known; this workflow assumption is not verified. Predictions are estimates, not guarantees.",
        "info",
    )
    data = _csv(
        "model_ready",
        ("Order_Date", "Time_Orderd", "Type_of_order", "City", "Festival", "distance_km"),
    )
    if data is not None:
        _prediction_form(data, "time")


def render_late_delivery_risk() -> None:
    _page_header(
        "Prediction",
        "Late Delivery Risk",
        "Estimate late-delivery risk with the V2 classifier and validation-selected threshold.",
    )
    threshold_data = _json("v2_threshold")
    if not isinstance(threshold_data, dict):
        return
    try:
        active_threshold = float(threshold_data["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"The saved V2 classification threshold is invalid: {exc}")
        return
    if not math.isfinite(active_threshold) or not 0 < active_threshold < 1:
        st.error(f"The saved V2 classification threshold is outside (0, 1): {active_threshold}.")
        return
    status_banner(
        f"Active V2 threshold {active_threshold:.2f} is loaded from the saved threshold JSON. "
        "Scores are not guaranteed calibrated probabilities; false positives and false negatives are both possible. "
        "distance_km assumes selected-store "
        "and customer coordinates are available before prediction; this operational timing is unverified.",
        "warning",
    )
    data = _csv(
        "model_ready",
        ("Order_Date", "Time_Orderd", "Type_of_order", "City", "Festival", "distance_km"),
    )
    if data is not None:
        _prediction_form(data, "risk")


def render_geographic_intelligence() -> None:
    _page_header(
        "Analytics",
        "Geographic Intelligence",
        "Explore DBSCAN density concentrations built from valid customer delivery coordinates.",
        "PS6 · Customer GPS represents delivery destinations in restaurant delivery data.",
    )
    assignments = _csv(
        "dbscan_assignments",
        (
            "dbscan_cluster_id",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
            "delivery_time",
            "is_late",
        ),
    )
    hotspots = _csv(
        "hotspot_summary",
        (
            "dbscan_cluster_id",
            "number_of_orders",
            "late_rate_pct",
            "centroid_latitude",
            "centroid_longitude",
            "demand_share_pct",
        ),
    )
    if assignments is None or hotspots is None:
        return
    labels = pd.to_numeric(assignments["dbscan_cluster_id"], errors="coerce")
    if labels.isna().any():
        st.error("The DBSCAN assignment file contains malformed cluster labels.")
        return

    invalid_count = int((labels == -2).sum())
    noise_count = int((labels == -1).sum())
    assigned_count = int((labels >= 0).sum())
    valid_count = noise_count + assigned_count
    kpi_grid(
        [
            {"label": "Valid GPS orders", "value": f"{valid_count:,}", "context": "Invalid/unavailable coordinates excluded"},
            {"label": "DBSCAN hotspots", "value": f"{len(hotspots):,}", "context": "Density-connected components"},
            {"label": "Noise observations", "value": f"{noise_count:,}", "context": "Valid coordinates outside a hotspot"},
            {"label": "Orders in hotspots", "value": f"{assigned_count:,}", "context": f"{invalid_count:,} invalid/unavailable GPS rows"},
        ]
    )
    status_banner(
        "These coordinate-defined customer-demand concentrations are not confirmed dark-store locations. "
        "Restaurant coordinates are store proxies only.",
        "info",
    )
    status_banner(
        "DBSCAN limits neighbor distance, not full component diameter. Density chaining can produce broad regions; "
        "review centroid radii and verify real localities before operational decisions.",
        "warning",
    )

    section_heading(
        "Presentation highlights",
        "Featured hotspot statistics are read directly from the saved DBSCAN summary.",
    )
    hotspot_ids_numeric = pd.to_numeric(hotspots["dbscan_cluster_id"], errors="coerce")
    featured = hotspots.loc[hotspot_ids_numeric.isin([52, 18])].copy()
    featured["dbscan_cluster_id"] = pd.to_numeric(
        featured["dbscan_cluster_id"], errors="coerce"
    ).astype(int)
    featured = featured.set_index("dbscan_cluster_id")
    if {52, 18}.issubset(featured.index):
        kpi_grid(
            [
                {
                    "label": "Hotspot 52 · orders",
                    "value": f"{int(featured.loc[52, 'number_of_orders']):,}",
                    "context": "Highest supported late-rate hotspot",
                },
                {
                    "label": "Hotspot 52 · late rate",
                    "value": f"{float(featured.loc[52, 'late_rate_pct']):.2f}%",
                    "context": "Observed DBSCAN assignments",
                },
                {
                    "label": "Hotspot 18 · orders",
                    "value": f"{int(featured.loc[18, 'number_of_orders']):,}",
                    "context": "Largest valid-GPS concentration",
                },
                {
                    "label": "Hotspot 18 · demand share",
                    "value": f"{float(featured.loc[18, 'demand_share_pct']):.2f}%",
                    "context": "Share of valid-GPS demand",
                },
            ]
        )
    else:
        st.warning("One or both featured hotspots are missing from the saved hotspot summary.")

    section_heading("Explore hotspot concentrations", "Filter by actual saved cluster IDs and sort by an observed metric.")
    first, second, third = st.columns([1.5, 1, 1], gap="medium")
    hotspot_ids = sorted(
        pd.to_numeric(hotspots["dbscan_cluster_id"], errors="coerce").dropna().astype(int)
    )
    with first:
        selected_ids = st.multiselect("Hotspot IDs", hotspot_ids, key="hotspot_filter")
    with second:
        metric_name = st.selectbox(
            "Rank by",
            ["Order volume", "Late-delivery rate", "Demand share"],
            key="hotspot_metric",
        )
    with third:
        minimum_orders = st.number_input(
            "Minimum orders",
            min_value=1,
            max_value=int(hotspots["number_of_orders"].max()),
            value=1,
            step=1,
        )
    metric_column = {
        "Order volume": "number_of_orders",
        "Late-delivery rate": "late_rate_pct",
        "Demand share": "demand_share_pct",
    }[metric_name]
    candidates = hotspots.loc[hotspots["number_of_orders"] >= minimum_orders].copy()
    if selected_ids:
        candidates = candidates.loc[candidates["dbscan_cluster_id"].isin(selected_ids)]
    if candidates.empty:
        st.warning("No saved hotspots match the selected filters.")
        return
    candidates = candidates.sort_values(metric_column, ascending=False)
    map_limit = st.select_slider(
        "Map detail",
        options=[10, 20, 40, 80, 132],
        value=40,
        format_func=lambda value: f"Top {value}",
    )
    map_data = candidates.head(min(map_limit, len(candidates))).copy()
    section_heading(
        "Customer demand map",
        "Marker size reflects order count; hover for the saved hotspot metrics and geographic extent.",
    )
    figure = px.scatter_map(
        map_data,
        lat="centroid_latitude",
        lon="centroid_longitude",
        size="number_of_orders",
        color=metric_column,
        hover_name="dbscan_cluster_id",
        hover_data={
            "number_of_orders": True,
            "late_rate_pct": ":.2f",
            "demand_share_pct": ":.2f",
            "radius_p95_from_centroid_km": ":.2f",
            "centroid_latitude": ":.4f",
            "centroid_longitude": ":.4f",
        },
        color_continuous_scale=["#bfdbfe", "#2563eb", "#0f172a"],
        map_style="open-street-map",
        zoom=3,
        height=610,
        labels={
            "dbscan_cluster_id": "Hotspot ID",
            "number_of_orders": "Orders",
            "late_rate_pct": "Late rate (%)",
            "demand_share_pct": "Demand share (%)",
            "radius_p95_from_centroid_km": "95th percentile radius (km)",
            metric_column: metric_name,
        },
    )
    figure.update_layout(**PLOT_LAYOUT, height=610, coloraxis_colorbar_title=metric_name)
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )

    volume_columns, late_columns = st.columns(2, gap="large")
    with volume_columns:
        section_heading("Highest-volume hotspots")
        volume_table = hotspots.nlargest(10, "number_of_orders")[
            ["dbscan_cluster_id", "number_of_orders", "demand_share_pct", "late_rate_pct"]
        ].rename(
            columns={
                "dbscan_cluster_id": "Hotspot",
                "number_of_orders": "Orders",
                "demand_share_pct": "Demand share (%)",
                "late_rate_pct": "Late rate (%)",
            }
        )
        st.dataframe(volume_table, hide_index=True, width="stretch")
    with late_columns:
        section_heading("Highest late-rate hotspots")
        late_table = hotspots.nlargest(10, "late_rate_pct")[
            ["dbscan_cluster_id", "number_of_orders", "late_rate_pct"]
        ].rename(
            columns={
                "dbscan_cluster_id": "Hotspot",
                "number_of_orders": "Orders",
                "late_rate_pct": "Late rate (%)",
            }
        )
        st.dataframe(late_table, hide_index=True, width="stretch")
    section_heading("Filtered hotspot details")
    detail_columns = [
        "dbscan_cluster_id",
        "number_of_orders",
        "demand_share_pct",
        "average_delivery_time_minutes",
        "late_rate_pct",
        "centroid_latitude",
        "centroid_longitude",
        "radius_p95_from_centroid_km",
    ]
    details = candidates[detail_columns].rename(
        columns={
            "dbscan_cluster_id": "Hotspot",
            "number_of_orders": "Orders",
            "demand_share_pct": "Demand share (%)",
            "average_delivery_time_minutes": "Average delivery (min)",
            "late_rate_pct": "Late rate (%)",
            "centroid_latitude": "Centroid latitude",
            "centroid_longitude": "Centroid longitude",
            "radius_p95_from_centroid_km": "P95 radius (km)",
        }
    )
    st.dataframe(details, hide_index=True, width="stretch")
    st.caption(
        "No actual city/locality identifiers are available. Confirm local supply, travel networks, and current "
        "quick-commerce demand before site or coverage decisions."
    )


def render_rider_intelligence() -> None:
    _page_header(
        "Analytics",
        "Rider Intelligence",
        "Review descriptive rider segments from the saved PS7 assignments and profiles.",
    )
    profiles = _csv(
        "rider_profiles",
        (
            "rider_segment_id",
            "rider_count",
            "delivery_count",
            "average_rating",
            "average_delivery_time_minutes",
            "late_rate_pct",
            "deliveries_per_active_day",
            "segment_name",
            "segment_label",
        ),
    )
    metadata = _json("outputs/ps7/rider_model_metadata.json")
    if profiles is None or not isinstance(metadata, dict):
        return
    silhouette = metadata.get("selected_silhouette")
    silhouette_text = (
        f"Silhouette score {float(silhouette):.3f} indicates weak segment separation. "
        if isinstance(silhouette, (float, int))
        else "The saved analysis notes weak segment separation. "
    )
    status_banner(
        silhouette_text
        + "These profiles are exploratory summaries, not definitive performance categories. "
        "Rider tenure is unavailable and age is not used as a substitute.",
        "warning",
    )
    kpi_grid(
        [
            {"label": "Rider IDs", "value": f"{int(profiles['rider_count'].sum()):,}", "context": "Observed identifiers"},
            {"label": "Segments", "value": f"{len(profiles):,}", "context": "Exploratory K-Means groups"},
            {"label": "Deliveries profiled", "value": f"{int(profiles['delivery_count'].sum()):,}", "context": "Underlying observed records"},
        ]
    )
    segment_zero = profiles.loc[
        pd.to_numeric(profiles["rider_segment_id"], errors="coerce") == 0
    ]
    section_heading(
        "Segment 0 · supportive review",
        "Highlighted for presentation; segment membership is exploratory, not a personnel judgment.",
    )
    if not segment_zero.empty:
        segment = segment_zero.iloc[0]
        kpi_grid(
            [
                {"label": "Riders", "value": f"{int(segment['rider_count']):,}", "context": "Exploratory segment 0"},
                {"label": "Deliveries", "value": f"{int(segment['delivery_count']):,}", "context": "Observed records"},
                {"label": "Late rate", "value": f"{float(segment['late_rate_pct']):.2f}%", "context": "Segment average"},
            ]
        )
    else:
        st.warning("Rider segment 0 is missing from the saved profile output.")
    section_heading("Segment performance", "Each point is an observed segment profile; bubble size reflects delivery volume.")
    figure = px.scatter(
        profiles,
        x="average_delivery_time_minutes",
        y="late_rate_pct",
        size="delivery_count",
        color="segment_label",
        hover_name="segment_name",
        hover_data={
            "rider_count": True,
            "delivery_count": True,
            "average_rating": ":.2f",
            "deliveries_per_active_day": ":.2f",
        },
        color_discrete_sequence=["#2563eb", "#0d9488", "#d97706", "#7c3aed"],
        labels={
            "average_delivery_time_minutes": "Average delivery time (min)",
            "late_rate_pct": "Late rate (%)",
            "delivery_count": "Deliveries",
            "segment_label": "Segment",
        },
    )
    figure.update_traces(marker=dict(line=dict(color="#ffffff", width=2), opacity=0.88))
    _plot(figure, 410)

    table = profiles[
        [
            "segment_label",
            "segment_name",
            "rider_count",
            "delivery_count",
            "average_rating",
            "average_delivery_time_minutes",
            "late_rate_pct",
            "deliveries_per_active_day",
        ]
    ].rename(
        columns={
            "segment_label": "Segment",
            "segment_name": "Observed profile",
            "rider_count": "Riders",
            "delivery_count": "Deliveries",
            "average_rating": "Average rating",
            "average_delivery_time_minutes": "Average delivery (min)",
            "late_rate_pct": "Late rate (%)",
            "deliveries_per_active_day": "Deliveries / active day",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Average rating": st.column_config.NumberColumn(format="%.2f"),
            "Average delivery (min)": st.column_config.NumberColumn(format="%.2f"),
            "Late rate (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Deliveries / active day": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    assignments = _csv("rider_assignments", ("Delivery_person_ID", "rider_segment_id"))
    if assignments is not None:
        st.caption(
            f"Saved assignments cover {assignments['Delivery_person_ID'].nunique():,} rider IDs. "
            "No individual leaderboard or tenure inference is presented."
        )


def render_management_recommendations() -> None:
    _page_header(
        "Decisions",
        "Management Recommendations",
        "Three prioritized actions grounded in the saved PS6 and PS7 findings.",
    )
    recommendations = _json("recommendations")
    if not isinstance(recommendations, dict):
        return
    items = recommendations.get("recommendations")
    if (
        not isinstance(items, list)
        or len(items) != 3
        or recommendations.get("exact_recommendation_count") != 3
    ):
        st.error("The saved PS8 report must contain exactly three validated recommendations.")
        return
    status_banner(
        str(recommendations.get("source_data_limitation", "Recommendations are limited by the source data.")),
        "info",
    )
    for recommendation in items:
        recommendation_card(recommendation)


def render_about() -> None:
    _page_header(
        "Project",
        "About the Project",
        "Quick-Commerce Delivery Intelligence · Manthan Hackathon · KLU + Path-3.",
    )
    status_banner(
        "This project uses a public food-delivery dataset as a proxy for quick-commerce delivery intelligence. "
        "Findings should be interpreted within the dataset's limitations.",
        "warning",
    )
    overview, methodology = st.columns([1.2, 1], gap="large")
    with overview:
        section_heading("Project overview")
        st.write(
            "The platform combines data quality review, delay analysis, delivery prediction, late-risk "
            "classification, geographic demand clustering, rider segmentation, and management recommendations."
        )
        section_heading("Problem statement")
        st.write(
            "Use historical delivery records to understand delay patterns, estimate delivery duration and "
            "late-order risk from order-placement information, identify customer-demand concentrations, and "
            "support evidence-based operational review."
        )
        section_heading("Dataset")
        st.write(
            "The available records describe restaurant food delivery and are a proxy—not direct evidence—"
            "for quick-commerce grocery operations. The 30-minute late definition is analytical. City is a "
            "service category rather than a locality name, and rider tenure is not recorded."
        )
        section_heading("Problem statements")
        statements = [
            ("PS1", "Data Quality & Reliability"),
            ("PS2", "Define the Delay Lens"),
            ("PS3", "Diagnose Delay Patterns"),
            ("PS4", "Delivery-Time Prediction"),
            ("PS5", "Late-Delivery Classification"),
            ("PS6", "Geographic Demand Hotspots"),
            ("PS7", "Rider Segmentation"),
            ("PS8", "Management Recommendations"),
        ]
        st.dataframe(
            pd.DataFrame(statements, columns=["Module", "Coverage"]),
            hide_index=True,
            width="stretch",
        )
    with methodology:
        section_heading("Model methodology")
        st.write(
            "The dashboard uses the saved V2 HistGradientBoostingRegressor and "
            "HistGradientBoostingClassifier pipelines. Each pipeline includes feature engineering, "
            "training-fitted imputers, and categorical encoding. V1 Random Forest artifacts remain available "
            "as backups and are not overwritten. V2 hyperparameters and its 0.48 classification threshold "
            "were selected using validation data; the held-out test partition is used only for final evaluation."
        )
        st.write(
            "The six order-placement features are Order_Date, Time_Orderd, Type_of_order, City, Festival, "
            "and distance_km. The app reuses the project feature engineering and does not execute notebooks "
            "or retrain models."
        )
        st.write(
            "The optimization compared V1 Random Forests with Ridge Regression and Logistic Regression "
            "controls, then evaluated constrained forests and V2 histogram gradient boosting. PS6 compares "
            "K-Means with Haversine DBSCAN; PS7 uses exploratory K-Means rider segmentation."
        )
        st.write(
            "Distance availability is conditional: the saved feature-availability report assumes selected-store "
            "and customer coordinates are known before prediction. The operational timing could not be verified."
        )
        section_heading("Technology")
        st.write("Python · Pandas · NumPy · scikit-learn · joblib · Streamlit · Plotly")
        section_heading("Assumptions and limitations")
        st.write(
            "The 30-minute delay lens is analytical, not an official customer promise. City is a service "
            "category, not a locality name. Rider tenure is not recorded. DBSCAN hotspots are demand "
            "concentrations—not confirmed dark-store locations—and can be broad due to density chaining. "
            "Rider clusters have weak separation (silhouette ≈0.247) and remain exploratory. Model scores are "
            "not calibrated probabilities, and the 30-minute threshold is an analytical lens, not an official promise."
        )
        section_heading("Future improvements")
        st.write(
            "Validate the models with actual quick-commerce orders; confirm when distance is available; "
            "capture locality, inventory, and dispatch context; and reassess model performance and score "
            "calibration on future chronological data."
        )

    section_heading("V2 held-out test evaluation", "Metrics loaded from the saved optimization summary; no values are hardcoded.")
    summary = _json("optimization_summary")
    try:
        test = summary["test_evaluation_after_selection_only"]["V2_experimental"]
        regression = test["regression"]
        classification = test["classification_at_selected_threshold"]
        threshold_data = _json("v2_threshold")
        if not isinstance(threshold_data, dict):
            return
        threshold = float(threshold_data["threshold"])
        regression_columns, classification_columns = st.columns(2, gap="large")
        with regression_columns:
            section_heading("Delivery-time regression")
            kpi_grid(
                [
                    {"label": "MAE", "value": f"{float(regression['mae']):.2f} min"},
                    {"label": "RMSE", "value": f"{float(regression['rmse']):.2f} min"},
                    {"label": "R²", "value": f"{float(regression['r2']):.3f}"},
                ]
            )
        with classification_columns:
            section_heading(f"Late-risk classification · threshold {threshold:.2f}")
            kpi_grid(
                [
                    {"label": "ROC-AUC", "value": f"{float(classification['roc_auc']):.3f}"},
                    {"label": "PR-AUC", "value": f"{float(classification['pr_auc']):.3f}"},
                    {"label": "Precision", "value": f"{float(classification['precision']):.3f}"},
                    {"label": "Recall", "value": f"{float(classification['recall']):.3f}"},
                    {"label": "F1", "value": f"{float(classification['f1']):.3f}"},
                    {"label": "Accuracy", "value": f"{float(classification['accuracy']):.3f}"},
                ]
            )
            st.caption(
                "Confusion matrix (actual on-time/late rows × predicted on-time/late columns): "
                f"{classification['confusion_matrix']}"
            )
    except (TypeError, KeyError, ValueError) as exc:
        st.error(f"Saved V2 evaluation summary is unavailable or malformed: {exc}")


def _navigation() -> None:
    groups = {
        "OVERVIEW": [
            st.Page(
                render_executive_overview,
                title="Executive Overview",
                icon=":material/dashboard:",
                default=True,
            )
        ],
        "ANALYTICS": [
            st.Page(
                render_delivery_analytics,
                title="Delivery Analytics",
                icon=":material/monitoring:",
            ),
            st.Page(
                render_geographic_intelligence,
                title="Geographic Intelligence",
                icon=":material/map:",
            ),
            st.Page(
                render_rider_intelligence,
                title="Rider Intelligence",
                icon=":material/groups:",
            ),
        ],
        "PREDICTION": [
            st.Page(
                render_delivery_time_prediction,
                title="Delivery Time Prediction",
                icon=":material/schedule:",
            ),
            st.Page(
                render_late_delivery_risk,
                title="Late Delivery Risk",
                icon=":material/health_and_safety:",
            ),
        ],
        "DECISIONS": [
            st.Page(
                render_management_recommendations,
                title="Management Recommendations",
                icon=":material/assignment_turned_in:",
            )
        ],
        "PROJECT": [
            st.Page(
                render_about,
                title="About the Project",
                icon=":material/info:",
            )
        ],
    }
    with st.sidebar:
        render_brand()
    navigation = st.navigation(groups, position="sidebar", expanded=True)
    with st.sidebar:
        st.html(
            '<div style="height:0.6rem"></div>'
            '<div class="sidebar-footnote">Manthan Hackathon&nbsp; · &nbsp;KLU + Path-3</div>'
        )
    navigation.run()


def main() -> None:
    load_stylesheet()
    _navigation()


if __name__ == "__main__":
    main()
