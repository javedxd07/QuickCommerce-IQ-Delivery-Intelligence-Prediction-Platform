"""Reusable presentation components for the Streamlit dashboard."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.dashboard_utils import PROJECT_ROOT


def load_stylesheet() -> None:
    css_path = PROJECT_ROOT / "assets" / "styles.css"
    if not css_path.is_file():
        st.error(f"Dashboard stylesheet is missing: {css_path}")
        return
    st.html(f"<style>{css_path.read_text(encoding='utf-8')}</style>")


def render_brand() -> None:
    st.logo(
        str(PROJECT_ROOT / "assets" / "brand.svg"),
        size="large",
        link=None,
        icon_image=str(PROJECT_ROOT / "assets" / "brand.svg"),
    )


def render_sidebar_footer() -> None:
    st.html(
        """
        <div class="sidebar-footnote">Manthan Hackathon <span aria-hidden="true">·</span> KLU + Path-3</div>
        """
    )


def page_header(
    eyebrow: str,
    title: str,
    description: str,
    context: str | None = None,
) -> None:
    context_html = (
        f'<div class="page-context">{html.escape(context)}</div>' if context else ""
    )
    st.html(
        f"""
        <header class="page-header">
          <div class="page-eyebrow">{html.escape(eyebrow)}</div>
          <h1 class="page-title">{html.escape(title)}</h1>
          <div class="page-description">{html.escape(description)}</div>
          {context_html}
        </header>
        """
    )


def section_heading(title: str, description: str | None = None) -> None:
    description_html = (
        f'<div class="section-heading-description">{html.escape(description)}</div>'
        if description
        else ""
    )
    st.html(
        f"""
        <div class="section-heading">
          <h2 class="section-heading-title">{html.escape(title)}</h2>
          {description_html}
        </div>
        """
    )


def kpi_grid(metrics: list[dict[str, str]]) -> None:
    cards = []
    for metric in metrics:
        label = html.escape(metric.get("label", ""))
        value = html.escape(metric.get("value", "—"))
        context = html.escape(metric.get("context", ""))
        context_html = f"<div class=\"kpi-context\">{context}</div>" if context else ""
        cards.append(
            f'<article class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{context_html}</article>'
        )
    count_class = (
        "kpi-count-4"
        if len(metrics) == 4
        else "kpi-count-3"
        if len(metrics) == 3
        else ""
    )
    st.html(f'<section class="kpi-grid {count_class}">{"".join(cards)}</section>')


def insight_panel(title: str, body: str) -> None:
    st.html(
        f"""
        <aside class="insight-panel">
          <div class="insight-title">{html.escape(title)}</div>
          <div class="insight-copy">{html.escape(body)}</div>
        </aside>
        """
    )


def status_banner(message: str, variant: str = "info") -> None:
    variants = {"success": "success", "warning": "warning", "danger": "danger"}
    suffix = variants.get(variant, "")
    st.html(
        f'<div class="status-banner status-banner-{suffix}">{html.escape(message)}</div>'
    )


def status_pill(label: str, level: str = "neutral") -> str:
    level_class = {
        "high": "high",
        "medium": "medium",
        "neutral": "neutral",
    }.get(level.casefold(), "neutral")
    return (
        f'<span class="status-pill status-{level_class}">'
        f"{html.escape(label)}</span>"
    )


def result_panel(
    label: str,
    value: str,
    description: str,
    status: str | None = None,
    progress: float | None = None,
    threshold: float | None = None,
) -> None:
    status_html = (
        f'<div>{status}</div>' if status else ""
    )
    progress_html = ""
    if progress is not None:
        safe_progress = min(max(progress, 0.0), 1.0)
        threshold_html = ""
        if threshold is not None:
            threshold_position = min(max(threshold, 0.0), 1.0) * 100
            threshold_html = (
                f'<span class="threshold-marker" style="left:{threshold_position:.3f}%"></span>'
            )
        progress_html = (
            '<div class="threshold-track" role="progressbar" '
            f'aria-valuenow="{safe_progress:.3f}" aria-valuemin="0" aria-valuemax="1">'
            f'<div class="threshold-fill" style="width:{safe_progress * 100:.3f}%"></div>'
            f"{threshold_html}</div>"
        )
    st.html(
        f"""
        <section class="result-panel">
          <div class="result-label">{html.escape(label)}</div>
          <div class="result-value">{html.escape(value)}</div>
          {status_html}
          {progress_html}
          <div class="result-copy">{html.escape(description)}</div>
        </section>
        """
    )


def recommendation_card(recommendation: dict[str, Any]) -> None:
    rank = html.escape(str(recommendation.get("rank", "")))
    title = html.escape(str(recommendation.get("title", "Recommendation")))
    priority = str(recommendation.get("priority", "Priority not specified"))
    priority_level = "high" if priority.casefold() == "high" else "medium"
    finding = html.escape(str(recommendation.get("observed_finding", "")))
    problem = html.escape(str(recommendation.get("operational_problem", "")))
    action = html.escape(str(recommendation.get("recommended_management_action", "")))
    evidence = recommendation.get("supporting_evidence", {})
    evidence_fields = [
        ("orders", "Orders", "number"),
        ("riders", "Riders", "number"),
        ("deliveries", "Deliveries", "number"),
        ("hotspot_orders", "Hotspot orders", "number"),
        ("late_rate_pct", "Late rate", "percent"),
        ("network_late_rate_pct", "Network late rate", "percent"),
        ("difference_vs_network_percentage_points", "Vs network", "points"),
        ("average_delivery_time_minutes", "Average time", "minutes"),
        ("rider_count", "Riders", "number"),
        ("segment_count", "Riders", "number"),
        ("hotspot_demand_share_pct", "Demand share", "percent"),
        ("demand_share_pct", "Demand share", "percent"),
        ("deliveries_per_active_day", "Deliveries / active day", "decimal"),
    ]
    evidence_items: list[str] = []
    if isinstance(evidence, dict):
        used_labels: set[str] = set()
        for field, label, style in evidence_fields:
            if field not in evidence or label in used_labels:
                continue
            value = evidence[field]
            if style == "percent":
                formatted = f"{float(value):.2f}%"
            elif style == "points":
                formatted = f"{float(value):+.2f} pp"
            elif style == "minutes":
                formatted = f"{float(value):.2f} min"
            elif style == "decimal":
                formatted = f"{float(value):.2f}"
            else:
                formatted = f"{int(value):,}"
            evidence_items.append(
                f'<div class="evidence-item"><div class="evidence-value">'
                f'{html.escape(formatted)}</div><div class="evidence-label">'
                f"{html.escape(label)}</div></div>"
            )
            used_labels.add(label)
    evidence_html = (
        f'<div class="evidence-grid">{"".join(evidence_items)}</div>'
        if evidence_items
        else ""
    )
    st.markdown(
        f"""
        <article class="recommendation-card">
          <div class="recommendation-topline">
            <span class="recommendation-rank">Recommendation {rank}</span>
            {status_pill(priority, priority_level)}
          </div>
          <h3 class="recommendation-title">{title}</h3>
          <div class="recommendation-label">Observed finding</div>
          <div class="recommendation-copy">{finding}</div>
          <div class="recommendation-label">Business problem</div>
          <div class="recommendation-copy">{problem}</div>
          <div class="recommendation-label">Supporting evidence</div>
          {evidence_html}
          <div class="recommendation-label">Recommended action</div>
          <div class="recommendation-copy">{action}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )
