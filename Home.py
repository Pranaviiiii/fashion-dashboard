# Home.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Home",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data
def load_trend_share(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df.sort_index()


labels_df = load_csv("data/processed/trends/cluster_labels.csv")
trend_scores_df = load_csv("data/processed/trends/instagram_trend_scores.csv")
forecasts_df = load_csv("data/processed/trends/cluster_forecasts.csv")
backtest_df = load_csv("data/processed/trends/forecast_backtest_metrics.csv")
ig_exemplars_df = load_csv("data/processed/exemplars/instagram_exemplars.csv")
pin_exemplars_df = load_csv("data/processed/exemplars/pinterest_exemplars.csv")
trend_share_df = load_trend_share("data/processed/trends/instagram_cluster_share_pct.csv")

label_map: Dict[int, str] = {}
if not labels_df.empty and {"cluster_id", "label"}.issubset(labels_df.columns):
    labels_df["cluster_id"] = pd.to_numeric(labels_df["cluster_id"], errors="coerce")
    labels_df = labels_df.dropna(subset=["cluster_id", "label"])
    labels_df["cluster_id"] = labels_df["cluster_id"].astype(int)
    labels_df["label"] = labels_df["label"].astype(str)
    label_map = dict(zip(labels_df["cluster_id"], labels_df["label"]))

st.title("Home")

st.markdown(
    """
## Interactive Dashboard for CLIP-Based Fashion Trend Discovery and Prediction

This dashboard presents the outputs of a multimodal fashion trend analysis pipeline.

### Data sources
- **Instagram**: caption-based fashion posts, used for time-based trend analysis
- **Pinterest**: image-led fashion content, used to enrich visual and semantic interpretation

### Processing pipeline
1. Instagram captions and Pinterest images were embedded into a shared feature space using **CLIP**
2. The combined embeddings were clustered into fashion archetypes using **K-means**
3. Monthly Instagram cluster shares were computed to identify **emerging** and **declining** trends
4. A lightweight baseline forecast model was applied to estimate short-term movement in cluster popularity
"""
)

st.markdown("## Platform note: Pinterest")
st.info(
    "Pinterest content in this dataset is highly visually consistent and maps to a single dominant cluster. "
    "This suggests weaker visual differentiation compared to Instagram’s text-driven trend structure."
)

st.caption(
    "This is likely driven by two factors: first, the Pinterest dataset is curated around visually similar outfit content; "
    "second, Pinterest’s ranking and recommendation system tends to surface aesthetically similar images. "
    "As a result, Instagram contributes most of the temporal and semantic diversity in this project, "
    "while Pinterest mainly provides visual context."
)

st.markdown("## Key Findings")
st.caption("This section summarises the most important findings from the current run of the pipeline.")

if not trend_scores_df.empty:
    emerging_top = trend_scores_df.sort_values("momentum_score", ascending=False).head(1)
    declining_top = trend_scores_df.sort_values("momentum_score", ascending=True).head(1)

    best_forecast_cluster = None
    if not backtest_df.empty and "mae" in backtest_df.columns:
        best_forecast_cluster = backtest_df.sort_values("mae", ascending=True).head(1)

    date_range = ""
    if not trend_share_df.empty and len(trend_share_df.index) > 0:
        start_date = pd.to_datetime(trend_share_df.index.min()).strftime("%Y-%m")
        end_date = pd.to_datetime(trend_share_df.index.max()).strftime("%Y-%m")
        date_range = f"{start_date} to {end_date}"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if not emerging_top.empty:
            st.info(f"**Top emerging cluster**\n\n{emerging_top.iloc[0]['label']}")

    with c2:
        if not declining_top.empty:
            st.info(f"**Top declining cluster**\n\n{declining_top.iloc[0]['label']}")

    with c3:
        if best_forecast_cluster is not None and not best_forecast_cluster.empty:
            st.info(f"**Best backtest fit**\n\n{best_forecast_cluster.iloc[0]['label']}")

    with c4:
        st.info(f"**Date coverage**\n\n{date_range if date_range else 'Unavailable'}")

st.markdown("## Overview")
st.caption("These summary metrics show the scale of the current analysis outputs.")

col1, col2, col3, col4 = st.columns(4)

n_clusters = len(label_map) if label_map else 0
n_trend_scores = len(trend_scores_df) if not trend_scores_df.empty else 0
n_forecasts = len(forecasts_df) if not forecasts_df.empty else 0
n_ig_ex = len(ig_exemplars_df) if not ig_exemplars_df.empty else 0

col1.metric("Named clusters", n_clusters)
col2.metric("Scored clusters", n_trend_scores)
col3.metric("Forecast rows", n_forecasts)
col4.metric("Instagram exemplars", n_ig_ex)

with st.expander("Methodology and limitations"):
    st.markdown(
        """
### Method summary
- CLIP was used to embed Instagram captions and Pinterest images into a shared multimodal space
- K-means clustering grouped these embeddings into interpretable fashion archetypes
- Monthly Instagram cluster shares were used to compute momentum scores
- Short-term forecasts were generated using a simple baseline model

### Important limitations
- The dataset is historical and covers a limited time window
- Forecasts are exploratory and based on a lightweight model
- Some clusters reflect **commercial activity** rather than purely stylistic trends
- Pinterest source URLs may not always remain accessible
- Pinterest contributes less cluster diversity than Instagram in this dataset
"""
    )

st.markdown("## How to navigate")
st.markdown(
    """
Use the page navigation in the sidebar to move between:

- **Trend Analysis**: emerging and declining trends plus historical trend chart
- **Forecasting**: forecast outputs and backtest metrics
- **Trend Explorer**: deep dive into one trend with charts and representative examples
"""
)