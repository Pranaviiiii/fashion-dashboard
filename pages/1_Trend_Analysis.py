# pages/1_Trend_Analysis.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Trend Analysis", layout="wide")


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


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


def get_cluster_type(label: str) -> str:
    label_l = label.lower()

    if any(k in label_l for k in ["resale", "promo", "promotion", "seller", "marketplace", "personal shopping", "preloved"]):
        return "Commerce-driven"
    if any(k in label_l for k in ["luxury", "boutique", "glam"]):
        return "Luxury"
    if any(k in label_l for k in ["blogger", "ootd", "lookbook"]):
        return "Editorial / Influencer"
    if any(k in label_l for k in ["menswear", "streetstyle", "denim"]):
        return "Style-driven"
    return "Mixed"


labels_df = load_csv("data/processed/trends/cluster_labels.csv")
trend_scores_df = load_csv("data/processed/trends/instagram_trend_scores.csv")
trend_share_df = load_trend_share("data/processed/trends/instagram_cluster_share_pct.csv")

label_map: Dict[int, str] = {}
if not labels_df.empty and {"cluster_id", "label"}.issubset(labels_df.columns):
    labels_df["cluster_id"] = pd.to_numeric(labels_df["cluster_id"], errors="coerce")
    labels_df = labels_df.dropna(subset=["cluster_id", "label"])
    labels_df["cluster_id"] = labels_df["cluster_id"].astype(int)
    label_map = dict(zip(labels_df["cluster_id"], labels_df["label"]))

if not trend_share_df.empty:
    renamed_cols = {}
    for c in trend_share_df.columns:
        cid = safe_int(c)
        renamed_cols[c] = label_map.get(cid, f"Cluster {cid}")
    trend_share_named_df = trend_share_df.rename(columns=renamed_cols)
else:
    trend_share_named_df = pd.DataFrame()

st.title("Trend Analysis")
st.markdown(
    """
This page shows which clusters are currently emerging or declining and how their popularity changes over time.
The trend ranking is based on momentum scores derived from the monthly Instagram cluster share series.
"""
)

show_top_n = st.sidebar.slider("Number of top clusters to display", min_value=3, max_value=10, value=5, key="trend_top_n", help="Controls how many of the strongest emerging and declining fashion trends are shown in the tables and chart.")

st.sidebar.markdown(
    "This setting filters the dashboard to show only the most important trend movements."
)

st.markdown("## Top Emerging and Declining Trends")
st.caption(
    "The number of clusters shown below is controlled by the sidebar slider. "
    "This allows you to focus on the most significant trends."
)

if not trend_scores_df.empty:
    left, right = st.columns(2)

    with left:
        st.markdown("### Emerging clusters")
        emerging = trend_scores_df.sort_values("momentum_score", ascending=False).head(show_top_n).copy()
        emerging["cluster_type"] = emerging["label"].apply(get_cluster_type)
        st.dataframe(
            emerging[
                ["cluster_id", "label", "cluster_type", "momentum_score", "last_share_pct", "slope_3m_pct_points_per_month"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Higher momentum scores indicate more strongly emerging clusters.")

    with right:
        st.markdown("### Declining clusters")
        declining = trend_scores_df.sort_values("momentum_score", ascending=True).head(show_top_n).copy()
        declining["cluster_type"] = declining["label"].apply(get_cluster_type)
        st.dataframe(
            declining[
                ["cluster_id", "label", "cluster_type", "momentum_score", "last_share_pct", "slope_3m_pct_points_per_month"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Lower momentum scores indicate declining or unstable clusters.")
else:
    st.warning("Trend scores file not found.")

st.markdown("## Trend Evolution")
st.caption(
    "This plot shows the historical monthly share of Instagram posts assigned to the most prominent clusters."
)

if not trend_share_named_df.empty:
    fig, ax = plt.subplots(figsize=(12, 5))

    latest_sorted = trend_share_named_df.iloc[-1].sort_values(ascending=False)
    plot_cols = latest_sorted.head(show_top_n).index.tolist()

    for col in plot_cols:
        ax.plot(
            trend_share_named_df.index,
            trend_share_named_df[col],
            marker="o",
            linewidth=2,
            label=col,
        )

    ax.set_title("Monthly Share of Instagram Posts by Cluster")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of Posts (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Trend share file not found.")