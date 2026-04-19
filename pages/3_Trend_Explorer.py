from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Trend Explorer", layout="wide")


# -----------------------------
# Load helpers
# -----------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data
def load_share_csv(path: str) -> pd.DataFrame:
    """
    Loads share/count trend tables where the first column is the month index.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    return df


def get_cluster_type(label: str) -> str:
    label_l = str(label).lower()

    if any(
        x in label_l
        for x in [
            "promo",
            "promotion",
            "shop",
            "seller",
            "resale",
            "preloved",
            "marketplace",
        ]
    ):
        return "Commerce-driven"
    if any(
        x in label_l
        for x in ["luxury", "boutique", "designer", "wedding", "glam"]
    ):
        return "Luxury"
    if any(
        x in label_l
        for x in ["blogger", "ootd", "streetstyle", "streetwear", "fashion week"]
    ):
        return "Editorial / Influencer"
    return "Style-driven"


def cluster_summary_sentence(label: str, latest_share: float, momentum: float) -> str:
    if momentum > 8:
        direction = "showing strong upward momentum"
    elif momentum > 2:
        direction = "showing moderate upward momentum"
    elif momentum < -8:
        direction = "declining sharply"
    elif momentum < -2:
        direction = "softening"
    else:
        direction = "remaining broadly stable"

    return (
        f"'{label}' currently represents {latest_share:.2f}% of observed Instagram cluster share "
        f"and is {direction}."
    )


def to_long_share_df(share_df: pd.DataFrame) -> pd.DataFrame:
    if share_df.empty:
        return pd.DataFrame(columns=["month", "cluster_id", "share_pct"])

    df = share_df.copy()

    # Ensure clean month index
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()

    # Keep only numeric cluster columns
    valid_cols = []
    for c in df.columns:
        try:
            int(str(c))
            valid_cols.append(c)
        except Exception:
            continue

    df = df[valid_cols].copy()
    df["month"] = df.index

    long_df = df.melt(
        id_vars=["month"],
        var_name="cluster_id",
        value_name="share_pct",
    )

    long_df["cluster_id"] = pd.to_numeric(long_df["cluster_id"], errors="coerce")
    long_df["share_pct"] = pd.to_numeric(long_df["share_pct"], errors="coerce")
    long_df = long_df.dropna(subset=["cluster_id", "share_pct"])
    long_df["cluster_id"] = long_df["cluster_id"].astype(int)

    return long_df.sort_values(["cluster_id", "month"])


def load_label_map(labels_df: pd.DataFrame) -> Dict[int, str]:
    if labels_df.empty or "cluster_id" not in labels_df.columns or "label" not in labels_df.columns:
        return {}
    df = labels_df.copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df.dropna(subset=["cluster_id", "label"])
    df["cluster_id"] = df["cluster_id"].astype(int)
    return dict(zip(df["cluster_id"], df["label"]))


def build_cluster_options(
    share_long: pd.DataFrame,
    labels_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
) -> List[int]:
    cluster_ids = set()

    if not share_long.empty and "cluster_id" in share_long.columns:
        cluster_ids.update(
            pd.to_numeric(share_long["cluster_id"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

    if not labels_df.empty and "cluster_id" in labels_df.columns:
        cluster_ids.update(
            pd.to_numeric(labels_df["cluster_id"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

    if not forecast_df.empty and "cluster_id" in forecast_df.columns:
        cluster_ids.update(
            pd.to_numeric(forecast_df["cluster_id"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

    if not backtest_df.empty and "cluster_id" in backtest_df.columns:
        cluster_ids.update(
            pd.to_numeric(backtest_df["cluster_id"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

    return sorted(cluster_ids)


def safe_float(row: pd.Series, key: str) -> float:
    return float(row[key]) if key in row.index and pd.notna(row[key]) else float("nan")


# -----------------------------
# Load data
# -----------------------------
share_df = load_share_csv("data/processed/trends/instagram_cluster_share_pct.csv")
counts_df = load_share_csv("data/processed/trends/instagram_cluster_counts.csv")
labels_df = load_csv("data/processed/trends/cluster_labels.csv")
forecast_df = load_csv("data/processed/trends/cluster_forecasts.csv")
backtest_df = load_csv("data/processed/trends/forecast_backtest_metrics.csv")
instagram_ex_df = load_csv("data/processed/exemplars/instagram_exemplars.csv")
pinterest_ex_df = load_csv("data/processed/exemplars/pinterest_exemplars.csv")

share_long = to_long_share_df(share_df)
label_map = load_label_map(labels_df)

for df in [forecast_df, backtest_df, instagram_ex_df, pinterest_ex_df]:
    if not df.empty and "cluster_id" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
        df.dropna(subset=["cluster_id"], inplace=True)
        df["cluster_id"] = df["cluster_id"].astype(int)

if not forecast_df.empty and "forecast_month" in forecast_df.columns:
    forecast_df["forecast_month"] = pd.to_datetime(forecast_df["forecast_month"], errors="coerce")

cluster_ids = build_cluster_options(share_long, labels_df, forecast_df, backtest_df)

st.title("Trend Explorer")

if not cluster_ids:
    st.warning("No trend data available. Run the pipeline first.")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
display_options = {}
for cid in cluster_ids:
    label = label_map.get(cid, f"Cluster {cid}")
    display_options[f"{label}"] = cid

selected_label = st.sidebar.selectbox("Choose a trend", list(display_options.keys()))
selected_cluster_id = display_options[selected_label]
selected_label_text = label_map.get(selected_cluster_id, f"Cluster {selected_cluster_id}")

# -----------------------------
# Filter
# -----------------------------
cluster_share = share_long[share_long["cluster_id"] == selected_cluster_id].copy().sort_values("month")
cluster_forecast = forecast_df[forecast_df["cluster_id"] == selected_cluster_id].copy() if not forecast_df.empty else pd.DataFrame()
cluster_forecast = cluster_forecast.sort_values("forecast_month") if not cluster_forecast.empty else cluster_forecast
cluster_backtest = backtest_df[backtest_df["cluster_id"] == selected_cluster_id].copy() if not backtest_df.empty else pd.DataFrame()
cluster_instagram_ex = instagram_ex_df[instagram_ex_df["cluster_id"] == selected_cluster_id].copy() if not instagram_ex_df.empty else pd.DataFrame()
cluster_pinterest_ex = pinterest_ex_df[pinterest_ex_df["cluster_id"] == selected_cluster_id].copy() if not pinterest_ex_df.empty else pd.DataFrame()

# -----------------------------
# Summary
# -----------------------------
latest_share = float(cluster_share["share_pct"].iloc[-1]) if not cluster_share.empty else 0.0
momentum_score = 0.0
if len(cluster_share) >= 2:
    lookback_idx = max(0, len(cluster_share) - 3)
    momentum_score = float(cluster_share["share_pct"].iloc[-1] - cluster_share["share_pct"].iloc[lookback_idx])

trend_type = get_cluster_type(selected_label_text)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Trend ID", selected_cluster_id)
m2.metric("Latest Share", f"{latest_share:.2f}%")
m3.metric("Momentum Score", f"{momentum_score:.2f}")
m4.metric("Trend Type", trend_type)

st.markdown(f"**Label:** {selected_label_text}")
st.caption(cluster_summary_sentence(selected_label_text, latest_share, momentum_score))

# -----------------------------
# Tabs to reduce page length
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Charts", "Backtest", "Exemplars"])

# -----------------------------
# TAB 1: Charts
# -----------------------------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Historical Trend")
        if not cluster_share.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(cluster_share["month"], cluster_share["share_pct"], marker="o", linewidth=2)
            ax.set_title(selected_label_text)
            ax.set_ylabel("Cluster Share of Posts (%)")
            ax.set_xlabel("Month")
            ax.grid(True, alpha=0.3)

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            st.pyplot(fig)
        else:
            st.info("No historical trend data available.")

    with c2:
        st.markdown("### Forecast View")
        if not cluster_share.empty or not cluster_forecast.empty:
            fig, ax = plt.subplots(figsize=(7, 4))

            if not cluster_share.empty:
                ax.plot(
                    cluster_share["month"],
                    cluster_share["share_pct"],
                    marker="o",
                    linewidth=2,
                    label="Historical share",
                )

            if not cluster_forecast.empty and "predicted_share_pct" in cluster_forecast.columns:
                ax.plot(
                    cluster_forecast["forecast_month"],
                    cluster_forecast["predicted_share_pct"],
                    marker="o",
                    linestyle="--",
                    linewidth=2,
                    label="Linear forecast",
                )

            if not cluster_forecast.empty and "predicted_share_pct_naive" in cluster_forecast.columns:
                ax.plot(
                    cluster_forecast["forecast_month"],
                    cluster_forecast["predicted_share_pct_naive"],
                    marker="o",
                    linestyle=":",
                    linewidth=2,
                    label="Naive baseline",
                )

            ax.set_title(selected_label_text)
            ax.set_ylabel("Cluster Share of Posts (%)")
            ax.set_xlabel("Month")
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            st.pyplot(fig)
        else:
            st.info("No forecast data available.")

# -----------------------------
# TAB 2: Backtest
# -----------------------------
with tab2:
    st.markdown("### Forecast Backtest")

    if not cluster_backtest.empty:
        row = cluster_backtest.iloc[0]

        actual_val = safe_float(row, "actual_last_share_pct")
        pred_linear = (
            safe_float(row, "predicted_last_share_pct_linear")
            if "predicted_last_share_pct_linear" in row.index
            else safe_float(row, "predicted_last_share_pct")
        )
        pred_naive = safe_float(row, "predicted_last_share_pct_naive")

        mae_linear = (
            safe_float(row, "mae_linear")
            if "mae_linear" in row.index
            else safe_float(row, "mae")
        )
        mae_naive = safe_float(row, "mae_naive")

        mape_linear = (
            safe_float(row, "mape_pct_linear")
            if "mape_pct_linear" in row.index
            else safe_float(row, "mape_pct")
        )
        mape_naive = safe_float(row, "mape_pct_naive")

        better_model = row["better_model"] if "better_model" in row.index and pd.notna(row["better_model"]) else "N/A"
        improvement = safe_float(row, "mae_improvement_vs_naive")

        b1, b2, b3 = st.columns(3)
        b1.metric("Actual last share", f"{actual_val:.2f}%")
        b2.metric("Linear MAE", f"{mae_linear:.2f}")
        b3.metric("Naive MAE", f"{mae_naive:.2f}" if pd.notna(mae_naive) else "N/A")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**Linear forecast:** {pred_linear:.2f}%")
            st.markdown(f"**Linear MAPE:** {mape_linear:.2f}%")
        with d2:
            if pd.notna(pred_naive):
                st.markdown(f"**Naive baseline forecast:** {pred_naive:.2f}%")
            if pd.notna(mape_naive):
                st.markdown(f"**Naive MAPE:** {mape_naive:.2f}%")

        if better_model != "N/A":
            st.markdown(f"**Better model:** {better_model}")

        if pd.notna(improvement):
            if improvement > 0:
                st.success(f"Linear model improved on the naive baseline by {improvement:.2f} MAE points.")
            elif improvement < 0:
                st.info(f"Naive baseline outperformed the linear model by {abs(improvement):.2f} MAE points.")
            else:
                st.info("Linear and naive models performed equally on this cluster.")
    else:
        st.info("No backtest metrics available for this cluster.")

# -----------------------------
# TAB 3: Exemplars
# -----------------------------
with tab3:
    e1, e2 = st.columns(2)

    with e1:
        st.markdown("### Instagram Exemplars")

        if not cluster_instagram_ex.empty:
            # Try common text column names
            text_candidates = [
                "caption",
                "text",
                "post_text",
                "caption_clean",
                "clean_caption",
                "content",
            ]
            distance_candidates = [
                "distance_to_centroid",
                "distance",
                "centroid_distance",
                "score",
            ]

            text_col = next((c for c in text_candidates if c in cluster_instagram_ex.columns), None)
            dist_col = next((c for c in distance_candidates if c in cluster_instagram_ex.columns), None)

            display_cols = [c for c in ["rank"] if c in cluster_instagram_ex.columns]
            if text_col:
                display_cols.append(text_col)
            if dist_col:
                display_cols.append(dist_col)

            # Fallback: show first useful columns if expected names don't exist
            if len(display_cols) <= 1:
                preferred_cols = [
                    c
                    for c in cluster_instagram_ex.columns
                    if c not in ["cluster_id"]
                ]
                display_cols = preferred_cols[:4]

            display_df = cluster_instagram_ex[display_cols].copy()

            # Rename for nicer display
            rename_map = {}
            if text_col and text_col in display_df.columns:
                rename_map[text_col] = "caption"
            if dist_col and dist_col in display_df.columns:
                rename_map[dist_col] = "distance_to_centroid"
            display_df = display_df.rename(columns=rename_map)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            with st.expander("Show raw Instagram exemplar columns"):
                st.write(list(cluster_instagram_ex.columns))
        else:
            st.info("No Instagram exemplars available.")

    with e2:
        st.markdown("### Pinterest Exemplars")

        if not cluster_pinterest_ex.empty:
            text_candidates = [
                "caption",
                "text",
                "post_text",
                "title",
                "description",
                "content",
            ]
            image_candidates = [
                "image_url",
                "url",
                "img_url",
                "image",
            ]
            distance_candidates = [
                "distance_to_centroid",
                "distance",
                "centroid_distance",
                "score",
            ]

            text_col = next((c for c in text_candidates if c in cluster_pinterest_ex.columns), None)
            image_col = next((c for c in image_candidates if c in cluster_pinterest_ex.columns), None)
            dist_col = next((c for c in distance_candidates if c in cluster_pinterest_ex.columns), None)

            display_cols = [c for c in ["rank"] if c in cluster_pinterest_ex.columns]
            if text_col:
                display_cols.append(text_col)
            if image_col:
                display_cols.append(image_col)
            if dist_col:
                display_cols.append(dist_col)

            if len(display_cols) <= 1:
                preferred_cols = [
                    c
                    for c in cluster_pinterest_ex.columns
                    if c not in ["cluster_id"]
                ]
                display_cols = preferred_cols[:5]

            display_df = cluster_pinterest_ex[display_cols].copy()

            rename_map = {}
            if text_col and text_col in display_df.columns:
                rename_map[text_col] = "caption"
            if image_col and image_col in display_df.columns:
                rename_map[image_col] = "image_url"
            if dist_col and dist_col in display_df.columns:
                rename_map[dist_col] = "distance_to_centroid"
            display_df = display_df.rename(columns=rename_map)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            with st.expander("Show raw Pinterest exemplar columns"):
                st.write(list(cluster_pinterest_ex.columns))
        else:
            st.info("No Pinterest exemplars available.")

with st.expander("How to interpret this page"):
    st.markdown(
        """
This page combines historical cluster share, short-term forecasting, backtest evidence, and exemplar content for a single selected trend.

The forecasting section compares a linear trend model against a naive persistence baseline.
Given the short and noisy time series, these results should be interpreted as exploratory rather than definitive.

The exemplar tables help interpret cluster meaning by showing representative Instagram captions and Pinterest entries closest to the cluster centroid.
"""
    )