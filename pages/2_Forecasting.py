from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Forecasting", layout="wide")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


forecasts_df = load_csv("data/processed/trends/cluster_forecasts.csv")
backtest_df = load_csv("data/processed/trends/forecast_backtest_metrics.csv")

st.title("Forecasting")
st.markdown(
    """
This page presents the short-term forecast outputs for each cluster and the backtest results used to evaluate forecast quality.

The current approach compares:
- a **linear trend forecast**, which projects recent movement forward, and
- a **naive persistence baseline**, which assumes the next value will be the same as the most recent observed value.

This makes the forecasting results more interpretable, as forecast errors are no longer shown in isolation.
"""
)

st.markdown("## Forecast Table")
st.caption(
    """
**Column guide**
- **cluster_id**: numeric cluster identifier  
- **label**: human-readable cluster name  
- **forecast_month**: month being predicted  
- **forecast_step**: how many steps ahead the prediction is  
- **predicted_share_pct**: forecast from the linear trend model  
- **predicted_share_pct_naive**: forecast from the naive persistence baseline  
- **last_observed_share_pct**: most recent observed cluster share used as the baseline reference  
- **slope**: trend direction and rate of change in the linear model  
- **intercept**: starting point of the linear model
"""
)

if not forecasts_df.empty:
    display_forecasts = forecasts_df.copy()

    forecast_cols = [
        "cluster_id",
        "label",
        "forecast_month",
        "forecast_step",
        "predicted_share_pct",
        "predicted_share_pct_naive",
        "last_observed_share_pct",
        "slope",
        "intercept",
    ]
    forecast_cols = [c for c in forecast_cols if c in display_forecasts.columns]

    st.dataframe(
        display_forecasts[forecast_cols],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("Forecast file not found.")

st.markdown("## Forecast Backtest")
st.caption(
    """
**Column guide**
- **cluster_id**: numeric cluster identifier  
- **label**: cluster name  
- **actual_last_share_pct**: observed final share value  
- **predicted_last_share_pct_linear**: backtest prediction from the linear trend model  
- **predicted_last_share_pct_naive**: backtest prediction from the naive baseline  
- **mae_linear**: absolute error of the linear model, lower is better  
- **mae_naive**: absolute error of the naive baseline, lower is better  
- **mape_pct_linear**: percentage error of the linear model  
- **mape_pct_naive**: percentage error of the naive baseline  
- **better_model**: which model performed better on the final holdout point  
- **mae_improvement_vs_naive**: positive values mean the linear model improved on the naive baseline
"""
)

if not backtest_df.empty:
    display_backtest = backtest_df.copy()

    # Summary metrics
    st.markdown("### Backtest Summary")

    total_clusters = len(display_backtest)
    linear_wins = (
        (display_backtest["better_model"] == "linear").sum()
        if "better_model" in display_backtest.columns
        else 0
    )
    naive_wins = (
        (display_backtest["better_model"] == "naive").sum()
        if "better_model" in display_backtest.columns
        else 0
    )
    ties = (
        (display_backtest["better_model"] == "tie").sum()
        if "better_model" in display_backtest.columns
        else 0
    )

    mean_mae_linear = (
        display_backtest["mae_linear"].mean()
        if "mae_linear" in display_backtest.columns
        else None
    )
    mean_mae_naive = (
        display_backtest["mae_naive"].mean()
        if "mae_naive" in display_backtest.columns
        else None
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clusters tested", total_clusters)
    c2.metric("Linear wins", linear_wins)
    c3.metric("Naive wins", naive_wins)
    c4.metric("Ties", ties)
    c5.metric(
        "Avg MAE improvement",
        (
            round(mean_mae_naive - mean_mae_linear, 4)
            if mean_mae_linear is not None and mean_mae_naive is not None
            else "N/A"
        ),
    )

    st.markdown("### Detailed Backtest Results")

    backtest_cols = [
        "cluster_id",
        "label",
        "actual_last_share_pct",
        "predicted_last_share_pct_linear",
        "predicted_last_share_pct_naive",
        "mae_linear",
        "mae_naive",
        "mape_pct_linear",
        "mape_pct_naive",
        "better_model",
        "mae_improvement_vs_naive",
    ]
    backtest_cols = [c for c in backtest_cols if c in display_backtest.columns]

    st.dataframe(
        display_backtest[backtest_cols],
        use_container_width=True,
        hide_index=True,
    )

    if "better_model" in display_backtest.columns:
        st.markdown("### Clusters where the linear model beat the naive baseline")
        linear_better = display_backtest[display_backtest["better_model"] == "linear"]

        if not linear_better.empty:
            st.dataframe(
                linear_better[
                    [
                        c
                        for c in [
                            "cluster_id",
                            "label",
                            "mae_linear",
                            "mae_naive",
                            "mae_improvement_vs_naive",
                        ]
                        if c in linear_better.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("The linear model did not outperform the naive baseline on any cluster.")
else:
    st.warning("Backtest metrics file not found.")

with st.expander("How to interpret these results"):
    st.markdown(
        """
The forecast table gives a short-term estimate of where each cluster may move next.

The backtest table evaluates the forecasting logic on a final holdout point by comparing:
- a **linear trend model**, and
- a **naive persistence baseline**.

This is a lightweight and exploratory evaluation rather than a production-grade forecasting framework.  
Given the short and noisy time series, the main purpose is to assess whether the linear model adds value beyond a simple baseline.
"""
    )