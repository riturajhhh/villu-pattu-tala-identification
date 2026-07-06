"""
frontend/components/visualizations.py
======================================
Interactive Plotly visualizations for the Streamlit UI.
"""

from __future__ import annotations

import base64
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st


def render_base64_image(base64_str: str, caption: str = "") -> None:
    """Safely decode base64 image and display in Streamlit."""
    try:
        img_bytes = base64.b64decode(base64_str)
        st.image(img_bytes, caption=caption, use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to render plot: {exc}")


def plot_prediction_distribution(predictions: List[Dict[str, float]]) -> None:
    """Plot an interactive Plotly bar chart of the Tala prediction confidences."""
    df = pd.DataFrame(predictions)
    
    # Sort for plotting (ascending so the highest is at top in horizontal bar)
    df = df.sort_values("confidence", ascending=True)

    fig = px.bar(
        df,
        x="confidence",
        y="tala",
        orientation="h",
        text_auto=".1%",
        color="confidence",
        color_continuous_scale="Viridis",
        labels={"confidence": "Confidence Score", "tala": "Tala Cycle"},
        title="Tala Classification Confidence Distribution"
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        coloraxis_showscale=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    fig.update_xaxes(showgrid=True, gridcolor="rgba(240, 246, 252, 0.1)", range=[0, 1])
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)
