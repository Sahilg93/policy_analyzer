import streamlit as st
import pandas as pd
import plotly.express as px
from pipeline.build_dataset import build

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = build()
    return df

df = load_data()

st.title("Policy Explorer Dashboard")

# ----------------------------
# RAW DATA VIEW
# ----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(50))

# ----------------------------
# POLICY PRESSURE INDEX
# ----------------------------
if "expansionary_policy" in df.columns:
    df["policy_pressure"] = (
        df["expansionary_policy"].fillna(0)
        - df["contractionary_policy"].fillna(0)
        + 0.5 * df["high_intensity_policy"].fillna(0)
    )

    st.subheader("Policy Pressure Over Time")

    fig = px.line(
        df.groupby("year")["policy_pressure"].mean().reset_index(),
        x="year",
        y="policy_pressure"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# GDP vs POLICY
# ----------------------------
st.subheader("GDP vs Policy Pressure")

if "policy_pressure" in df.columns:
    fig = px.scatter(
        df,
        x="policy_pressure",
        y="gdp_growth",
        hover_data=["year", "state"]
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# UNEMPLOYMENT TREND
# ----------------------------
st.subheader("Unemployment Trend")

fig = px.line(
    df.groupby("year")["unemployment_rate"].mean().reset_index(),
    x="year",
    y="unemployment_rate"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# AI POLICY INSPECTION
# ----------------------------
st.subheader("AI-Classified Policy Samples")

if "policy_type" in df.columns:
    st.dataframe(
        df[[
            "title",
            "policy_type",
            "direction",
            "intensity",
            "sector"
        ]].head(30)
    )