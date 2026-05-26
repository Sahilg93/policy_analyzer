import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import requests
from pipeline.build_dataset import build

st.set_page_config(layout="wide")

# State tracking matrix mapping human titles to standard postal keys
STATE_CODE_MAP = {
    "Federal": None,
    "California": "CA",
    "Texas": "TX",
    "New York": "NY",
    "Florida": "FL",
    "Ohio": "OH",
    "Illinois": "IL",
    "Pennsylvania": "PA",
    "Michigan": "MI"
}

def calculate_cheap_pca(vectors_matrix, num_components=2):
    # Center the data matrix
    X_centered = vectors_matrix - np.mean(vectors_matrix, axis=0)
    # Vectorized Singular Value Decomposition (SVD)
    # X_centered = U * S * Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Project X_centered onto top singular vectors (Vt[:num_components].T)
    # Using np.dot for continuous memory space alignment
    return np.dot(X_centered, Vt[:num_components].T)


@st.cache_data
def get_semantic_map_data():
    hist_bills_path = "data/processed/historical_bills.parquet"
    embeddings_path = "data/processed/historical_embeddings.parquet"
    
    if not os.path.exists(hist_bills_path) or not os.path.exists(embeddings_path):
        return pd.DataFrame()
        
    try:
        df_bills = pd.read_parquet(hist_bills_path)
        df_embed = pd.read_parquet(embeddings_path)
        
        # Merge on bill_id
        df_merged = df_bills.merge(df_embed, on="bill_id", how="inner")
        if df_merged.empty:
            return pd.DataFrame()
            
        # Extract embeddings into a 2D numpy array
        vectors = np.stack(df_merged["embedding"].values)
        
        # Calculate PCA
        pca_coords = calculate_cheap_pca(vectors, num_components=2)
        df_merged["PCA 1"] = pca_coords[:, 0]
        df_merged["PCA 2"] = pca_coords[:, 1]
        
        return df_merged.drop(columns=["embedding"])
    except Exception as e:
        return pd.DataFrame()


# Parameterized cache loop allowing state injection updates
@st.cache_data(show_spinner=False)
def load_scored_bills(jurisdiction="federal", state_code=None):
    raw_results = build(jurisdiction=jurisdiction, state_code=state_code)
    if not raw_results:
        return pd.DataFrame()

    df = pd.DataFrame(raw_results)
    if "estimated_impacts" in df.columns:
        impacts_df = pd.json_normalize(df["estimated_impacts"]).add_prefix("impact_")
        df = pd.concat(
            [df.drop(columns=["estimated_impacts"]), impacts_df],
            axis=1
        )
    return df


@st.cache_data
def load_macro_data():
    return pd.read_parquet("data/processed/policy_dataset.parquet")


# ============================================================
# SIDEBAR CONTROL INTERFACE
# ============================================================
st.sidebar.title("Policy Intelligence Controls")
st.sidebar.markdown("Configure target parameters and fire live evaluation pipelines across multi-tiered jurisdictions.")

selected_pipeline = st.sidebar.selectbox(
    "Target Ingestion Pipeline",
    options=list(STATE_CODE_MAP.keys()),
    index=0
)

target_state_code = STATE_CODE_MAP[selected_pipeline]

# Add an explicit processing button trigger to prevent layout flashing
run_pipeline = st.sidebar.button("Run Simulation Pipeline Core")

# Handle baseline memory mapping states securely
if run_pipeline:
    st.sidebar.info(f"Invoking {selected_pipeline} pipeline core...")
    # Parameterized cache arguments prevent wiping other static caches
    df = load_scored_bills(jurisdiction=selected_pipeline.lower(), state_code=target_state_code)
else:
    # Warm fallbacks to protect layout structures on initialization
    df = load_scored_bills(jurisdiction="federal", state_code=None)

macro_df = load_macro_data()

# ============================================================
# DASHBOARD LAYOUT VIEW
# ============================================================
st.title("Policy Explorer Dashboard")
st.markdown(f"Currently visualizing evaluation profiles under active jurisdiction channel: **{selected_pipeline.upper()}**")

# ----------------------------
# RAW DATA VIEW
# ----------------------------
st.subheader("Scored Bills Matrix")

if df.empty:
    st.info("No scored bills are available for this specific path framework. Hit the sidebar simulation trigger to fetch live data targets.")
else:
    preview_cols = [
        "bill_id", "title", "policy_type", "direction", "net_score", 
        "confidence", "impact_num_analogs_matched", "impact_avg_similarity"
    ]
    preview_cols = [col for col in preview_cols if col in df.columns]
    st.dataframe(df[preview_cols].head(50), use_container_width=True)

# ----------------------------
# POLICY SCORE DISTRIBUTION
# ----------------------------
if not df.empty and {"title", "net_score", "confidence"}.issubset(df.columns):
    st.subheader("Policy Scores Volatility")

    fig = px.bar(
        df.sort_values("net_score", ascending=False),
        x="net_score",
        y="title",
        color="confidence",
        orientation="h",
        hover_data=["bill_id", "policy_type", "direction"],
        labels={"net_score": "Net Economic Impact Score", "title": "Legislative Act Title"}
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# ANALOG MATCH QUALITY
# ----------------------------
if not df.empty and {"impact_avg_similarity", "impact_num_analogs_matched", "net_score"}.issubset(df.columns):
    st.subheader("Analog Match Quality Dispersion")

    fig = px.scatter(
        df,
        x="impact_avg_similarity",
        y="net_score",
        size="impact_num_analogs_matched",
        color="policy_type" if "policy_type" in df.columns else None,
        hover_data=["bill_id", "title"],
        labels={"impact_avg_similarity": "Historical Peer Cosine Similarity", "net_score": "Net Calculated Score"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# MACRO DATA PREVIEW
# ----------------------------
st.subheader("Macro Dataset Time-Series Baseline")
st.dataframe(macro_df.head(50), use_container_width=True)

# ----------------------------
# GDP TREND
# ----------------------------
if {"year", "gdp"}.issubset(macro_df.columns):
    st.subheader("GDP Structural Performance Trend")

    fig = px.line(
        macro_df.groupby("year")["gdp"].mean().reset_index(),
        x="year",
        y="gdp",
        labels={"gdp": "Gross Domestic Product Baseline Value", "year": "Session Timeline"}
    )
    st.plotly_chart(fig, use_container_width=True)

if {"gdp_growth", "unemployment_rate"}.issubset(macro_df.columns):
    st.subheader("GDP Growth Curve vs Regional Unemployment Deltas")

    fig = px.scatter(
        macro_df,
        x="unemployment_rate",
        y="gdp_growth",
        color="state" if "state" in macro_df.columns else None,
        hover_data=["year", "state"],
        labels={"unemployment_rate": "BLS LAUS Unemployment %", "gdp_growth": "BEA SAGDP2 Annual Growth Rate"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# UNEMPLOYMENT TREND
# ----------------------------
st.subheader("Aggregated Unemployment Structural Trend")

fig = px.line(
    macro_df.groupby("year")["unemployment_rate"].mean().reset_index(),
    x="year",
    y="unemployment_rate",
    labels={"unemployment_rate": "Mean Unemployment Delta Score", "year": "Session Timeline"}
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# AI POLICY INSPECTION
# ----------------------------
st.subheader("AI-Enriched Deep Policy Inspection")

inspection_cols = ["title", "policy_type", "direction", "net_score", "confidence", "explanation"]
inspection_cols = [col for col in inspection_cols if col in df.columns]

if inspection_cols and not df.empty:
    st.dataframe(df[inspection_cols].head(30), use_container_width=True)

# ----------------------------
# SEMANTIC POLICY SPACE (PCA)
# ----------------------------
st.subheader("Semantic Policy Space Network Matrix (PCA)")
st.markdown(
    "This interactive semantic map projects the high-dimensional bill clean summaries "
    "down to 2D using a pure NumPy Principal Component Analysis (PCA) algorithm. Bills closer together "
    "are semantically similar in subject matter and intent."
)

df_semantic = get_semantic_map_data()

if df_semantic.empty:
    st.info("Semantic vector topological coordinate matrix not found. Build out an embed repository on disk cache first.")
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Topological Visual Filters")
    jurisdiction_options = ["All", "Federal", "California", "Texas", "New York", "Florida", "Ohio", "Illinois", "Pennsylvania", "Michigan"]
    selected_map_filter = st.sidebar.selectbox("Filter Display Slices on Scatter Map", options=jurisdiction_options, index=0)
    
    if selected_map_filter != "All":
        df_plot = df_semantic[df_semantic["jurisdiction"] == selected_map_filter.lower()]
    else:
        df_plot = df_semantic.copy()
        
    if df_plot.empty:
        st.warning(f"No cached vector markers identified for target slice: {selected_map_filter}")
    else:
        color_col = "jurisdiction" if "jurisdiction" in df_plot.columns else "policy_type"
        symbol_col = "level" if "level" in df_plot.columns else None
        
        hover_cols = ["bill_id", "title", "direction", "intensity", "state", "sponsor_party", "major_topic", "session_year"]
        hover_cols = [col for col in hover_cols if col in df_plot.columns]
        
        fig_pca = px.scatter(
            df_plot,
            x="PCA 1",
            y="PCA 2",
            color=color_col,
            symbol=symbol_col,
            hover_data=hover_cols,
            title="Cross-Jurisdictional Semantic Policy Cluster Projections",
            labels={"PCA 1": "Principal Component 1 (Axis Alpha)", "PCA 2": "Principal Component 2 (Axis Beta)"}
        )
        fig_pca.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
        fig_pca.update_layout(dragmode="pan", legend_title_text="Jurisdictional Stratification Structure")
        st.plotly_chart(fig_pca, use_container_width=True)