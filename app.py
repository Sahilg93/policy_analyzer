import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import requests
from pipeline.build_dataset import build

st.set_page_config(layout="wide")


def calculate_cheap_pca(vectors_matrix, num_components=2):
    # Center the data matrix
    X_centered = vectors_matrix - np.mean(vectors_matrix, axis=0)
    # Compute Covariance Matrix
    covariance_matrix = np.cov(X_centered.T)
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort descending and project
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_vectors = eigenvectors[:, sorted_indices[:num_components]]
    return np.dot(X_centered, top_vectors)


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


@st.cache_data
def load_scored_bills():
    raw_results = build()
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


df = load_scored_bills()
macro_df = load_macro_data()

st.title("Policy Explorer Dashboard")

# ----------------------------
# RAW DATA VIEW
# ----------------------------
st.subheader("Scored Bills")

if df.empty:
    st.info("No scored bills are available. Check the Congress API connection and rerun the pipeline.")
else:
    preview_cols = [
        "bill_id",
        "title",
        "policy_type",
        "direction",
        "net_score",
        "confidence",
        "impact_num_analogs_matched",
        "impact_avg_similarity",
    ]
    preview_cols = [col for col in preview_cols if col in df.columns]
    st.dataframe(df[preview_cols].head(50), use_container_width=True)

# ----------------------------
# POLICY SCORE DISTRIBUTION
# ----------------------------
if not df.empty and {"title", "net_score", "confidence"}.issubset(df.columns):
    st.subheader("Policy Scores")

    fig = px.bar(
        df.sort_values("net_score", ascending=False),
        x="net_score",
        y="title",
        color="confidence",
        orientation="h",
        hover_data=["bill_id", "policy_type", "direction"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# ANALOG MATCH QUALITY
# ----------------------------
if not df.empty and {"impact_avg_similarity", "impact_num_analogs_matched", "net_score"}.issubset(df.columns):
    st.subheader("Analog Match Quality")

    fig = px.scatter(
        df,
        x="impact_avg_similarity",
        y="net_score",
        size="impact_num_analogs_matched",
        color="policy_type" if "policy_type" in df.columns else None,
        hover_data=["bill_id", "title"],
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# MACRO DATA PREVIEW
# ----------------------------
st.subheader("Macro Dataset Preview")
st.dataframe(macro_df.head(50), use_container_width=True)

# ----------------------------
# GDP TREND
# ----------------------------
if {"year", "gdp"}.issubset(macro_df.columns):
    st.subheader("GDP Trend")

    fig = px.line(
        macro_df.groupby("year")["gdp"].mean().reset_index(),
        x="year",
        y="gdp",
    )
    st.plotly_chart(fig, use_container_width=True)

if {"gdp_growth", "unemployment_rate"}.issubset(macro_df.columns):
    st.subheader("GDP Growth vs Unemployment")

    fig = px.scatter(
        macro_df,
        x="unemployment_rate",
        y="gdp_growth",
        hover_data=["year", "state"],
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# UNEMPLOYMENT TREND
# ----------------------------
st.subheader("Unemployment Trend")

fig = px.line(
    macro_df.groupby("year")["unemployment_rate"].mean().reset_index(),
    x="year",
    y="unemployment_rate",
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# AI POLICY INSPECTION
# ----------------------------
st.subheader("AI-Classified Policy Samples")

inspection_cols = [
    "title",
    "policy_type",
    "direction",
    "net_score",
    "confidence",
    "explanation",
]
inspection_cols = [col for col in inspection_cols if col in df.columns]

if inspection_cols:
    st.dataframe(
        df[inspection_cols].head(30),
        use_container_width=True,
    )

# ----------------------------
# SEMANTIC POLICY SPACE (PCA)
# ----------------------------
st.subheader("Semantic Policy Space (PCA)")
st.markdown(
    "This interactive semantic map projects the high-dimensional bill title embeddings (768 dimensions) "
    "down to 2D using a pure NumPy Principal Component Analysis (PCA) algorithm. Bills closer together "
    "are semantically similar in subject matter and intent."
)

df_semantic = get_semantic_map_data()

if df_semantic.empty:
    st.info("Semantic map data not available. Ensure that the embedding cache has been generated by running the pipeline.")
else:
    # Build scatter plot
    fig_pca = px.scatter(
        df_semantic,
        x="PCA 1",
        y="PCA 2",
        color="policy_type",
        hover_data=["bill_id", "title", "direction", "intensity", "state"],
        title="Historical Bills Semantic Map",
        labels={"PCA 1": "Principal Component 1", "PCA 2": "Principal Component 2"}
    )
    fig_pca.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
    fig_pca.update_layout(dragmode="pan")
    st.plotly_chart(fig_pca, use_container_width=True)
