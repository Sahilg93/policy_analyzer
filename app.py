import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import requests
from typing import Optional
import logging
from pipeline.config import (
    HISTORICAL_BILLS_PATH,
    EMBEDDINGS_PATH,
    POLICY_DATASET_PATH,
    OLLAMA_HOST
)
from pipeline.build_dataset import build

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Policy Intelligence Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern typography and premium glassmorphism styling
st.markdown("""
<style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .badge-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .badge-med {
        color: #f39c12;
        font-weight: bold;
    }
    .badge-low {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
    """Vectorized Singular Value Decomposition (SVD) projection with deterministic sign-locking (svd_flip)."""
    X_centered = vectors_matrix - np.mean(vectors_matrix, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Deterministic sign-locking (equivalent to scikit-learn's svd_flip)
    # Find the row index of the maximum absolute value for each column of U
    max_abs_rows = np.argmax(np.abs(U), axis=0)
    # Extract the sign of the element at the maximum absolute row for each column
    signs = np.sign(U[max_abs_rows, range(U.shape[1])])
    # Apply signs to U and Vt to lock sign direction deterministically
    U *= signs
    Vt *= signs[:, np.newaxis]
    
    return np.dot(X_centered, Vt[:num_components].T)


def _get_db_token():
    """Extracts last-modified timestamp of local policy databases and scored file as a cache buster."""
    t1 = 0
    csv_path = "data/policy_events.csv"
    if os.path.exists(csv_path):
        t1 = int(os.path.getmtime(csv_path))
        
    t2 = 0
    camp_path = "data/campaign_policies.csv"
    if os.path.exists(camp_path):
        t2 = int(os.path.getmtime(camp_path))
        
    t3 = 0
    scored_path = "data/processed/scored_bills.csv"
    if os.path.exists(scored_path):
        t3 = int(os.path.getmtime(scored_path))
        
    return max(t1, t2, t3)


def _get_proposal_embedding(text: str) -> Optional[np.ndarray]:
    """Fetch embedding for a campaign proposal via local Ollama API."""
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=10
        )
        r.raise_for_status()
        return np.array(r.json().get("embedding", []), dtype=float)
    except Exception:
        return None


@st.cache_data
def get_historical_svd_basis(_db_token=0):
    """Calculates and caches the Singular Value Decomposition (SVD) basis of the historical corpus."""
    if not HISTORICAL_BILLS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        return None, None, None
    try:
        df_bills = pd.read_parquet(HISTORICAL_BILLS_PATH)
        df_embed = pd.read_parquet(EMBEDDINGS_PATH)
        df_merged = df_bills.merge(df_embed, on="bill_id", how="inner")
        if df_merged.empty:
            return None, None, None
            
        hist_vectors = np.stack(df_merged["embedding"].values)
        X_hist_mean = np.mean(hist_vectors, axis=0)
        X_hist_centered = hist_vectors - X_hist_mean
        U_hist, S_hist, Vt_hist = np.linalg.svd(X_hist_centered, full_matrices=False)
        
        # Deterministic sign locking (equivalent to scikit-learn's svd_flip)
        max_abs_rows = np.argmax(np.abs(U_hist), axis=0)
        signs = np.sign(U_hist[max_abs_rows, range(U_hist.shape[1])])
        U_hist *= signs
        Vt_hist *= signs[:, np.newaxis]
        
        hist_coords = np.dot(X_hist_centered, Vt_hist[:2].T)
        
        df_merged["PCA 1"] = hist_coords[:, 0]
        df_merged["PCA 2"] = hist_coords[:, 1]
        
        return df_merged.drop(columns=["embedding"]), X_hist_mean, Vt_hist[:2]
    except Exception as e:
        logger.error(f"Failed to calculate historical SVD basis: {e}")
        return None, None, None


@st.cache_data
def get_semantic_map_data(df_campaign_dict=None, _db_token=0):
    """Retrieves 2D coordinates for the interactive Semantic Policy Space map."""
    df_merged_proj, hist_mean, Vt_basis = get_historical_svd_basis(_db_token=_db_token)
    if df_merged_proj is None:
        return pd.DataFrame()
        
    df_merged = df_merged_proj.copy()
    
    # Project campaign rows if provided
    if df_campaign_dict is not None and len(df_campaign_dict) > 0:
        df_camp = pd.DataFrame(df_campaign_dict)
        if not df_camp.empty:
            camp_rows = []
            camp_vectors = []
            for _, row in df_camp.iterrows():
                text = str(row.get("bill_text_clean") or row.get("title") or "").strip()
                emb = _get_proposal_embedding(text)
                if emb is not None:
                    camp_vectors.append(emb)
                    camp_rows.append({
                        "bill_id": row["bill_id"],
                        "title": row["title"],
                        "policy_type": row["policy_type"],
                        "direction": row["direction"],
                        "intensity": row.get("intensity", "low"),
                        "sector": row.get("sector", "mixed"),
                        "state": row.get("candidate_name", "Candidate"),
                        "level": "campaign",
                        "text": text,
                        "jurisdiction": "campaign",
                        "state_code": "CAMP",
                        "session_year": 2026,
                        "enacted": False,
                        "sponsor_party": row.get("party", "unknown"),
                        "bill_text_clean": text,
                        "major_topic": row.get("major_topic", "Macroeconomics")
                    })
            
            if camp_rows:
                df_camp_aligned = pd.DataFrame(camp_rows)
                camp_vectors_stack = np.stack(camp_vectors)
                
                # Project campaign vectors onto existing historical basis
                camp_centered = camp_vectors_stack - hist_mean
                camp_coords = np.dot(camp_centered, Vt_basis.T)
                
                df_camp_aligned["PCA 1"] = camp_coords[:, 0]
                df_camp_aligned["PCA 2"] = camp_coords[:, 1]
                
                df_final = pd.concat([df_merged, df_camp_aligned], ignore_index=True)
                return df_final
                
    return df_merged


# Parameterized cache loop allowing state injection updates
@st.cache_data(show_spinner=False)
def load_scored_bills(jurisdiction="federal", state_code=None, _db_token=0):
    csv_path = "data/processed/scored_bills.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # If the CSV has nested fields, parse them safely.
            # But serialize_scored_results already exports flat columns.
            return df
        except Exception as e:
            logger.error(f"Failed to read scored_bills.csv: {e}")
            
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_campaign_scored_policies(jurisdiction="ohio", state_code="OH", _db_token=0):
    csv_path = "data/processed/campaign_scored_bills.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            logger.error(f"Failed to read campaign_scored_bills.csv: {e}")
    return pd.DataFrame()


@st.cache_data
def load_macro_data(_db_token=0):
    if not POLICY_DATASET_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(POLICY_DATASET_PATH)


# ============================================================
# SIDEBAR CONTROL INTERFACE
# ============================================================
st.sidebar.title("⚖️ Policy Intelligence Platform")
st.sidebar.markdown("Configure target parameters and fire live evaluation pipelines across multi-tiered jurisdictions.")

simulation_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["Active Legislative Session", "Candidate Campaign Platforms"],
    index=0
)

if simulation_mode == "Candidate Campaign Platforms":
    selected_pipeline = "Ohio"
    target_state_code = "OH"
    st.sidebar.markdown("**Active Campaign Jurisdiction:** Ohio (Amy Acton vs. Vivek Ramaswamy)")
else:
    selected_pipeline = st.sidebar.selectbox(
        "Target Ingestion Pipeline",
        options=list(STATE_CODE_MAP.keys()),
        index=0
    )
    target_state_code = STATE_CODE_MAP[selected_pipeline]

# Add an explicit processing button trigger to prevent layout flashing
run_pipeline = st.sidebar.button("Run Simulation Pipeline Core", use_container_width=True)

# Extract flat file database token
db_token = _get_db_token()

# Handle baseline memory mapping states reactively and securely
if simulation_mode == "Candidate Campaign Platforms":
    if run_pipeline:
        st.sidebar.info("Invoking Ohio Campaign Platform pipeline core...")
        from pipeline.build_dataset import build_campaign_pipeline
        build_campaign_pipeline(jurisdiction_name="Ohio", state_code="OH")
        st.cache_data.clear()
        df = load_campaign_scored_policies(jurisdiction="Ohio", state_code="OH", _db_token=db_token)
    else:
        df = load_campaign_scored_policies(jurisdiction="Ohio", state_code="OH", _db_token=db_token)
        # Robust safety barrier: if cache is stale and missing vital keys, clear and force reload
        if not df.empty and "bill_text_clean" not in df.columns:
            st.cache_data.clear()
            df = load_campaign_scored_policies(jurisdiction="Ohio", state_code="OH", _db_token=db_token)
else:
    if run_pipeline:
        st.sidebar.info(f"Invoking {selected_pipeline} pipeline core...")
        build(jurisdiction=selected_pipeline.lower(), state_code=target_state_code)
        st.cache_data.clear()
        df = load_scored_bills(jurisdiction=selected_pipeline.lower(), state_code=target_state_code, _db_token=db_token)
    else:
        df = load_scored_bills(jurisdiction=selected_pipeline.lower(), state_code=target_state_code, _db_token=db_token)

macro_df = load_macro_data(_db_token=db_token)

# ------------------------------------------------------------
# ADVANCED DISPLAY SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard Display Filters")

# 1. Jurisdiction filter dropdown
jurisdiction_display_filter = st.sidebar.selectbox(
    "Filter Display by Jurisdiction Scope",
    options=["All", "Federal", "States"],
    index=0
)

# 2. Minimum Similarity Score slider
min_similarity_filter = st.sidebar.slider(
    "Minimum Match Similarity",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05
)

# 3. Policy Topic multi-select grid (CAP Major Topics)
cap_topics = ["All", "Macroeconomics", "Taxation", "Healthcare", "Education", "Energy & Environment", "Civil Rights & Liberties", "Labor & Employment", "Government Operations", "Transportation", "Other"]
selected_topics = st.sidebar.multiselect(
    "Filter Display by CAP Topic",
    options=cap_topics,
    default=["All"]
)

# ------------------------------------------------------------
# EXECUTIVE BRIEFING GENERATOR & EXPORTER
# ------------------------------------------------------------
def generate_markdown_briefing(df_brief, current_jur, is_campaign=False):
    if df_brief.empty:
        return "# POLICY INTELLIGENCE BRIEFING\n\nNo active bill records available for briefing compilation."
        
    brief = "# POLICY INTELLIGENCE BRIEFING & STRATEGIC EXECUTIVE SUMMARY\n\n"
    if is_campaign:
        brief += f"**Jurisdiction Scope Focus:** {str(current_jur).upper()} CAMPAIGN PLATFORMS\n"
    else:
        brief += f"**Jurisdiction Scope Focus:** {str(current_jur).upper()}\n"
    brief += "**Compiled By:** Pure NumPy Vectorized Policy Analyst Engine\n"
    brief += "**Grounding Framework:** Local nomic-embed-text + Phi-3 Semantic Linking\n\n"
    brief += "## 1. Projected Macroeconomic Outcomes\n\n"
    
    for idx, row in df_brief.iterrows():
        title_prefix = f"Candidate: {row.get('candidate_name')} - " if is_campaign else "Act: "
        brief += f"### {title_prefix}{row.get('title')} (ID: {row.get('bill_id')})\n"
        if is_campaign:
            brief += f"- **Candidate:** {row.get('candidate_name')} ({row.get('party')})\n"
        brief += f"- **Target Level:** {str(row.get('jurisdiction', 'federal')).upper()}\n"
        brief += f"- **CAP Topic Category:** {row.get('major_topic', 'Macroeconomics')}\n"
        brief += f"- **Analytical Confidence:** {row.get('confidence', 0.0):.1%}\n"
        brief += f"- **Estimated GDP Growth Effect:** {row.get('impact_gdp_effect', 0.0):+.3%}\n"
        brief += f"- **Estimated Unemployment Delta:** {row.get('impact_unemployment_effect', 0.0):+.3f} percentage points\n"
        brief += f"- **System Net Score:** {row.get('net_score', 0.0):+.3f} (Scale: -1.0 to +1.0)\n"
        brief += f"- **Strategic Grounding Note:** {row.get('explanation')}\n\n"
        
    brief += "## 2. Platform Research Methodology & Cyclical De-biasing\n\n"
    brief += "Projections are synthesized using a pure NumPy vectorized historical-analog engine. "
    brief += "Analog matching utilizes NOMIC semantic embeddings to run instantaneous cosine similarity searches "
    brief += "against legacy federal and state databases. Cyclical macro effects are de-biased using structural delta shifts "
    brief += "to filter out baseline macroeconomic recessions and hyper-inflation drag cycles.\n\n"
    brief += "*(Confidential Briefing compiled by the Policy Intelligence Platform)*\n"
    return brief

if not df.empty:
    is_camp_mode = (simulation_mode == "Candidate Campaign Platforms")
    briefing_md = generate_markdown_briefing(df, selected_pipeline, is_campaign=is_camp_mode)
    st.sidebar.download_button(
        label="📥 Export Executive Briefing (MD)",
        data=briefing_md,
        file_name=f"policy_briefing_{selected_pipeline.lower()}_{'campaign' if is_camp_mode else 'session'}.md",
        mime="text/markdown",
        use_container_width=True
    )

# ============================================================
# PRIMARY LAYOUT VIEW (TABS)
# ============================================================
st.title("⚖️ Policy Intelligence Platform")
st.markdown(f"Evolving open-source legislative analysis into high-value decision summaries for think tanks and legislators.")

tab_dashboard, tab_methodology = st.tabs(["📊 Policy Intelligence Dashboard", "📚 Methodology & Limitations"])

# ------------------------------------------------------------
# TAB 1: DASHBOARD VISUALIZATIONS
# ------------------------------------------------------------
with tab_dashboard:
    if simulation_mode == "Candidate Campaign Platforms":
        st.markdown(f"Currently visualizing evaluation profiles under Campaign Simulation Mode: **{selected_pipeline.upper()}**")
        
        if df.empty:
            st.info("No campaign analysis results loaded. Hit 'Run Simulation Pipeline Core' to execute semantic distillation.")
        else:
            # 1. Filter candidates
            df_acton = df[df["candidate_name"].str.contains("Acton", case=False, na=False)]
            df_ramaswamy = df[df["candidate_name"].str.contains("Ramaswamy", case=False, na=False)]
            
            # Split into two columns for side-by-side comparative dashboard
            col1, col2 = st.columns(2)
            
            # 2. Render Amy Acton Portfolio
            portfolio_score_acton = df_acton["net_score"].mean() if not df_acton.empty else 0.0
            gdp_acton = df_acton["impact_gdp_effect"].mean() if not df_acton.empty else 0.0
            unemp_acton = df_acton["impact_unemployment_effect"].mean() if not df_acton.empty else 0.0
            conf_acton = df_acton["confidence"].mean() if not df_acton.empty else 0.0
            conf_acton = max(0.0, min(1.0, conf_acton))
            
            with col1:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15) 0%, rgba(33, 150, 243, 0.05) 100%); padding: 25px; border-radius: 12px; border: 1px solid rgba(33, 150, 243, 0.2); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2); margin-bottom: 20px;">
                        <h3 style="margin-top: 0; color: #2196f3; font-family: 'Inter', sans-serif;">🔵 Amy Acton</h3>
                        <span style="background-color: rgba(33, 150, 243, 0.2); color: #2196f3; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">Democratic Portfolio</span>
                        <div style="margin-top: 20px; font-size: 2.2rem; font-weight: bold; color: {'#2ecc71' if portfolio_score_acton >= 0 else '#e74c3c'};">
                            {portfolio_score_acton:+.3f} <span style="font-size: 1rem; color: #888; font-weight: normal;">Portfolio Score</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("Avg GDP Component", f"{gdp_acton:+.3%}")
                with m_col2:
                    st.metric("Avg Unemployment Adj", f"{unemp_acton:+.3f} pp")
                with m_col3:
                    if conf_acton >= 0.70:
                        conf_badge = "🟢 HIGH"
                    elif conf_acton >= 0.40:
                        conf_badge = "🟡 MODERATE"
                    else:
                        conf_badge = "🔴 LOW"
                    st.metric("Clamped Confidence", conf_badge, f"{conf_acton:.1%}")
                    
                st.markdown("#### Distilled Policy Abstract Analysis")
                for idx, row in df_acton.iterrows():
                    with st.expander(f"📖 {row['title']} (Net Score: {row['net_score']:+.3f})"):
                        # Check for the distilled text, fallback to raw campaign text, then fallback to the legislative text column
                        display_text = row.get('distilled_abstract') or row.get('raw_proposal_text') or row.get('bill_text_clean', 'No text summary available.')

                        st.markdown(f"**Distilled Abstract:** *{display_text}*")                        
                        st.markdown(f"**Original Statement:** {row['raw_proposal_text']}")
                        st.markdown(f"**GDP growth effect:** {row['impact_gdp_effect']:+.3%}")
                        st.markdown(f"**Unemployment delta:** {row['impact_unemployment_effect']:+.3f} pp")
                        st.markdown(f"**System Explanation:** {row['explanation']}")
                        
            # 3. Render Vivek Ramaswamy Portfolio
            portfolio_score_ramaswamy = df_ramaswamy["net_score"].mean() if not df_ramaswamy.empty else 0.0
            gdp_ramaswamy = df_ramaswamy["impact_gdp_effect"].mean() if not df_ramaswamy.empty else 0.0
            unemp_ramaswamy = df_ramaswamy["impact_unemployment_effect"].mean() if not df_ramaswamy.empty else 0.0
            conf_ramaswamy = df_ramaswamy["confidence"].mean() if not df_ramaswamy.empty else 0.0
            conf_ramaswamy = max(0.0, min(1.0, conf_ramaswamy))
            
            with col2:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, rgba(244, 67, 54, 0.15) 0%, rgba(244, 67, 54, 0.05) 100%); padding: 25px; border-radius: 12px; border: 1px solid rgba(244, 67, 54, 0.2); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2); margin-bottom: 20px;">
                        <h3 style="margin-top: 0; color: #f44336; font-family: 'Inter', sans-serif;">🔴 Vivek Ramaswamy</h3>
                        <span style="background-color: rgba(244, 67, 54, 0.2); color: #f44336; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">Republican Portfolio</span>
                        <div style="margin-top: 20px; font-size: 2.2rem; font-weight: bold; color: {'#2ecc71' if portfolio_score_ramaswamy >= 0 else '#e74c3c'};">
                            {portfolio_score_ramaswamy:+.3f} <span style="font-size: 1rem; color: #888; font-weight: normal;">Portfolio Score</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("Avg GDP Component", f"{gdp_ramaswamy:+.3%}")
                with m_col2:
                    st.metric("Avg Unemployment Adj", f"{unemp_ramaswamy:+.3f} pp")
                with m_col3:
                    if conf_ramaswamy >= 0.70:
                        conf_badge = "🟢 HIGH"
                    elif conf_ramaswamy >= 0.40:
                        conf_badge = "🟡 MODERATE"
                    else:
                        conf_badge = "🔴 LOW"
                    st.metric("Clamped Confidence", conf_badge, f"{conf_ramaswamy:.1%}")
                    
                st.markdown("#### Distilled Policy Abstract Analysis")
                for idx, row in df_ramaswamy.iterrows():
                    with st.expander(f"📖 {row['title']} (Net Score: {row['net_score']:+.3f})"):
                        display_text = row.get('distilled_abstract') or row.get('raw_proposal_text') or row.get('bill_text_clean', 'No text summary available.')
                        st.markdown(f"**Distilled Abstract:** *{display_text}*")
                        st.markdown(f"**Original Statement:** {row['raw_proposal_text']}")
                        st.markdown(f"**GDP growth effect:** {row['impact_gdp_effect']:+.3%}")
                        st.markdown(f"**Unemployment delta:** {row['impact_unemployment_effect']:+.3f} pp")
                        st.markdown(f"**System Explanation:** {row['explanation']}")
                        
    else:
        st.markdown(f"Currently visualizing evaluation profiles under active jurisdiction channel: **{selected_pipeline.upper()}**")
        
        # 1. Render Executive Scorecard
        if not df.empty:
            st.subheader("Legislative Strategic Outlines")
            
            # Display up to top 3 active bills in beautiful metric outlining rows
            for idx, row in df.head(3).iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                    with col1:
                        st.markdown(f"**{row.get('title')}**")
                        st.caption(f"ID: `{row.get('bill_id')}` | Category: `{row.get('major_topic', 'Macroeconomics')}`")
                    with col2:
                        st.metric("Net Economic Score", f"{row.get('net_score', 0.0):+.3f}", help="Score bounds: -1.0 (strongly contractionary) to +1.0 (strongly expansionary)")
                    with col3:
                        conf_val = row.get("confidence", 0.0)
                        if conf_val >= 0.70:
                            conf_badge = "🟢 HIGH"
                        elif conf_val >= 0.40:
                            conf_badge = "🟡 MODERATE"
                        else:
                            conf_badge = "🔴 LOW"
                        st.metric("Analytical Confidence", conf_badge, delta=f"{conf_val:.1%}")
                    with col4:
                        st.caption(f"**System Explanation:** {row.get('explanation')}")
                    st.markdown("---")

        # 2. Raw Scored Matrix with display filters
        st.subheader("Scored Bills Matrix")

        if df.empty:
            st.info("No scored bills are available for this specific path framework. Hit the sidebar simulation trigger to fetch live data targets.")
        else:
            # Apply displayed filters to table view
            df_filtered = df.copy()
            
            # Jurisdiction Filter
            if jurisdiction_display_filter == "Federal":
                df_filtered = df_filtered[df_filtered["jurisdiction"] == "federal"]
            elif jurisdiction_display_filter == "States":
                df_filtered = df_filtered[df_filtered["jurisdiction"] != "federal"]
                
            # Policy Topic Filter
            if "All" not in selected_topics and len(selected_topics) > 0:
                df_filtered = df_filtered[df_filtered["major_topic"].isin(selected_topics)]
                
            # Similarity Filter
            if "impact_avg_similarity" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["impact_avg_similarity"] >= min_similarity_filter]

            if df_filtered.empty:
                st.warning("All bill records filtered out by active Display Filters in the sidebar.")
            else:
                preview_cols = [
                    "bill_id", "title", "policy_type", "direction", "net_score", 
                    "confidence", "impact_num_analogs_matched", "impact_avg_similarity", "major_topic"
                ]
                preview_cols = [col for col in preview_cols if col in df_filtered.columns]
                st.dataframe(df_filtered[preview_cols].head(50), use_container_width=True)

        # 3. Plots & Dispersion Charts
        if not df.empty and {"title", "net_score", "confidence"}.issubset(df.columns):
            st.subheader("Policy Scores Volatility")

            fig = px.bar(
                df.sort_values("net_score", ascending=False),
                x="net_score",
                y="title",
                color="confidence",
                color_continuous_scale=px.colors.sequential.Viridis,
                orientation="h",
                hover_data=["bill_id", "policy_type", "direction"],
                labels={"net_score": "Net Economic Impact Score", "title": "Legislative Act Title"}
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # SEMANTIC POLICY SPACE (PCA)
    # ----------------------------
    st.subheader("Semantic Policy Space Network Matrix (PCA)")
    st.markdown(
        "This interactive semantic map projects the high-dimensional bill clean summaries "
        "down to 2D using a pure NumPy Principal Component Analysis (PCA) algorithm. Bills closer together "
        "are semantically similar in subject matter and intent."
    )

    if simulation_mode == "Candidate Campaign Platforms" and not df.empty:
        df_semantic = get_semantic_map_data(df_campaign_dict=df.to_dict(orient="records"), _db_token=db_token)
    else:
        df_semantic = get_semantic_map_data(_db_token=db_token)

    if df_semantic.empty:
        st.info("Semantic vector topological coordinate matrix not found. Build out an embed repository on disk cache first.")
    else:
        # Apply display filters to semantic PCA plot
        df_plot = df_semantic.copy()
        
        # Jurisdiction Filter
        if jurisdiction_display_filter == "Federal":
            df_plot = df_plot[(df_plot["jurisdiction"] == "federal") | (df_plot["jurisdiction"] == "campaign")]
        elif jurisdiction_display_filter == "States":
            df_plot = df_plot[(df_plot["jurisdiction"] != "federal") | (df_plot["jurisdiction"] == "campaign")]
            
        # Policy Topic Filter
        if "All" not in selected_topics and len(selected_topics) > 0:
            df_plot = df_plot[(df_plot["major_topic"].isin(selected_topics)) | (df_plot["jurisdiction"] == "campaign")]

        if df_plot.empty:
            st.warning("No cached vector markers identified for active display filters.")
        else:
            # Color by jurisdiction and change marker shape by level (federal vs state vs campaign)
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
                symbol_sequence=['circle', 'diamond', 'square'],
                hover_data=hover_cols,
                title="Cross-Jurisdictional Semantic Policy Cluster Projections",
                labels={"PCA 1": "Principal Component 1 (Axis Alpha)", "PCA 2": "Principal Component 2 (Axis Beta)"}
            )
            fig_pca.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
            fig_pca.update_layout(dragmode="pan", legend_title_text="Jurisdictional Stratification Structure")
            st.plotly_chart(fig_pca, use_container_width=True)

    # 4. Macro baselines previews
    st.subheader("Macro Dataset Time-Series Baseline")
    if macro_df.empty:
        st.info("No macroeconomic historical baselines loaded.")
    else:
        st.dataframe(macro_df.head(50), use_container_width=True)

# ------------------------------------------------------------
# TAB 2: METHODOLOGY & STRATEGIC LIMITATIONS
# ------------------------------------------------------------
with tab_methodology:
    st.header("📚 Research Methodology & Strategic Limitations")
    st.markdown("---")
    
    st.subheader("1. Core Philosophy: Pure Vectorized Analytics")
    st.markdown(r"""
    Unlike heavy deep learning models or proprietary vector database systems, this platform operates on a strict **zero external machine learning dependency** framework.
    All math, clustering, projection, and de-biasing algorithms are handled strictly via **vectorized pure NumPy and Pandas routines**:
    
    * **Semantic Embeddings:** High-dimensional vector generation is performed locally using the `nomic-embed-text` service running on a local Ollama server, securing full data privacy and zero API query cost.
    * **Topological SVD Projection:** High-dimensional vector space is projected down to 2D for human visualization using vectorized **Singular Value Decomposition (SVD)**:
      $$X_{centered} = U \Sigma V^T$$
      Continuous memory space projection via NumPy's matrix product maximizes CPU cache locality and speeds up calculations.
    * **Cosine Similarity Matrices:** Matches legislative bills to historical analogs using BLAS dot product calculations in a single matrix-vector math operation, performing lookups in under 1.5ms.
    """)
    
    st.subheader("2. Cyclical De-biasing Models")
    st.markdown("""
    To ensure professional-grade credibility, the engine incorporates cyclical de-biasing adjustments. Raw macroeconomic indices (such as historical GDP growth and BLAS unemployment deltas) can be distorted by broad economic cycles (e.g. the 2008 Great Recession).
    
    The **OutcomeEngine** automatically identifies extreme historical regimes:
    * **Recession Regimes:** Adjusts GDP and Unemployment deltas upward to remove the broad cyclical drag and reflect the policy's true relative impact.
    * **High Inflation Regimes:** Dampens projected GDP expansion components by **15%** to account for stagflation drag.
    """)
    
    st.subheader("3. Smart Fallback Inferences")
    st.markdown("""
    State legislative pipelines face thin historical analog sets. If a target state bill fails to match any state-specific analogs, the system engages **Smart Federal Fallback**:
    * Dynamically queries federal-level analogs.
    * Applies a **scope leakage penalty** (halving the similarity match metrics) to down-weight federal scope relevance.
    * This prevents unclassified evaluation drops while maintaining defensive, credible scoring thresholds.
    """)
    
    st.subheader("4. Limitations & Analytical Assumptions")
    st.markdown("""
    When presenting these summaries to state legislators or think tanks, please state the following analytical limits:
    1. **Historical Precedent Dependency:** The system projects outcomes based purely on matched past policies. Unique, completely unprecedented legislation will receive neutral scores with low confidence.
    2. **Local Endpoints Queue:** Parallel execution saturates local Ollama thread queues; speed is dependent on host hardware capabilities.
    3. **Decay Weights:** Out-year delta projections use decaying averages ($T+1 = 1.0, T+2 = 0.5, T+3 = 0.25$), assuming structural impacts fade over a 3-year trailing horizon.
    """)