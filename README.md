# Intelipol - Policy Intelligence Platform

Policy Explorer is a modular, deterministic decision-support dashboard and data ingestion pipeline designed to model and project the macroeconomic impacts of proposed U.S. Congressional bills. 

By analyzing semantic similarities between new legislation and a deep historical corpus, the platform maps policy intents directly to documented real-world economic outcomes (specifically GDP and unemployment rates) over a 3-year trailing horizon.

---

## 🛡️ Architectural Guardrail: Zero External ML Libraries

A primary engineering constraint of this platform is **strict modular decoupling and zero external machine learning package dependencies**. 
*   **No** PyTorch, TensorFlow, scikit-learn, or scipy.
*   **No** external SaaS vector databases or closed-source LLM APIs.
*   **All analytical math**—including cosine similarity, multi-year decay-weighted aggregations, linear projection confidence scaling, covariance-based Principal Component Analysis (PCA), and leave-one-out cross-validation—is implemented in **pure Python, Pandas, and NumPy**.
*   **Local air-gapped intelligence** is achieved via a local instance of **Ollama** running `llama3` for coarse characteristic classification and `nomic-embed-text` for generating 768-dimensional float embedding arrays.

---

## ⚙️ Core System Architecture

The platform is organized into four independent, decoupled engines:

```mermaid
graph TD
    subgraph Data Sources
        A1[BEA API / Parquet: GDP] & A2[BLS API / Parquet: Unemployment] --> A3[data/processed/policy_dataset.parquet]
        B1[data/policy_events.csv] --> B2[data/processed/historical_bills.parquet]
        C1[US Congress API] --> C2[Current Bills Raw Data]
    end

    subgraph Core Pipeline Ingest & Processing
        C2 --> D[AI Classification: Ollama llama3]
        D -->|Metadata & Macro Relevance| E{Macro Relevant?}
        
        E -->|No| F[Score = 0.0, Confidence = 0.0]
        
        E -->|Yes| G[AnalogMatcher: pipeline/policy_matching.py]
        B2 -->|Read Cached Embeddings| G
        
        G -->|Top K Analog Bills & Similarity| H[OutcomeEngine: pipeline/policy_impact_linker.py]
        A3 -->|Lookup Macro Values: Year T to T+3| H
        
        H -->|Decay-Weighted Multi-Year Deltas| I[ScoringEngine: pipeline/policy_score.py]
        
        I -->|Clamped Net Score [-1.0, 1.0] & Confidence [0.0, 1.0]| J[Scored Results List]
    end

    subgraph User Dashboard
        J --> K[Streamlit App: app.py]
        J --> L[Unified Output JSON]
    end
```

### 1. `AnalogMatcher` (`pipeline/policy_matching.py`)
*   **Purpose:** Matches newly ingested bills to historical analogs using semantic embedding vectors.
*   **Mechanism:** Loads title embedding vectors in $O(1)$ from a persistent cache. Calculates the **Cosine Similarity** between the target title's vector and all historical vectors, keeping matches that meet a custom threshold (default `min_threshold = 0.75`).

### 2. `OutcomeEngine` (`pipeline/policy_impact_linker.py`)
*   **Purpose:** Evaluates macroeconomic changes associated with historical analog bills.
*   **Mechanism:** Looks up the historical state/national macro data for each analog bill during its introduction/enactment year $T$ and over a trailing 3-year window ($T+1, T+2, T+3$). It aggregates outcomes across all matched analogs using similarity-score-weighted averages.
*   **Scope Safeguard:** Applies a `0.5x` penalty to similarity scores when comparing state-level analogs against federal-level targets to prevent national scope leakage.

### 3. `ScoringEngine` (`pipeline/policy_score.py`)
*   **Purpose:** Translates macroeconomic impacts into a bounded net score and determines a confidence metric.
*   **Scoring Weight Formula:**
    $$\text{net\_score} = \text{clamp}((\text{GDP effect} \times 0.4) + (\text{Unemployment effect} \times -0.3), -1.0, 1.0)$$
*   **Confidence Calculation:**
    $$\text{confidence} = \text{clamp}(\text{avg\_similarity} \times \min(1.0, \frac{\text{num\_analogs}}{5.0}), 0.0, 1.0)$$
    *(If only 1 analog is matched, confidence is penalized by $0.5x$ to prevent over-indexing on single data points).*

### 4. `Master Orchestrator` (`pipeline/build_dataset.py`)
*   **Purpose:** Coordinates live data ingestion from the **US Congress API**, runs LLM classification via **Ollama**, routes data through the three core engines, and outputs a unified JSON schema.

---

## ⚡ Recent Production Upgrades

### 1. Multi-Year Trailing Horizon Window
Instead of evaluating only $T$ vs $T+1$, the `OutcomeEngine` tracks trailing macro trends across years $T+1, T+2$, and $T+3$ relative to Base Year $T$. Projections are weighted with a linear time-decay to prioritize near-term policy effects:
*   **Year T+1 Weight:** $1.0$
*   **Year T+2 Weight:** $0.5$
*   **Year T+3 Weight:** $0.25$

### 2. Persistent Parquet Vector Caching
At startup, the `AnalogMatcher` reads cached embeddings directly from `data/processed/historical_embeddings.parquet` in a single disk read operation. The engine only requests embeddings from the Ollama API for newly added bills, updating the parquet cache dynamically upon completion.

### 3. Pure NumPy PCA Semantic Map
We added an interactive, two-dimensional **Semantic Policy Space Map** in Streamlit. The projection of 768-dimensional title vectors down to 2D coordinates is achieved via a custom Singular Value Decomposition (Covariance Eigen-decomposition) implemented purely in NumPy:
```python
def calculate_cheap_pca(vectors_matrix, num_components=2):
    X_centered = vectors_matrix - np.mean(vectors_matrix, axis=0)
    covariance_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_vectors = eigenvectors[:, sorted_indices[:num_components]]
    return np.dot(X_centered, top_vectors)
```

### 4. Cross-Validation Backtesting Infrastructure
A dedicated validation framework in `scripts/backtest.py` performs a **"leave-one-out" cross-validation (LOO-CV)** test:
1. Selects a historical bill from the corpus.
2. Temporarily removes it from the historical analog pool.
3. Passes it through the pipeline as a new "test bill".
4. Compares the engine's projected macro effects to the actual historical outcome that occurred.
5. Calculates and reports the **Mean Absolute Error (MAE)**.

---

## 📊 Backtesting Grounding Metrics

Running the LOO-CV backtest on historical corpus samples yields strong predictive grounding:
*   **GDP Growth Effect MAE:** **`0.016052`** (Average absolute error of **1.6%** in predicting real-world GDP percentage swings)
*   **Unemployment Delta Effect MAE:** **`0.385720`** (Average absolute error of **0.38** points on unemployment swings)

---

## 🛠️ Setup & Operations

### 1. Prerequisites
Ensure python 3.8+ is active, and install the lightweight requirements:
```bash
pip install pandas numpy pyarrow requests streamlit plotly openpyxl
```

Ensure a local instance of **Ollama** is running on your host machine (`localhost:11434`):
```bash
# Start Ollama service
ollama serve

# Pull models required for live classification and embedding
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Ingesting & Running the Live Pipeline
To ingest new bills from the U.S. Congress API, classify them, score them against historical analogs, and save results:
```bash
python3 pipeline/build_dataset.py
```

### 3. Running the Validation Backtest
To execute the leave-one-out cross-validation script and review predictive error statistics:
```bash
# Run validation on a sample of 15 bills
python3 scripts/backtest.py 15
```

### 4. Launching the Dashboard
To launch the Streamlit dashboard and view live projections, historical economic trends, and the interactive **NumPy PCA Semantic Policy Space Map**:
```bash
streamlit run app.py
```

