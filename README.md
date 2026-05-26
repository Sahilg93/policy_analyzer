# Intelipol - Policy Intelligence Platform

Intelipol is a highly modular, deterministic decision-support dashboard and data ingestion pipeline designed to model and project the macroeconomic impacts of proposed U.S. Congressional and state-level legislative bills, as well as political candidate platforms.

By mapping semantic similarities between new legislation and a deep historical policy events corpus, the platform simulates policy transmission channels and projects real-world economic outcomes (specifically GDP and unemployment rates) over a 3-year trailing horizon.

---

## 🛡️ Architectural Guardrail: Zero External ML Libraries

A primary engineering constraint of Intelipol is **strict modular decoupling and zero external machine learning package dependencies**.
*   **No** PyTorch, TensorFlow, scikit-learn, or SciPy.
*   **No** external SaaS vector databases or closed-source LLM APIs.
*   **All analytical math**—including cosine similarity, Min-Max coordinate stretching, multi-year decay-weighted category aggregations, standard Z-score moments, inverse Z-score transforms, covariance-based Principal Component Analysis (PCA), leave-one-out cross-validation, and SVD coordinate sign-locking—is implemented in **pure Python, Pandas, and NumPy**.
*   **Local air-gapped intelligence** is achieved via a local instance of **Ollama** running `llama3` for coarse characteristic classification and `nomic-embed-text` for generating 768-dimensional float embedding arrays.

---

## ⚙️ Core System Architecture

The platform is organized into decoupled, highly specialized modules:

```mermaid
graph TD
    subgraph Data Layer
        A1[BEA API / Parquet: GDP] & A2[BLS API / Parquet: Unemployment] --> A3[data/processed/policy_dataset.parquet]
        B1[data/policy_events.csv] --> B2[data/processed/historical_bills.parquet]
        C1[US Congress API / OpenStates API] --> C2[Raw Legislative Bills]
        D1[data/campaign_policies.csv] --> D2[Candidate Campaign Proposals]
    end

    subgraph Semantic Ingestion & Rate-Limiting Shield
        C2 & D2 --> E[OpenStatesClient: Rate-Limiting Bucket & Backoff Shield]
        E --> F[Rhetoric Distillation: Ollama + zero-shot economist re-prompting]
    end

    subgraph Analytical Core Processing
        F --> G[AnalogMatcher: Min-Max Scaled Cosine Similarity]
        B2 -->|Read Cached Embeddings| G
        
        G -->|Top K Analog Bills & Similarity| H[OutcomeEngine: 3-Year Trailing Decay-Weighted Window]
        A3 -->|Lookup Macro Values: Year T to T+3 & Fallbacks| H
        
        H -->|Multi-Year Deltas| I[ScoringEngine: Z-Score Standardization & Inverse Transform]
        
        I -->|Realistic Fractional Empirical Deltas| J[Hierarchical Override Gate: Priority Topics]
        J -->|Clamped Net Score [-1.0, 1.0]| K[Unified Scored Results Frame]
    end

    subgraph Interactive Interface
        K --> L[Streamlit App: app.py]
        K --> M[Persistent data/processed/scored_bills.csv]
    end
```

---

## ⚡ Production & Core Upgrades

### 1. Robust Rate-Limiting Shield (`pipeline/openstates_client.py`)
To safeguard legislative fetches under Plural Policy’s 30 Requests-Per-Minute (RPM) ceiling, the client integrates:
*   **Thread-Safe Token Bucket Limiter:** Implements a lock-protected token replenishment structure (`max_capacity = 5`, refilling 1 token every 2.0 seconds) to ensure multi-threaded safety.
*   **Exponential Backoff with Random Jitter:** Automatically wraps all HTTP requests in a defensive 5-attempt retry loop. If a 429 status code is encountered, it parses the `Retry-After` header or calculates a randomized exponential sleep window.
*   **Graceful Boundary Preservation:** In the event of persistent connection errors, the engine catches failures and returns partial, schema-safe datasets rather than triggering system crashes.

### 2. Rigid LLM Preamble Pruning (`pipeline/campaign_adapter.py`)
To prevent conversational prompt pollution on the frontend:
*   **Conversational Header Stripping:** Automatically matches and strips preambles such as *"Here is a"*, *"2-sentence dry summary"*, *"Abstract:"*, or *"The proposed policy"* using dynamic partition string splitting (`.split(":")[-1]`) or explicit regex pruning.
*   **Zero-Shot Economist Re-prompting:** If the distilled abstract matches the original statement exactly or is too long (echoing), the adapter halts execution and re-prompts the local LLM model under strict economist constraints: *"Output ONLY the dry administrative mechanism. Do not echo the original text."* It recursively cleans the new response with the re-prompt gate safely bypassed to prevent infinite loops.

### 3. Z-Score Inverse Transform Denormalization (`pipeline/policy_score.py`)
To display expected real-world macroeconomic changes instead of standard dimensionless scores:
*   **Inverse Z-Score Layer:** Computes true baseline economic percentage adjustments:
    $$\text{true\_gdp\_delta} = ((z_{\text{gdp}} \times \sigma_{\text{gdp}}) + \mu_{\text{gdp}}) \times \text{policy\_scale}$$
    $$\text{true\_unemp\_delta} = ((z_{\text{unemp}} \times \sigma_{\text{unemp}}) + \mu_{\text{unemp}}) \times \text{policy\_scale}$$
*   **Contribution Scale Tuning:** Sets `policy_scale = 0.01` to restore projected outcomes to fractional percentages (e.g., $+0.086\%$ GDP change instead of $+8.6\%$ spikes).
*   **Unemployment Variation Fallbacks:** Handles first differences (`.diff()`) on labor arrays. Introduces a historical federal baseline map (`US_HISTORICAL_UNEMP`) and a national proxy fallback to successfully resolve missing state-level unemployment series.

### 4. Hierarchical Override Gate (`pipeline/build_dataset.py`)
*   To prevent unclassified macroeconomic score drops, a deterministic override gate evaluates major policy topics (e.g., *Macroeconomics, Taxation, Labor, Domestic Commerce*). If a bill belongs to these core areas, the engine overrides the LLM and enforces `macro_relevance = True` before routing it to the analog matching core.

### 5. Streamlit Cache-Busting & Reactive State Updates (`app.py`)
*   **Reactive State Invalidation:** Tracks file modification times of historical, campaign, and newly generated scored datasets via `_get_db_token()`. This token is passed directly to `@st.cache_data` decorators, triggering instant, flicker-free dashboard card and plot updates the moment new data is written to disk.
*   **PCA Sign-Locking (`svd_flip`):** Standardizes coordinate projection signs on the SVD Semantic Map, preventing the interactive visualization from flipping or mirroring during state updates.

---

## 📊 Cross-Validation Backtesting Results

Predictive grounding is validated via a **"leave-one-out" cross-validation (LOO-CV)** backtesting suite inside `scripts/backtest.py`. Under rigorous testing, the engines yield high alignment:
*   **GDP Growth Effect MAE:** **`0.077087`** (Average absolute error of **0.07%** in predicting real-world GDP growth changes)
*   **Unemployment Delta Effect MAE:** **`0.156471`** (Average absolute error of **0.15** points on unemployment adjustments)

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Intelipol requires python 3.8+ and lightweight packages:
```bash
pip install pandas numpy pyarrow requests streamlit plotly openpyxl
```

Ensure a local instance of **Ollama** is running on your host machine (`localhost:11434`):
```bash
# Start Ollama service
ollama serve

# Pull models required for live classification, embedding, and distillation
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Centralized Ingestion & Pipeline Configuration
Copy the sample environment file to `.env` and fill in your API tokens (supports OpenStates, Congress.gov, BEA, and BLS):
```bash
cp .env.example .env
```

### 3. Ingesting & Running State Simulations
To run a complete legislative ingestion, AI classification, scoring, and persistent serialization cycle for a specific state:
```bash
# Run Ohio legislative session simulation
python3 pipeline/build_dataset.py --jurisdiction Ohio --state-code OH

# Force embedding rebuild from scratch
python3 pipeline/build_dataset.py --jurisdiction Ohio --state-code OH --rebuild-embeddings
```

### 4. Seeding State Corpora
To seed a high-density, multi-jurisdictional state historical corpus:
```bash
python3 scripts/seed_state_corpus.py
```

### 5. Running the Validation Backtest
To execute the leave-one-out cross-validation script and review predictive error statistics:
```bash
# Run validation on historical corpus
python3 scripts/backtest.py
```

### 6. Launching the Dashboard UI
To launch the premium Streamlit interactive dashboard:
```bash
streamlit run app.py
```
This serves the visual platform on `http://localhost:8501`. Toggle **"Candidate Campaign Platforms"** in the sidebar to review side-by-side comparative portfolios (e.g., Amy Acton vs. Vivek Ramaswamy for Ohio) complete with clean, pruned abstracts and realistic denormalized macro metrics.
