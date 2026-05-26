# Policy Explorer - U.S. State Legislatures Integration Guide (Revised)

This guide details the updated architectural blueprint, database schemas, code blocks, and integration instructions to expand the Policy Intelligence Platform for state legislative bills. It incorporates advanced bulk processing, rate limiting, CAP-style topic classification, localized macro indicators, and embedding generations over full-text clean summaries.

---

## 💾 1. Evolved Database Schemas

To support state legislatures alongside federal records, the primary Parquet data stores will be updated. The schemas are evolved by appending **additive columns**, allowing old federal records to default gracefully:

### A. `historical_bills.parquet` & `historical_embeddings.parquet`
For existing federal bills, new columns are auto-filled with default constants:

| Column Name | Data Type | Default for Federal Bills | Description |
| :--- | :---: | :---: | :--- |
| **`jurisdiction`** | `string` | `"federal"` | Name of state (e.g. `"california"`) or `"federal"` |
| **`state_code`** | `string` | `"US"` | Two-letter state abbreviation (e.g. `"CA"`, `"TX"`) |
| **`session_year`** | `int64` | *Extracted from Date* | The legislative session year (extracted from date field) |
| **`enacted`** | `boolean` | `true` | Boolean flag indicating if the bill successfully became law |
| **`sponsor_party`** | `string` | `"mixed"` | Primary sponsor's political party affiliation |
| **`bill_text_clean`**| `string` | *Populated from Title* | **[NEW]** Pre-processed full text or clean bill summary (used for embeddings) |
| **`major_topic`** | `string` | `"macroeconomics"` | **[NEW]** CAP-style topic classification (e.g. *macroeconomics, tax, health*) |

### B. `policy_dataset.parquet` (Macro indicators database)
Maps rows under the `state` column (e.g. `"California"`, `"Texas"`, etc.) alongside `"United States"` for federal indicators:
- **Columns:** `['year', 'state', 'gdp', 'unemployment_rate', 'gdp_growth']`

---

## 📂 2. Evolved File Structure

No existing files need to be deleted. The expansion introduces a new API client, a new macro dataset ingestion script, and selective backward-compatible updates to the core engines:

```
policy-explorer/
├── data/
│   ├── policy_events.csv              # Seed historical events (unaltered)
│   ├── state_gdp_clean.csv            # Clean regional state GDP data (unaltered)
│   └── processed/
│       ├── historical_bills.parquet   # Expanded with state metadata + summaries + topics
│       ├── historical_embeddings.parquet # Updated to track state vector mappings
│       └── policy_dataset.parquet     # Integrated with state-level BEA/BLS macro paths
├── pipeline/
│   ├── __init__.py
│   ├── openstates_client.py           # [NEW] Bulk Open States API client with pagination & rate-limiting
│   ├── policy_matching.py             # [MODIFIED] Supports cross-jurisdiction matching penalties
│   ├── policy_impact_linker.py        # [MODIFIED] Resolves regional GDP/unemployment outcomes
│   ├── policy_score.py                # Bounded scoring engine (unaltered)
│   └── build_dataset.py               # [MODIFIED] Live pipeline router fetching federal/state bills
├── scripts/
│   ├── backtest.py                    # Backtesting framework (unaltered/compatible)
│   └── ingest_state_macro.py          # [NEW] Utility for fetching actual BEA/BLS state macro data
└── app.py                             # [MODIFIED] Streamlit visualization with jurisdiction selectors
```

---

## 💻 3. Key Code Implementations

### Deliverable A: Bulk & Pagination-Ready Open States API v3 Client (`pipeline/openstates_client.py`)
This production-ready client is built with **exponential backoff retry mechanisms**, bulk pagination handling, rate-limit resilience, and text clean-up routines to build the critical `bill_text_clean` field:

```python
"""
Open States API v3 Client
Fetches legislative bills with pagination, rate-limiting, and clean summaries.
"""
import time
import requests
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OpenStatesClient:
    def __init__(self, api_key: str, max_retries: int = 5, backoff_factor: float = 1.5):
        self.api_key = api_key
        self.base_url = "https://v3.openstates.org"
        self.headers = {"X-API-KEY": self.api_key}
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _request_with_retry(self, url: str, params: Dict) -> Optional[Dict]:
        """Performs GET requests with exponential backoff rate-limiting protection."""
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                r = requests.get(url, headers=self.headers, params=params, timeout=15)
                if r.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited (429). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(delay)
                delay *= self.backoff_factor
        return None

    def fetch_state_bills_bulk(self, state_code: str, max_bills: int = 200) -> pd.DataFrame:
        """
        Fetches state legislative bills utilizing paginated requests with rate-limiting resilience.
        """
        st_lower = state_code.strip().lower()
        jurisdiction = f"ocd-jurisdiction/country:us/state:{st_lower}/government"
        endpoint = f"{self.base_url}/bills"
        
        bills = []
        page = 1
        per_page = 50
        
        state_map = {
            "ca": "California", "tx": "Texas", "ny": "New York", "fl": "Florida",
            "oh": "Ohio", "il": "Illinois", "pa": "Pennsylvania", "mi": "Michigan"
        }
        state_name = state_map.get(st_lower, "United States")
        
        logger.info(f"Starting bulk ingest for {state_code.upper()} (Max: {max_bills} bills)...")
        
        while len(bills) < max_bills:
            params = {
                "jurisdiction": jurisdiction,
                "per_page": per_page,
                "page": page,
                "include": ["sponsorships", "actions"]
            }
            
            data = self._request_with_retry(endpoint, params=params)
            if not data:
                break
                
            results = data.get("results", [])
            if not results:
                break  # No more pages available
                
            for b in results:
                if len(bills) >= max_bills:
                    break
                    
                # 1. Sponsor Party
                sponsorships = b.get("sponsorships", [])
                sponsor_party = "unknown"
                for sp in sponsorships:
                    if sp.get("primary"):
                        sponsor_party = sp.get("party", "unknown")
                        break
                
                # 2. Enactment Status
                enacted = False
                for action in b.get("actions", []):
                    classifications = action.get("classification", [])
                    if "executive-signature" in classifications:
                        enacted = True
                        break
                
                # 3. Clean Text Summary and Process
                title = b.get("title", "").strip()
                abstract = b.get("abstract", "") or ""
                bill_text_clean = f"{title} | {abstract}".strip() if abstract else title
                
                introduced_date = b.get("first_action_date") or b.get("created_at")
                session_year = int(introduced_date[:4]) if introduced_date else None
                
                bills.append({
                    "bill_id": b.get("identifier", f"state_{b.get('id')[-8:]}"),
                    "title": title,
                    "introduced_date": introduced_date,
                    "enacted_date": b.get("latest_action_date") if enacted else None,
                    "policy_type": "unknown",
                    "direction": "neutral",
                    "intensity": "low",
                    "sector": "mixed",
                    "state": state_name,
                    "level": "state",
                    "text": bill_text_clean,
                    "jurisdiction": st_lower,
                    "state_code": state_code.upper(),
                    "session_year": session_year,
                    "enacted": enacted,
                    "sponsor_party": sponsor_party,
                    "bill_text_clean": bill_text_clean,
                    "major_topic": "unknown"
                })
                
            logger.info(f"Retrieved page {page} | Total bills: {len(bills)}")
            page += 1
            time.sleep(0.5)  # General rate-limiting grace delay
            
        logger.info(f"Bulk Ingest Completed. Total records: {len(bills)}")
        return pd.DataFrame(bills)
```

---

### Deliverable B: Hybrid Similarity Upgrades (`pipeline/policy_matching.py`)
Upgrades similarity lookups to use `bill_text_clean` for embedding matches and apply jurisdiction penalties:

```python
    def find_similar_bills(
        self,
        target_policy: Dict,
        min_threshold: float = 0.75,
        state_to_state_penalty: float = 0.15,
        fed_to_state_penalty: float = 0.30
    ) -> List[Dict]:
        """
        Matches bills using full-text clean summaries, applying jurisdiction mismatch penalties.
        """
        if self.history is None or self.history.empty:
            logger.warning("No historical bills available")
            return []

        if not self.embedding_cache:
            logger.warning("No historical embeddings available")
            return []
        
        # Use clean bill text for high-fidelity embedding, fallback to title
        target_text = str(target_policy.get("bill_text_clean", "") or target_policy.get("title", "")).strip()
        target_jur = str(target_policy.get("jurisdiction", "federal")).strip().lower()
        
        logger.info(
            "Searching analogs (Target Jur: %s) above %.2f: %s",
            target_jur.upper(),
            min_threshold,
            target_text[:50]
        )

        target_vector = self._get_embedding(target_text)
        if target_vector is None:
            return []
        
        threshold = max(0.0, min(1.0, min_threshold))
        analogs = []
        
        for idx, hist_bill in self.history.iterrows():
            bill_id = hist_bill.get("bill_id", f"hist_{idx}")
            hist_vector = self.embedding_cache.get(bill_id)
            if hist_vector is None:
                continue

            sim = self._cosine_similarity(target_vector, hist_vector)
            if sim is None:
                continue
                
            # Apply jurisdiction penalties
            analog_jur = str(hist_bill.get("jurisdiction", "federal")).strip().lower()
            if target_jur != analog_jur:
                if target_jur == "federal" or analog_jur == "federal":
                    sim -= fed_to_state_penalty
                else:
                    sim -= state_to_state_penalty
            
            sim = max(0.0, min(1.0, sim))
            if sim < threshold:
                continue
            
            analogs.append({
                "bill_id": bill_id,
                "title": hist_bill.get("title", ""),
                "introduced_date": hist_bill.get("introduced_date"),
                "enacted_date": hist_bill.get("enacted_date"),
                "policy_type": hist_bill.get("policy_type", "unknown"),
                "direction": hist_bill.get("direction", "neutral"),
                "intensity": hist_bill.get("intensity", "low"),
                "sector": hist_bill.get("sector", "mixed"),
                "similarity_score": sim,
                "state": hist_bill.get("state", "United States"),
                "level": hist_bill.get("level", "federal"),
                "jurisdiction": analog_jur
            })
        
        analogs.sort(key=lambda analog: analog["similarity_score"], reverse=True)
        return analogs
```

---

### Deliverable C: Localized Topic & CAP Classification (`pipeline/build_dataset.py`)
To map state bills into consistent Comparative Agendas Project (CAP) topics (e.g. *Macroeconomics, Taxation, Healthcare, Education*), the live pipeline orchestrator utilizes local Ollama prompts inside `build_dataset.py`:

```python
def classify_major_topic_ollama(title: str, clean_text: str) -> str:
    """
    Local LLM-based CAP-style major topic classifier.
    Categorizes the legislative bill into standard CAP policy classifications.
    """
    prompt = f"""
    You are a policy classification system.
    Analyze the legislative bill details and return ONLY the matching major category.
    
    Title: {title}
    Text/Summary: {clean_text}
    
    Choose ONLY from the following categories:
    - Macroeconomics
    - Taxation
    - Healthcare
    - Education
    - Energy & Environment
    - Civil Rights & Liberties
    - Labor & Employment
    - Government Operations
    - Transportation
    - Other
    
    Category:
    """
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=20)
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
        
        # Match against our allowed CAP topics
        allowed = ["Macroeconomics", "Taxation", "Healthcare", "Education", "Energy & Environment", 
                   "Civil Rights & Liberties", "Labor & Employment", "Government Operations", "Transportation"]
        for topic in allowed:
            if topic.lower() in raw_text.lower():
                return topic
        return "Other"
    except Exception:
        return "Other"
```

---

## 📈 4. Fetching Real Macro Data (`scripts/ingest_state_macro.py`)

Rather than relying on mock files, this script fetches real regional macroeconomic variables directly:
1.  **State GDP (BEA Regional API):** Calls the Bureau of Economic Analysis API.
2.  **State Unemployment Rates (BLS LAUS API):** Queries the Local Area Unemployment Statistics (LAUS) database.

```python
"""
State Macroeconomic Data Ingestor
Downloads real GDP and Unemployment indicators directly from BEA and BLS APIs,
and compiles them into data/processed/policy_dataset.parquet.
"""
import os
import requests
import pandas as pd
import numpy as np

BEA_API_KEY = "84DF9CAA-34FB-4555-BDF0-130FEA791DA2"
BLS_API_KEY = "71ca07a939aa4e71a82ae2f88ac8ad1e"

def fetch_real_state_gdp(api_key: str) -> pd.DataFrame:
    """Fetch real state-level GDP (Annual) from the BEA Regional GDP API."""
    url = "https://apps.bea.gov/api/data"
    params = {
        "UserID": api_key,
        "Method": "GetData",
        "DataSetName": "Regional",
        "TableName": "SAGDP2",  # Annual GDP by State
        "LineCode": "1",        # All industries total
        "GeoFips": "STATE",     # All states
        "Year": "2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025",
        "ResultFormat": "json"
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        results = data.get("BEAAPI", {}).get("Results", {}).get("Data", [])
        rows = []
        for d in results:
            rows.append({
                "year": int(d.get("TimePeriod")),
                "state": d.get("GeoName"),
                "gdp": float(d.get("DataValue").replace(",", "")) if d.get("DataValue") else None
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error fetching BEA GDP: {e}")
        return pd.DataFrame()

def fetch_real_state_unemployment(api_key: str, state_fips_map: dict) -> pd.DataFrame:
    """Fetch real state unemployment rate histories from BLS LAUS API."""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-Type": "application/json"}
    
    # Compile BLS LAUS timeseries IDs: LASST<FIPS>00000000003 (Seasonally Adjusted Rate)
    series_ids = []
    id_to_state = {}
    for fips, state in state_fips_map.items():
        series_id = f"LASST{fips:02d}00000000003"
        series_ids.append(series_id)
        id_to_state[series_id] = state
        
    payload = {
        "seriesid": series_ids,
        "startyear": "2010",
        "endyear": "2025",
        "registrationkey": api_key
    }
    
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        rows = []
        for series in data.get("Results", {}).get("series", []):
            series_id = series.get("seriesID")
            state = id_to_state.get(series_id)
            for d in series.get("data", []):
                year = int(d.get("year"))
                # Unemployment Rate
                rows.append({
                    "year": year,
                    "state": state,
                    "unemployment_rate": float(d.get("value"))
                })
        # BLS provides monthly data; aggregate to annual averages
        df = pd.DataFrame(rows)
        if not df.empty:
            return df.groupby(["year", "state"])["unemployment_rate"].mean().reset_index()
        return df
    except Exception as e:
        print(f"Error fetching BLS Unemployment: {e}")
        return pd.DataFrame()

def integrate_real_macro_data():
    """Compiles BEA state GDP and BLS state unemployment data into the Parquet store."""
    parquet_path = "data/processed/policy_dataset.parquet"
    
    state_fips = {
        6: "California", 48: "Texas", 36: "New York", 12: "Florida",
        39: "Ohio", 17: "Illinois", 42: "Pennsylvania", 26: "Michigan"
    }
    
    print("Downloading live state GDP from BEA...")
    df_gdp = fetch_real_state_gdp(BEA_API_KEY)
    
    print("Downloading live state Unemployment from BLS...")
    df_unemp = fetch_real_state_unemployment(BLS_API_KEY, state_fips)
    
    if df_gdp.empty or df_unemp.empty:
        print("Data download failed. Check API credentials or network connections.")
        return
        
    # Merge GDP and Unemployment
    df_merged = pd.merge(df_gdp, df_unemp, on=["year", "state"], how="outer")
    
    # Integrate into policy_dataset.parquet
    if os.path.exists(parquet_path):
        df_old = pd.read_parquet(parquet_path)
        # Exclude matching state records in old DB to prevent duplicates
        states_to_exclude = list(state_fips.values())
        df_old_clean = df_old[~df_old["state"].isin(states_to_exclude)]
        df_final = pd.concat([df_old_clean, df_merged], ignore_index=True)
    else:
        df_final = df_merged
        
    # Re-calculate state-level GDP growth
    df_final = df_final.sort_values(by=["state", "year"]).reset_index(drop=True)
    for state in df_final["state"].unique():
        mask = df_final["state"] == state
        df_state = df_final[mask]
        gdp_prev = df_state["gdp"].shift(1)
        df_final.loc[mask, "gdp_growth"] = (df_state["gdp"] - gdp_prev) / gdp_prev
        
    df_final.to_parquet(parquet_path, index=False)
    print(f"Data Integration Complete! Updated database shape: {df_final.shape}")

if __name__ == "__main__":
    integrate_real_macro_data()
```

---

## 🛠️ 5. Clean Backward Compatibility Controls

To deploy these enhancements without breaking existing federal pipelines or dashboard tools, adhere strictly to these integration instructions:

1.  **Parquet Cache Auto-Expansion:**
    When reading the historical bills dataset inside `build_dataset.py`, apply an active columns safeguard using Pandas:
    ```python
    # Inside build_dataset.py -> _normalize_historical_bills()
    for col, default in [
        ("jurisdiction", "federal"),
        ("state_code", "US"),
        ("session_year", None),
        ("enacted", True),
        ("sponsor_party", "mixed"),
        ("bill_text_clean", ""),
        ("major_topic", "Macroeconomics")
    ]:
        if col not in dataset.columns:
            dataset[col] = default
    ```
    This ensures that loading older Parquet caches automatically expands the records into the new schema in-memory without failing.
    
2.  **Omit Penalties for Federal baseline comparisons:**
    Set the default jurisdiction value inside `AnalogMatcher` search structures to `"federal"` if not explicitly provided:
    ```python
    target_jur = str(target_policy.get("jurisdiction", "federal")).strip().lower()
    ```
    If target is `"federal"`, and analogs are `"federal"`, both penalties (`state_to_state_penalty` and `fed_to_state_penalty`) evaluate to **zero**, matching original mathematical metrics exactly.
    
3.  **Graceful UI Controls in Streamlit (`app.py`):**
    Group scatter plot elements by a categorical variable mapping `jurisdiction` to Plotly series colors:
    ```python
    fig_pca = px.scatter(
        df_semantic,
        x="PCA 1",
        y="PCA 2",
        color="jurisdiction",  # Color map is automatically generated based on state name or federal status
        symbol="level",        # visual marker distinction (e.g. circle for state, star/diamond for federal)
        hover_data=["bill_id", "title", "direction", "state"]
    )
    ```
