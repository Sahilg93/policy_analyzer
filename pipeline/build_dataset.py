"""
Master Orchestrator: Policy Intelligence Pipeline
Routes bills through the historical-analog engine for scoring.
Supports both Federal and State legislative pipelines dynamically via CLI flags.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import requests
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

from pipeline.ingest_bea import fetch_gdp
from pipeline.bls_client import BLSClient
from pipeline.congress_ingest import CongressIngestor
from pipeline.policy_matching import AnalogMatcher
from pipeline.policy_impact_linker import OutcomeEngine
from pipeline.policy_score import ScoringEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

API_KEY = "c9426a2c-debd-4870-9304-616b5e463ea3"
BLS_API_KEY = "71ca07a939aa4e71a82ae2f88ac8ad1e"
HISTORICAL_BILLS_PATH = Path("data/processed/historical_bills.parquet")
POLICY_EVENTS_PATH = Path("data/policy_events.csv")


REGISTRY_PATH = Path("data/processed/ai_classification_registry.parquet")
registry_lock = threading.Lock()
_registry_df = None

def get_classification_from_cache(text_hash: str) -> Optional[Dict]:
    global _registry_df
    with registry_lock:
        if _registry_df is None:
            if REGISTRY_PATH.exists():
                try:
                    _registry_df = pd.read_parquet(REGISTRY_PATH)
                except Exception as e:
                    logger.warning(f"Failed to read registry parquet: {e}")
                    _registry_df = pd.DataFrame(columns=["hash", "policy_type", "direction", "intensity", "sector", "macro_relevance"])
            else:
                _registry_df = pd.DataFrame(columns=["hash", "policy_type", "direction", "intensity", "sector", "macro_relevance"])
        
        match = _registry_df[_registry_df["hash"] == text_hash]
        if not match.empty:
            row = match.iloc[0]
            return {
                "policy_type": str(row["policy_type"]),
                "direction": str(row["direction"]),
                "intensity": str(row["intensity"]),
                "sector": str(row["sector"]),
                "macro_relevance": bool(row["macro_relevance"])
            }
    return None

def save_classification_to_cache(text_hash: str, classification: Dict):
    global _registry_df
    with registry_lock:
        if _registry_df is None:
            if REGISTRY_PATH.exists():
                try:
                    _registry_df = pd.read_parquet(REGISTRY_PATH)
                except Exception as e:
                    _registry_df = pd.DataFrame(columns=["hash", "policy_type", "direction", "intensity", "sector", "macro_relevance"])
            else:
                _registry_df = pd.DataFrame(columns=["hash", "policy_type", "direction", "intensity", "sector", "macro_relevance"])
        
        new_row = pd.DataFrame([{
            "hash": text_hash,
            "policy_type": classification["policy_type"],
            "direction": classification["direction"],
            "intensity": classification["intensity"],
            "sector": classification["sector"],
            "macro_relevance": classification["macro_relevance"]
        }])
        
        _registry_df = _registry_df[_registry_df["hash"] != text_hash]
        _registry_df = pd.concat([_registry_df, new_row], ignore_index=True)
        
        try:
            REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            _registry_df.to_parquet(REGISTRY_PATH, index=False)
        except Exception as e:
            logger.warning(f"Failed to write registry parquet: {e}")


def interpret_bill_ollama(title: str, bill_text_clean: str = "") -> Dict:
    """
    Local LLM-based policy interpreter using Ollama.
    Uses SHA-256 caching and payload truncation to maximize performance.
    """
    DEFAULT_LLM_FALLBACK = {
        "policy_type": "unknown",
        "direction": "neutral",
        "intensity": "low",
        "sector": "unknown",
        "macro_relevance": False
    }

    # Compute a SHA-256 hash of title + bill_text_clean
    combined_text = (title + bill_text_clean).strip()
    text_hash = hashlib.sha256(combined_text.encode('utf-8')).hexdigest()
    
    # Check cache first
    cached_val = get_classification_from_cache(text_hash)
    if cached_val is not None:
        return cached_val

    # Truncate input payload to title or heavily summarized text to reduce context processing
    truncated_input = title.strip()
    if bill_text_clean and len(bill_text_clean) > len(title):
        if len(bill_text_clean) > 200:
            truncated_input = f"{title} - {bill_text_clean[:180]}..."
        else:
            truncated_input = f"{title} - {bill_text_clean}"

    prompt = f"""
You are a macroeconomic policy classification system.

Analyze the US congressional bill title and return ONLY valid JSON.

Title: {truncated_input}

Set "macro_relevance" to true only when the bill has a direct, plausible
transmission mechanism to federal or state GDP or unemployment, such as
changes to public spending, taxes, labor markets, trade, industrial policy,
healthcare financing, education funding, or sector-wide economic regulation.
Set it to false for judicial appointments, criminal law changes, ceremonial
resolutions, constitutional technicalities, subpoenas, agency procedure, or
purely administrative changes without a direct macroeconomic channel.

JSON schema:
{{
  "policy_type": "tax | healthcare | education | regulation | spending | trade | other",
  "direction": "expansionary | contractionary | neutral",
  "intensity": "low | medium | high",
  "sector": "business | households | government | mixed",
  "macro_relevance": true | false
}}
"""

    payload = {
        "model": "phi3:mini",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": 64
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=90
        )
        response.raise_for_status()
        raw_llm_text = response.json().get("response", "").strip()
        json_match = re.search(r"\{.*\}", raw_llm_text, re.DOTALL)

        if json_match:
            clean_json_str = json_match.group(0)
        else:
            clean_json_str = raw_llm_text

        bill_features = json.loads(clean_json_str)
        required_keys = ["policy_type", "direction", "intensity", "sector", "macro_relevance"]

        for key in required_keys:
            if key not in bill_features:
                bill_features[key] = DEFAULT_LLM_FALLBACK[key]

        bill_features["macro_relevance"] = _coerce_bool(bill_features["macro_relevance"])
        
        # Save cache hit
        save_classification_to_cache(text_hash, bill_features)
        
        return bill_features

    except (requests.exceptions.RequestException, json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Ollama classification failed: {e}. Using defaults.")
        return DEFAULT_LLM_FALLBACK


def _coerce_bool(value) -> bool:
    """Parse boolean-like LLM output without trusting string truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def compress_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory footprint of DataFrame by downcasting numeric columns.
    E.g., float64 -> float32, int64 -> int32.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        col_type = df[col].dtype
        if 'float' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif 'int' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


def load_macro_data() -> Optional[pd.DataFrame]:
    """Load macro data (GDP + Unemployment) from processed datasets."""
    logger.info("Loading macro data...")
    try:
        macro = pd.read_parquet("data/processed/policy_dataset.parquet")
        macro = compress_dataframe(macro)
        logger.info(f"Loaded macro data: shape={macro.shape}")
        return macro
    except Exception as e:
        logger.warning(f"Failed to load processed macro data: {e}")
        try:
            gdp = fetch_gdp(API_KEY)
            bls = BLSClient(api_key=BLS_API_KEY)
            unemp = bls.fetch_series(series_ids=["LNS14000000"], start_year=2010, end_year=2025)
            unemp = unemp.groupby("year")["value"].mean().reset_index()
            unemp.columns = ["year", "unemployment_rate"]
            
            macro = gdp.merge(unemp, on="year", how="left")
            logger.info(f"Built macro data fallback: shape={macro.shape}")
            return macro
        except Exception as e2:
            logger.error(f"Failed to build macro data fallback: {e2}")
            return None


def load_historical_bills() -> Optional[pd.DataFrame]:
    """Load historical bills corpus for analog matching."""
    logger.info("Loading historical bills...")
    if POLICY_EVENTS_PATH.exists():
        dataset = build_historical_bills_corpus()
        if dataset.empty:
            return None
        dataset = compress_dataframe(dataset)
        try:
            HISTORICAL_BILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(HISTORICAL_BILLS_PATH, index=False)
            logger.info(f"Built historical bills from {POLICY_EVENTS_PATH}: shape={dataset.shape}")
        except Exception as e:
            logger.warning(f"Built historical bills but could not cache them: {e}")
        return dataset

    try:
        dataset = pd.read_parquet(HISTORICAL_BILLS_PATH)
        dataset = _normalize_historical_bills(dataset)
        dataset = compress_dataframe(dataset)
        logger.info(f"Loaded historical bills: shape={dataset.shape}")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to load historical bills: {e}")
    return None


def build_historical_bills_corpus() -> pd.DataFrame:
    """Build a deterministic historical corpus from generated policy events."""
    if not POLICY_EVENTS_PATH.exists():
        logger.warning(f"Policy events file not found: {POLICY_EVENTS_PATH}")
        return pd.DataFrame()

    rows = []
    try:
        policy_events = pd.read_csv(POLICY_EVENTS_PATH)
        for idx, event in policy_events.iterrows():
            rows.append(_policy_event_to_historical_bill(event, idx))
    except Exception as e:
        logger.warning(f"Failed to load policy events from {POLICY_EVENTS_PATH}: {e}")
        return pd.DataFrame()

    dataset = pd.DataFrame(rows)
    return _normalize_historical_bills(dataset)


def _policy_event_to_historical_bill(event: pd.Series, idx: int) -> Dict:
    """Convert a row from data/policy_events.csv into analog-matcher shape."""
    year = pd.to_numeric(event.get("year"), errors="coerce")
    year = int(year) if pd.notna(year) else None
    policy_change = str(event.get("policy_change", "")).strip()
    description = str(event.get("description", "")).strip()
    title = description or policy_change or f"Policy Event {idx + 1}"
    inferred_type, inferred_direction, inferred_intensity, inferred_sector = (
        _infer_policy_attributes(f"{policy_change} {description}")
    )
    policy_type = str(event.get("policy_type", inferred_type) or inferred_type).strip()
    direction = str(event.get("direction", inferred_direction) or inferred_direction).strip()
    intensity = str(event.get("intensity", inferred_intensity) or inferred_intensity).strip()
    sector = str(event.get("sector", inferred_sector) or inferred_sector).strip()
    state = str(event.get("state", "United States")).strip() or "United States"
    level = "federal" if state in {"United States", "US", "Federal"} else "state"

    return {
        "bill_id": f"policy_event_{idx + 1}",
        "title": title,
        "introduced_date": f"{year}-01-01" if year else None,
        "enacted_date": f"{year}-01-01" if year else None,
        "policy_type": policy_type,
        "direction": direction,
        "intensity": intensity,
        "sector": sector,
        "state": state,
        "level": level,
        "text": f"{policy_change} {description}".strip(),
    }


def _infer_policy_attributes(text: str) -> tuple[str, str, str, str]:
    """Infer coarse deterministic policy attributes for local seed events."""
    normalized = text.lower()
    if "tax" in normalized or "revenue" in normalized or "surtax" in normalized:
        policy_type = "tax"
        sector = "business" if "income" not in normalized else "households"
    elif "health" in normalized or "medicaid" in normalized or "medicare" in normalized:
        policy_type = "healthcare"
        sector = "households"
    elif "school" in normalized or "education" in normalized or "student" in normalized:
        policy_type = "education"
        sector = "government"
    elif "regulation" in normalized or "reform" in normalized:
        policy_type = "regulation"
        sector = "business"
    else:
        policy_type = "other"
        sector = "mixed"

    if any(word in normalized for word in ["cut", "relief", "incentive", "spending"]):
        direction = "expansionary"
    elif any(word in normalized for word in ["increase", "surtax", "limit"]):
        direction = "contractionary"
    else:
        direction = "neutral"

    intensity = "high" if any(word in normalized for word in ["large", "major", "high"]) else "medium"
    return policy_type, direction, intensity, sector


def _normalize_historical_bills(dataset: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the columns expected by AnalogMatcher and OutcomeEngine."""
    if dataset is None or dataset.empty:
        return pd.DataFrame()

    dataset = dataset.copy()
    state_map = {
        "California": "CA", "Texas": "TX", "New York": "NY", "Florida": "FL",
        "Ohio": "OH", "Illinois": "IL", "Pennsylvania": "PA", "Michigan": "MI",
        "United States": "US"
    }
    
    required_defaults = {
        "bill_id": "", "title": "", "introduced_date": None, "enacted_date": None,
        "policy_type": "unknown", "direction": "neutral", "intensity": "low",
        "sector": "mixed", "state": "United States", "level": "federal", "text": "",
        "jurisdiction": "federal", "state_code": "US", "session_year": None,
        "enacted": True, "sponsor_party": "mixed", "bill_text_clean": "",
        "major_topic": "Macroeconomics"
    }

    for column, default in required_defaults.items():
        if column not in dataset.columns:
            dataset[column] = default

    dataset["bill_id"] = dataset["bill_id"].fillna("").astype(str)
    missing_ids = dataset["bill_id"].str.strip().eq("")
    dataset.loc[missing_ids, "bill_id"] = [f"hist_{idx}" for idx in dataset.index[missing_ids]]
    dataset["title"] = dataset["title"].fillna("").astype(str)
    dataset["state"] = dataset["state"].fillna("United States").astype(str)
    dataset["state"] = dataset["state"].replace({"US": "United States", "Federal": "United States"})
    dataset["level"] = dataset["level"].fillna("federal").astype(str).str.lower()
    dataset.loc[dataset["state"].eq("United States"), "level"] = "federal"
    dataset["text"] = (
        dataset["text"].fillna("").astype(str) + " " + dataset["title"].fillna("").astype(str)
    ).str.strip()
    
    for idx, row in dataset.iterrows():
        st = row["state"]
        if st != "United States":
            dataset.at[idx, "jurisdiction"] = st.lower()
            dataset.at[idx, "level"] = "state"
        else:
            dataset.at[idx, "jurisdiction"] = "federal"
            dataset.at[idx, "level"] = "federal"
            
        dataset.at[idx, "state_code"] = state_map.get(st, "US")
        evt_date = row.get("enacted_date") or row.get("introduced_date")
        if evt_date and pd.notna(evt_date):
            try:
                dataset.at[idx, "session_year"] = int(str(evt_date)[:4])
            except Exception:
                pass
                
        if not row.get("bill_text_clean"):
            dataset.at[idx, "bill_text_clean"] = row.get("text", row.get("title", ""))

    return dataset[list(required_defaults.keys())].drop_duplicates(subset=["bill_id"])


def score_bill(
    bill: Dict,
    analog_matcher: AnalogMatcher,
    outcome_engine: OutcomeEngine,
    scoring_engine: ScoringEngine
) -> Dict:
    """Score a single bill through the full pipeline."""
    bill_id = bill.get("bill_id", "unknown")
    title = bill.get("title", "")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Scoring bill: {bill_id}")
    logger.info(f"Title: {title}")
    
    target_policy = {
        "title": title,
        "policy_type": bill.get("policy_type", "unknown"),
        "direction": bill.get("direction", "neutral"),
        "intensity": bill.get("intensity", "low"),
        "sector": bill.get("sector", "unknown"),
        "level": bill.get("level", "federal"),
        "state": bill.get("state", "United States"),
        "jurisdiction": bill.get("jurisdiction", "federal"),
        "state_code": bill.get("state_code", "US"),
        "bill_text_clean": bill.get("bill_text_clean", ""),
        "major_topic": bill.get("major_topic", "Macroeconomics"),
        "macro_relevance": _coerce_bool(bill.get("macro_relevance", True))
    }

    if not target_policy["macro_relevance"]:
        logger.info("Skipping macro scoring for non-economic bill: %s", bill_id)
        return {
            "bill_id": bill_id, "title": title, "policy_type": target_policy.get("policy_type"),
            "direction": target_policy.get("direction"), "macro_relevance": False, "similar_bills": [],
            "estimated_impacts": {"gdp_effect": 0.0, "unemployment_effect": 0.0, "num_analogs_matched": 0, "avg_similarity": 0.0},
            "net_score": 0.0, "confidence": 0.0, "explanation": "Policy classified as having no direct macroeconomic relevance."
        }
    
    analogs = analog_matcher.find_similar_bills(target_policy, min_threshold=0.7)
    
    if not analogs:
        logger.warning(f"No analogs found for {bill_id}")
        return {
            "bill_id": bill_id, "title": title, "policy_type": target_policy.get("policy_type"),
            "direction": target_policy.get("direction"), "macro_relevance": True, "similar_bills": [],
            "estimated_impacts": {"gdp_effect": 0.0, "unemployment_effect": 0.0, "num_analogs_matched": 0, "avg_similarity": 0.0},
            "net_score": 0.0, "confidence": 0.0, "explanation": "No historical analogs found in corpus"
        }
    
    impacts = outcome_engine.estimate_directional_impacts(analogs, current_bill=target_policy)
    score_result = scoring_engine.calculate_net_score(impacts, analogs)
    
    analog_list = [
        {
            "bill_id": a.get("bill_id"),
            "title": a.get("title"),
            "similarity_score": round(a.get("similarity_score", 0.0), 3),
            "level": a.get("level", "federal")
        }
        for a in analogs
    ]
    
    explanation = _build_explanation(target_policy, impacts, score_result, len(analogs))
    
    result = {
        "bill_id": bill_id, "title": title, "policy_type": target_policy.get("policy_type"),
        "direction": target_policy.get("direction"), "macro_relevance": True, "similar_bills": analog_list,
        "estimated_impacts": {
            "gdp_effect": round(impacts.get("gdp_effect", 0.0), 3),
            "unemployment_effect": round(impacts.get("unemployment_effect", 0.0), 3),
            "num_analogs_matched": impacts.get("num_analogs_matched", 0),
            "avg_similarity": round(impacts.get("avg_similarity", 0.0), 3)
        },
        "net_score": round(score_result.get("net_score", 0.0), 3),
        "confidence": round(score_result.get("confidence", 0.0), 3),
        "explanation": explanation
    }
    
    logger.info(f"Final score: {result['net_score']:.3f} | Confidence: {result['confidence']:.3f}")
    return result


def _build_explanation(target: Dict, impacts: Dict, score_result: Dict, num_analogs: int) -> str:
    """Generate human-readable explanation of score."""
    parts = [f"Found {num_analogs} historical analogs."]
    gdp_dir = "positive" if impacts.get("gdp_effect", 0.0) > 0.05 else "negative" if impacts.get("gdp_effect", 0.0) < -0.05 else "neutral"
    unemp_dir = "beneficial" if impacts.get("unemployment_effect", 0.0) < -0.05 else "harmful" if impacts.get("unemployment_effect", 0.0) > 0.05 else "neutral"
    parts.append(f"Historical impact on GDP: {gdp_dir}. Impact on unemployment: {unemp_dir}.")
    conf = score_result.get("confidence", 0.0)
    conf_desc = "high" if conf > 0.7 else "moderate" if conf > 0.4 else "low"
    parts.append(f"Confidence in estimate: {conf_desc} ({conf:.1%}).")
    return " ".join(parts)


def build(jurisdiction: str = "federal", state_code: Optional[str] = None):
    """
    Main orchestrator: Detects target jurisdiction and dynamically routes
    to either the Federal Congress pipeline or the State Legislature pipeline.
    """
    jurisdiction_clean = jurisdiction.strip().lower()
    
    if jurisdiction_clean != "federal":
        if not state_code:
            logger.error("State code (e.g., 'FL') must be provided for state-level pipelines.")
            return None
        return build_state_pipeline(
            jurisdiction_name=jurisdiction,
            state_code=state_code,
            openstates_key="c9426a2c-debd-4870-9304-616b5e463ea3",
            max_bills = 10
        )
        
    logger.info("Policy Intelligence Platform - Federal Orchestration Started")
    
    macro_data = load_macro_data()
    if macro_data is None or macro_data.empty:
        logger.error("No macro data available. Cannot proceed.")
        return None
    
    historical_bills = load_historical_bills()
    if historical_bills is None or historical_bills.empty:
        logger.warning("No historical bills available. Analog matching disabled.")
        historical_bills = pd.DataFrame()
    
    logger.info("Initializing engines...")
    outcome_engine = OutcomeEngine(macro_df=macro_data)
    analog_matcher = AnalogMatcher(historical_bills_df=historical_bills)
    scoring_engine = ScoringEngine(gdp_weight=0.4, unemployment_weight=-0.3)
    
    logger.info("Fetching Congressional bills...")
    congress_ingestor = CongressIngestor("fodYfBmI4cxpLigjhMdpY8jfEqUhbeSJKHKAKq4U")
    congress_bills = congress_ingestor.fetch_bills()
    
    if congress_bills is None or congress_bills.empty:
        logger.warning("No Congressional bills available")
        congress_bills = pd.DataFrame()
    else:
        logger.info(f"Fetched {len(congress_bills)} Congressional bills")
    
    if not congress_bills.empty:
        logger.info("Running Ollama policy classification...")
        sample_size = min(10, len(congress_bills))
        congress_sample = congress_bills.head(sample_size).copy()
        
        ai_outputs = congress_sample.apply(
            lambda r: interpret_bill_ollama(r["title"], r["bill_text_clean"] if "bill_text_clean" in r else ""),
            axis=1
        )
        ai_df = pd.json_normalize(ai_outputs)
        
        congress_bills = pd.concat([congress_sample.reset_index(drop=True), ai_df], axis=1)
        logger.info(f"AI enrichment complete: {congress_bills.shape}")
    
    scored_results = []
    if not congress_bills.empty:
        logger.info(f"\nScoring {len(congress_bills)} bills...")
        for idx, row in congress_bills.iterrows():
            bill = {
                "bill_id": row.get("bill_id", f"bill_{idx}"),
                "title": row.get("title", ""),
                "policy_type": row.get("policy_type", "unknown"),
                "direction": row.get("direction", "neutral"),
                "intensity": row.get("intensity", "low"),
                "sector": row.get("sector", "unknown"),
                "macro_relevance": _coerce_bool(row.get("macro_relevance", False)),
                "introduced_date": row.get("introduced_date"),
                "state": "United States",
                "level": "federal",
                "jurisdiction": "federal",
                "state_code": "US",
                "bill_text_clean": row.get("title", ""),
                "major_topic": "Macroeconomics"
            }
            result = score_bill(bill, analog_matcher, outcome_engine, scoring_engine)
            scored_results.append(result)
    
    logger.info("\n" + "="*60)
    logger.info("FEDERAL SCORING COMPLETE")
    logger.info("="*60)
    
    return scored_results


def classify_major_topic_ollama(title: str, clean_text: str) -> str:
    """Local LLM-based CAP-style major topic classifier."""
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
        "options": {
            "temperature": 0.0,
            "num_predict": 32
        }
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=20)
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
        
        allowed = ["Macroeconomics", "Taxation", "Healthcare", "Education", "Energy & Environment", 
                   "Civil Rights & Liberties", "Labor & Employment", "Government Operations", "Transportation"]
        for topic in allowed:
            if topic.lower() in raw_text.lower():
                return topic
        return "Other"
    except Exception:
        return "Other"


def build_state_pipeline(
    jurisdiction_name: str, 
    state_code: str, 
    openstates_key: str, 
    max_bills: int = 20
) -> Optional[List[Dict]]:
    """
    State Orchestrator: Ingest state bills, classify, and score via engines.
    Dynamically routes to the target jurisdiction without hardcoded state strings.
    Guarantees robust macro enrichment to prevent unclassified evaluation drops.
    """
    logger.info("="*60)
    logger.info(f"Starting State Pipeline for jurisdiction: {jurisdiction_name.strip().title()} ({state_code.upper()})")
    logger.info("="*60)
    
    from pipeline.openstates_client import OpenStatesClient
    
    # 1. Load macro historical baselines
    macro_data = load_macro_data()
    if macro_data is None or macro_data.empty:
        logger.error("No macro data available. Cannot proceed.")
        return None
        
    historical_bills = load_historical_bills()
    if historical_bills is None or historical_bills.empty:
        logger.warning("No historical bills available. Analog matching disabled.")
        historical_bills = pd.DataFrame()
        
    # 2. Ingest state bills dynamically passing both the proper name and code
    client = OpenStatesClient(api_key=openstates_key)
    raw_state_df = client.fetch_state_bills_bulk(jurisdiction_name=jurisdiction_name, state_code=state_code, max_bills=max_bills)
    
    if raw_state_df.empty:
        logger.warning("No state bills found.")
        return None
    state_bills = raw_state_df.copy(deep=True)
        
    # 3. AI Enrichment & Policy Classification
    logger.info("Running Ollama policy classification and macro-filtering in parallel...")
    
    def enrich_single_bill(row_tuple) -> Dict:
        idx, row = row_tuple
        title = row.get("title", "")
        summary = row.get("bill_text_clean", "")
        
        # Fire the macro interpreter to populate policy_type, direction, macro_relevance
        ai_features = interpret_bill_ollama(title, bill_text_clean=summary)
        
        # Build out the target structured payload
        bill_dict = row.to_dict()
        bill_dict.update(ai_features)
        
        # Explicitly enforce jurisdictional tracking tags for state scoping
        bill_dict["level"] = "state"
        bill_dict["jurisdiction"] = jurisdiction_name.strip().lower()
        bill_dict["state_code"] = state_code.upper()
        
        # Keep the local CAP topic classifier for visual analytics grouping
        topic = classify_major_topic_ollama(title, summary)
        bill_dict["major_topic"] = topic
        
        return bill_dict

    row_tuples = list(state_bills.iterrows())
    with ThreadPoolExecutor(max_workers=4) as executor:
        enriched_bills = list(executor.map(enrich_single_bill, row_tuples))
        
    # 4. Initialize Core Processing Engines
    outcome_engine = OutcomeEngine(macro_df=macro_data)
    analog_matcher = AnalogMatcher(historical_bills_df=historical_bills)
    scoring_engine = ScoringEngine()
    
    # 5. Execute scoring matrix loop using enriched dictionaries
    scored_results = []
    logger.info(f"Scoring {len(enriched_bills)} enriched state bills...")
    for bill in enriched_bills:
        result = score_bill(bill, analog_matcher, outcome_engine, scoring_engine)
        scored_results.append(result)
        
    return scored_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrate Policy Intelligence Engine")
    parser.add_argument(
        "--jurisdiction",
        type=str,
        default="federal",
        help="Target runtime jurisdiction: 'federal' or a specific state (e.g. 'Florida')"
    )
    parser.add_argument(
        "--state-code",
        type=str,
        default=None,
        help="Two-letter state postal abbreviation code (e.g. 'FL')"
    )
    
    args = parser.parse_args()
    results = build(jurisdiction=args.jurisdiction, state_code=args.state_code)
    
    if results:
        print("\n" + "="*60)
        print(f"FINAL SYSTEM OUTPUT ({args.jurisdiction.upper()})")
        print("="*60)
        for res in results:
            print(json.dumps(res, indent=2, default=str))