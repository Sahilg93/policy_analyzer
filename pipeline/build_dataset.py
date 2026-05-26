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

from pipeline.config import (
    OPENSTATES_API_KEY,
    BEA_API_KEY,
    BLS_API_KEY,
    CONGRESS_API_KEY,
    OLLAMA_HOST,
    HISTORICAL_BILLS_PATH,
    POLICY_EVENTS_PATH,
    AI_REGISTRY_PATH
)

API_KEY = OPENSTATES_API_KEY
REGISTRY_PATH = AI_REGISTRY_PATH
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
        "model": "llama3",
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


def clean_state_bill_title(title: str) -> str:
    """
    Strips out archaic legal preambles from state legislative titles.
    Isolates core thematic policy content to optimize semantic text embeddings.
    """
    if not title:
        return ""
        
    cleaned = title.strip()
    
    # 1. Remove bracketed or parenthesized P.L. / Act / No. citations
    cleaned = re.sub(r'\(P\.L\..*?,?\s*No\..*?\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'P\.L\..*?,\s*No\..*?\b', '', cleaned, flags=re.IGNORECASE)
    
    # 2. Match standard "An Act amending..." or "An Act providing for..." patterns
    match_prov = re.search(r'(?:further\s+)?providing\s+for\s+(.*)', cleaned, flags=re.IGNORECASE)
    if match_prov:
        cleaned = match_prov.group(1)
    else:
        match_rel = re.search(r'relating\s+to\s+(.*)', cleaned, flags=re.IGNORECASE)
        if match_rel:
            cleaned = match_rel.group(1)
            
    # 3. Strip leading/trailing punctuation or noise leftover
    cleaned = re.sub(r'^[\s,.;:-]+', '', cleaned)
    cleaned = re.sub(r'[\s,.;:-]+$', '', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned if cleaned else title.strip()


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


def rebuild_historical_embeddings() -> pd.DataFrame:
    """
    Directly re-compiles the historical bills corpus from data/policy_events.csv,
    updates the local parquet cache, and batch-vectorizes missing embeddings
    using the AnalogMatcher, serializing them atomically to disk.
    """
    logger.info("Starting automated re-embedding and corpus reconstruction...")
    
    # 1. Force rebuild corpus from CSV to Parquet
    dataset = build_historical_bills_corpus()
    if dataset.empty:
        logger.warning("Historical bills corpus is empty. Cannot rebuild embeddings.")
        return pd.DataFrame()
        
    dataset = compress_dataframe(dataset)
    try:
        HISTORICAL_BILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(HISTORICAL_BILLS_PATH, index=False)
        logger.info(f"Overwrote historical bills parquet cache at {HISTORICAL_BILLS_PATH}: shape={dataset.shape}")
    except Exception as e:
        logger.error(f"Failed to save historical bills parquet cache: {e}")
        
    # 2. Trigger AnalogMatcher to batch-query Ollama and update parquet embeddings cache
    logger.info("Initializing AnalogMatcher to batch-vectorize any new or updated policies...")
    matcher = AnalogMatcher(historical_bills_df=dataset, rebuild_embeddings=False)
    
    logger.info("Re-embedding management completed successfully.")
    return dataset



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

    bill_id = str(event.get("bill_id", "")).strip() if "bill_id" in event and pd.notna(event.get("bill_id")) else ""
    if not bill_id:
        bill_id = f"policy_event_{idx + 1}"

    return {
        "bill_id": bill_id,
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
        is_state = st != "United States"
        if is_state:
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
                
        raw_text = row.get("bill_text_clean") or row.get("text") or row.get("title", "")
        if is_state:
            cleaned_title = clean_state_bill_title(row.get("title", ""))
            cleaned_text = clean_state_bill_title(raw_text)
            dataset.at[idx, "title"] = cleaned_title
            dataset.at[idx, "bill_text_clean"] = cleaned_text
        else:
            dataset.at[idx, "bill_text_clean"] = raw_text

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
    
    # Hierarchical Override Gate completely outside the concurrent thread pool map:
    # Evaluate fully compiled major_topic string variable against CORE_ECONOMIC_TOPICS before skip check
    CORE_ECONOMIC_TOPICS = ["Macroeconomics", "Labor", "Domestic Commerce", "Agriculture"]
    if bill.get("major_topic") in CORE_ECONOMIC_TOPICS and not _coerce_bool(bill.get("macro_relevance", True)):
        logger.info(
            "[OVERRIDE SUCCESS] Enforced macro_relevance=True for %s under Topic: %s",
            bill_id,
            bill.get("major_topic")
        )
        bill["macro_relevance"] = True

    # Ensure that before `if not bill.get("macro_relevance")` is checked for skips, the override is evaluated.
    if not bill.get("macro_relevance"):
        logger.info("Skipping macro scoring for non-economic bill: %s", bill_id)
        return {
            "bill_id": bill_id, "title": title, "policy_type": bill.get("policy_type", "unknown"),
            "direction": bill.get("direction", "neutral"), "macro_relevance": False, "similar_bills": [],
            "estimated_impacts": {"gdp_effect": 0.0, "unemployment_effect": 0.0, "num_analogs_matched": 0, "avg_similarity": 0.0},
            "net_score": 0.0, "confidence": 0.0, "explanation": "Policy classified as having no direct macroeconomic relevance."
        }

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
            "gdp_effect": round(score_result.get("true_gdp_delta", 0.0), 5),
            "unemployment_effect": round(score_result.get("true_unemployment_delta", 0.0), 5),
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


def build(jurisdiction: str = "federal", state_code: Optional[str] = None, rebuild_embeddings: bool = False):
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
            max_bills = 20,
            rebuild_embeddings = rebuild_embeddings
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
    analog_matcher = AnalogMatcher(historical_bills_df=historical_bills, rebuild_embeddings=rebuild_embeddings)
    
    scoring_engine = ScoringEngine(
        gdp_weight=0.4,
        unemployment_weight=-0.3,
        macro_df=macro_data
    )
    
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
        sample_size = min(20, len(congress_bills))
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
            
    # Serialize to scored_bills.csv
    import os
    rows = []
    for r in scored_results:
        impacts = r.get("estimated_impacts", {})
        row_dict = {
            "bill_id": r.get("bill_id"),
            "title": r.get("title"),
            "policy_type": r.get("policy_type"),
            "direction": r.get("direction"),
            "macro_relevance": r.get("macro_relevance", True),
            "net_score": r.get("net_score"),
            "raw_score": r.get("net_score"),
            "confidence": r.get("confidence"),
            "explanation": r.get("explanation"),
            "gdp_comp": impacts.get("gdp_effect", 0.0),
            "unemp_comp": impacts.get("unemployment_effect", 0.0),
            "impact_gdp_effect": impacts.get("gdp_effect", 0.0),
            "impact_unemployment_effect": impacts.get("unemployment_effect", 0.0),
            "impact_num_analogs_matched": impacts.get("num_analogs_matched", 0),
            "impact_avg_similarity": impacts.get("avg_similarity", 0.0)
        }
        rows.append(row_dict)
        
    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/scored_bills.csv", index=False)
    
    logger.info(f"[SERIALIZATION SUCCESS] Exported {len(df)} scored records to data/processed/scored_bills.csv")
    
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
    max_bills: int = 20,
    rebuild_embeddings: bool = False
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
        raw_title = row.get("title", "")
        raw_summary = row.get("bill_text_clean", "")
        
        # Clean archaic East Coast legal preambles before vectorizing or classifying
        title = clean_state_bill_title(raw_title)
        summary = clean_state_bill_title(raw_summary) if raw_summary else title
        
        # Fire the macro interpreter to populate policy_type, direction, macro_relevance
        ai_features = interpret_bill_ollama(title, bill_text_clean=summary)
        
        # Build out the target structured payload
        bill_dict = row.to_dict()
        bill_dict.update(ai_features)
        
        # Update with cleaned title and summary fields for optimized down-stream embedding matching
        bill_dict["title"] = title
        bill_dict["bill_text_clean"] = summary
        
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
    analog_matcher = AnalogMatcher(historical_bills_df=historical_bills, rebuild_embeddings=rebuild_embeddings)
    
    scoring_engine = ScoringEngine(
        gdp_weight=0.4,
        unemployment_weight=-0.3,
        macro_df=macro_data
    )
    
    # 5. Execute scoring matrix loop using enriched dictionaries
    scored_results = []
    logger.info(f"Scoring {len(enriched_bills)} enriched state bills...")
    for bill in enriched_bills:
        result = score_bill(bill, analog_matcher, outcome_engine, scoring_engine)
        scored_results.append(result)
        
    # Serialize to scored_bills.csv
    import os
    rows = []
    for r in scored_results:
        impacts = r.get("estimated_impacts", {})
        row_dict = {
            "bill_id": r.get("bill_id"),
            "title": r.get("title"),
            "policy_type": r.get("policy_type"),
            "direction": r.get("direction"),
            "macro_relevance": r.get("macro_relevance", True),
            "net_score": r.get("net_score"),
            "raw_score": r.get("net_score"),
            "confidence": r.get("confidence"),
            "explanation": r.get("explanation"),
            "gdp_comp": impacts.get("gdp_effect", 0.0),
            "unemp_comp": impacts.get("unemployment_effect", 0.0),
            "impact_gdp_effect": impacts.get("gdp_effect", 0.0),
            "impact_unemployment_effect": impacts.get("unemployment_effect", 0.0),
            "impact_num_analogs_matched": impacts.get("num_analogs_matched", 0),
            "impact_avg_similarity": impacts.get("avg_similarity", 0.0)
        }
        rows.append(row_dict)
        
    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/scored_bills.csv", index=False)
    
    logger.info(f"[SERIALIZATION SUCCESS] Exported {len(df)} scored records to data/processed/scored_bills.csv")
    
    return scored_results
        
def build_campaign_pipeline(
    jurisdiction_name: str,
    state_code: str,
    rebuild_embeddings: bool = False
) -> Optional[List[Dict]]:
    """
    Campaign Orchestrator: Ingest candidate proposals, distill campaign rhetoric,
    classify policies, and score via vectorized engines.
    """
    logger.info("="*60)
    logger.info(f"Starting Campaign Simulation Pipeline for: {jurisdiction_name.strip().title()} ({state_code.upper()})")
    logger.info("="*60)
    
    from pipeline.campaign_adapter import load_and_distill_campaign_policies
    
    # 1. Load macro historical baselines
    macro_data = load_macro_data()
    if macro_data is None or macro_data.empty:
        logger.error("No macro data available. Cannot proceed.")
        return None
        
    historical_bills = load_historical_bills()
    if historical_bills is None or historical_bills.empty:
        logger.warning("No historical bills available. Analog matching disabled.")
        historical_bills = pd.DataFrame()
        
    # 2. Ingest and distill campaign policies
    campaign_df = load_and_distill_campaign_policies(jurisdiction_name)
    if campaign_df.empty:
        logger.warning("No campaign policies found.")
        return None
        
    # 3. AI Enrichment & Policy Classification
    logger.info("Running Ollama policy classification and topic mapping for campaign proposals...")
    
    def enrich_campaign_proposal(row_tuple) -> Dict:
        idx, row = row_tuple
        title = clean_state_bill_title(row.get("policy_title", ""))
        summary = clean_state_bill_title(row.get("distilled_abstract", ""))
        
        # Fire the macro interpreter to populate policy_type, direction, macro_relevance
        ai_features = interpret_bill_ollama(title, bill_text_clean=summary)
        
        # Keep the local CAP topic classifier
        topic = classify_major_topic_ollama(title, summary)
        
        proposal_dict = {
            "bill_id": str(row.get("candidate_id")) + f"_{idx}",
            "title": title,
            "bill_text_clean": summary,
            "state": jurisdiction_name,
            "level": "state",
            "jurisdiction": jurisdiction_name.lower(),
            "state_code": state_code.upper(),
            "candidate_id": row.get("candidate_id"),
            "candidate_name": row.get("candidate_name"),
            "party": row.get("party"),
            "raw_proposal_text": row.get("raw_proposal_text"),
            "major_topic": topic
        }
        proposal_dict.update(ai_features)
        return proposal_dict

    row_tuples = list(campaign_df.iterrows())
    with ThreadPoolExecutor(max_workers=4) as executor:
        enriched_proposals = list(executor.map(enrich_campaign_proposal, row_tuples))
        
    # 4. Initialize Core Processing Engines
    outcome_engine = OutcomeEngine(macro_df=macro_data)
    analog_matcher = AnalogMatcher(historical_bills_df=historical_bills, rebuild_embeddings=rebuild_embeddings)
    scoring_engine = ScoringEngine(
        gdp_weight=0.4,
        unemployment_weight=-0.3,
        macro_df=macro_data
    )
    
    # 5. Execute scoring matrix loop using enriched dictionaries
    scored_results = []
    logger.info(f"Scoring {len(enriched_proposals)} campaign proposals...")
    for prop in enriched_proposals:
        result = score_bill(prop, analog_matcher, outcome_engine, scoring_engine)
        # Retain candidate tracking tags
        result["candidate_id"] = prop["candidate_id"]
        result["candidate_name"] = prop["candidate_name"]
        result["party"] = prop["party"]
        result["raw_proposal_text"] = prop["raw_proposal_text"]
        result["bill_text_clean"] = prop["bill_text_clean"]
        result["major_topic"] = prop.get("major_topic", "Macroeconomics")
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
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Force deletion of local disk cache and rebuild embeddings from scratch"
    )
    
    args = parser.parse_args()
    results = build(
        jurisdiction=args.jurisdiction, 
        state_code=args.state_code, 
        rebuild_embeddings=args.rebuild_embeddings
    )
    
    if results:
        print("\n" + "="*60)
        print(f"FINAL SYSTEM OUTPUT ({args.jurisdiction.upper()})")
        print("="*60)
        for res in results:
            print(json.dumps(res, indent=2, default=str))