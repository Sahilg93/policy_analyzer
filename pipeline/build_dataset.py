"""
Master Orchestrator: Policy Intelligence Pipeline
Routes bills through the historical-analog engine for scoring.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

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

API_KEY = "84DF9CAA-34FB-4555-BDF0-130FEA791DA2"
BLS_API_KEY = "71ca07a939aa4e71a82ae2f88ac8ad1e"
HISTORICAL_BILLS_PATH = Path("data/processed/historical_bills.parquet")
POLICY_EVENTS_PATH = Path("data/policy_events.csv")


def interpret_bill_ollama(title: str) -> Dict:
    """
    Local LLM-based policy interpreter using Ollama.
    Returns structured policy classification.
    """

    DEFAULT_LLM_FALLBACK = {
        "policy_type": "unknown",
        "direction": "neutral",
        "intensity": "low",
        "sector": "unknown",
        "macro_relevance": False
    }

    prompt = f"""
You are a macroeconomic policy classification system.

Analyze the US congressional bill title and return ONLY valid JSON.

Title: {title}

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
        "format": "json"
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=45
        )

        response.raise_for_status()

        raw_llm_text = response.json().get("response", "").strip()

        json_match = re.search(r"\{.*\}", raw_llm_text, re.DOTALL)

        if json_match:
            clean_json_str = json_match.group(0)
        else:
            clean_json_str = raw_llm_text

        bill_features = json.loads(clean_json_str)

        required_keys = [
            "policy_type",
            "direction",
            "intensity",
            "sector",
            "macro_relevance"
        ]

        for key in required_keys:
            if key not in bill_features:
                bill_features[key] = DEFAULT_LLM_FALLBACK[key]

        bill_features["macro_relevance"] = _coerce_bool(
            bill_features["macro_relevance"]
        )

        return bill_features

    except (
        requests.exceptions.RequestException,
        json.JSONDecodeError,
        AttributeError
    ) as e:

        logger.warning(
            f"Ollama classification failed: {e}. Using defaults."
        )

        return DEFAULT_LLM_FALLBACK


def _coerce_bool(value) -> bool:
    """Parse boolean-like LLM output without trusting string truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def load_macro_data() -> Optional[pd.DataFrame]:
    """
    Load macro data (GDP + Unemployment) from processed datasets.
    """
    logger.info("Loading macro data...")
    
    try:
        # Load pre-processed macro metrics
        macro = pd.read_parquet("data/processed/policy_dataset.parquet")
        logger.info(f"Loaded macro data: shape={macro.shape}")
        return macro
    except Exception as e:
        logger.warning(f"Failed to load processed macro data: {e}")
        
        # Fallback: build from raw sources
        try:
            gdp = fetch_gdp(API_KEY)
            bls = BLSClient(api_key=BLS_API_KEY)
            unemp = bls.fetch_series(
                series_ids=["LNS14000000"],
                start_year=2010,
                end_year=2025
            )
            unemp = unemp.groupby("year")["value"].mean().reset_index()
            unemp.columns = ["year", "unemployment_rate"]
            
            macro = gdp.merge(unemp, on="year", how="left")
            logger.info(f"Built macro data fallback: shape={macro.shape}")
            return macro
        except Exception as e2:
            logger.error(f"Failed to build macro data fallback: {e2}")
            return None


def load_historical_bills() -> Optional[pd.DataFrame]:
    """
    Load historical bills corpus for analog matching.
    Prefers the generated policy-events CSV and falls back to cached parquet.
    """
    logger.info("Loading historical bills...")

    if POLICY_EVENTS_PATH.exists():
        dataset = build_historical_bills_corpus()
        if dataset.empty:
            return None

        try:
            HISTORICAL_BILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(HISTORICAL_BILLS_PATH, index=False)
            logger.info(
                f"Built historical bills from {POLICY_EVENTS_PATH}: "
                f"shape={dataset.shape}"
            )
        except Exception as e:
            logger.warning(f"Built historical bills but could not cache them: {e}")

        return dataset

    try:
        dataset = pd.read_parquet(HISTORICAL_BILLS_PATH)
        dataset = _normalize_historical_bills(dataset)
        logger.info(f"Loaded historical bills: shape={dataset.shape}")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to load historical bills: {e}")

    return None


def build_historical_bills_corpus() -> pd.DataFrame:
    """
    Build a deterministic historical corpus from generated policy events.
    """
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
    required_defaults = {
        "bill_id": "",
        "title": "",
        "introduced_date": None,
        "enacted_date": None,
        "policy_type": "unknown",
        "direction": "neutral",
        "intensity": "low",
        "sector": "mixed",
        "state": "United States",
        "level": "federal",
        "text": "",
    }

    for column, default in required_defaults.items():
        if column not in dataset.columns:
            dataset[column] = default

    dataset["bill_id"] = dataset["bill_id"].fillna("").astype(str)
    missing_ids = dataset["bill_id"].str.strip().eq("")
    dataset.loc[missing_ids, "bill_id"] = [
        f"hist_{idx}" for idx in dataset.index[missing_ids]
    ]

    dataset["title"] = dataset["title"].fillna("").astype(str)
    dataset["state"] = dataset["state"].fillna("United States").astype(str)
    dataset["state"] = dataset["state"].replace({"US": "United States", "Federal": "United States"})
    dataset["level"] = dataset["level"].fillna("federal").astype(str).str.lower()
    dataset.loc[dataset["state"].eq("United States"), "level"] = "federal"
    dataset["text"] = (
        dataset["text"].fillna("").astype(str) + " " + dataset["title"].fillna("").astype(str)
    ).str.strip()

    return dataset[list(required_defaults.keys())].drop_duplicates(subset=["bill_id"])


def score_bill(
    bill: Dict,
    analog_matcher: AnalogMatcher,
    outcome_engine: OutcomeEngine,
    scoring_engine: ScoringEngine
) -> Dict:
    """
    Score a single bill through the full pipeline.
    
    Args:
        bill: Dict with title, policy_type, direction, intensity, sector, bill_id, etc.
        analog_matcher, outcome_engine, scoring_engine: Pipeline engines
    
    Returns:
        Structured result dict with score, impacts, and confidence
    """
    bill_id = bill.get("bill_id", "unknown")
    title = bill.get("title", "")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Scoring bill: {bill_id}")
    logger.info(f"Title: {title}")
    
    # Step 1: Find historical analogs
    target_policy = {
        "title": title,
        "policy_type": bill.get("policy_type", "unknown"),
        "direction": bill.get("direction", "neutral"),
        "intensity": bill.get("intensity", "low"),
        "sector": bill.get("sector", "unknown"),
        "level": bill.get("level", "federal"),
        "macro_relevance": _coerce_bool(bill.get("macro_relevance", True))
    }

    if not target_policy["macro_relevance"]:
        logger.info("Skipping macro scoring for non-economic bill: %s", bill_id)
        return {
            "bill_id": bill_id,
            "title": title,
            "policy_type": target_policy.get("policy_type"),
            "direction": target_policy.get("direction"),
            "macro_relevance": False,
            "similar_bills": [],
            "estimated_impacts": {
                "gdp_effect": 0.0,
                "unemployment_effect": 0.0,
                "num_analogs_matched": 0,
                "avg_similarity": 0.0
            },
            "net_score": 0.0,
            "confidence": 0.0,
            "explanation": "Policy classified as having no direct macroeconomic relevance."
        }
    
    analogs = analog_matcher.find_similar_bills(target_policy, min_threshold=0.7)
    
    if not analogs:
        logger.warning(f"No analogs found for {bill_id}")
        return {
            "bill_id": bill_id,
            "title": title,
            "policy_type": target_policy.get("policy_type"),
            "direction": target_policy.get("direction"),
            "macro_relevance": True,
            "similar_bills": [],
            "estimated_impacts": {
                "gdp_effect": 0.0,
                "unemployment_effect": 0.0,
                "num_analogs_matched": 0,
                "avg_similarity": 0.0
            },
            "net_score": 0.0,
            "confidence": 0.0,
            "explanation": "No historical analogs found in corpus"
        }
    
    # Step 2: Estimate directional impacts from analogs
    impacts = outcome_engine.estimate_directional_impacts(
        analogs,
        current_bill=target_policy
    )
    
    # Step 3: Calculate net score and confidence
    score_result = scoring_engine.calculate_net_score(impacts, analogs)
    
    # Format analog list for output
    analog_list = [
        {
            "bill_id": a.get("bill_id"),
            "title": a.get("title"),
            "similarity_score": round(a.get("similarity_score", 0.0), 3),
            "level": a.get("level", "federal")
        }
        for a in analogs
    ]
    
    # Build explanation
    explanation = _build_explanation(
        target_policy,
        impacts,
        score_result,
        len(analogs)
    )
    
    result = {
        "bill_id": bill_id,
        "title": title,
        "policy_type": target_policy.get("policy_type"),
        "direction": target_policy.get("direction"),
        "macro_relevance": True,
        "similar_bills": analog_list,
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


def _build_explanation(
    target: Dict,
    impacts: Dict,
    score_result: Dict,
    num_analogs: int
) -> str:
    """Generate human-readable explanation of score."""
    parts = []
    
    # Analog info
    parts.append(f"Found {num_analogs} historical analogs.")
    
    # Impact direction
    gdp_dir = "positive" if impacts.get("gdp_effect", 0.0) > 0.05 else "negative" if impacts.get("gdp_effect", 0.0) < -0.05 else "neutral"
    unemp_dir = "beneficial" if impacts.get("unemployment_effect", 0.0) < -0.05 else "harmful" if impacts.get("unemployment_effect", 0.0) > 0.05 else "neutral"
    
    parts.append(f"Historical impact on GDP: {gdp_dir}. Impact on unemployment: {unemp_dir}.")
    
    # Confidence note
    conf = score_result.get("confidence", 0.0)
    conf_desc = "high" if conf > 0.7 else "moderate" if conf > 0.4 else "low"
    parts.append(f"Confidence in estimate: {conf_desc} ({conf:.1%}).")
    
    return " ".join(parts)


def build():
    """
    Main orchestrator: build macro data, ingest bills, score via analogs.
    """
    logger.info("Policy Intelligence Platform - Orchestration Started")
    
    # ============================================================
    # LOAD DATA
    # ============================================================
    
    macro_data = load_macro_data()
    if macro_data is None or macro_data.empty:
        logger.error("No macro data available. Cannot proceed.")
        return None
    
    historical_bills = load_historical_bills()
    if historical_bills is None or historical_bills.empty:
        logger.warning("No historical bills available. Analog matching disabled.")
        historical_bills = pd.DataFrame()
    
    # ============================================================
    # INITIALIZE ENGINES
    # ============================================================
    
    logger.info("Initializing engines...")
    outcome_engine = OutcomeEngine(macro_df=macro_data)
    analog_matcher = AnalogMatcher(historical_bills_df=historical_bills)
    scoring_engine = ScoringEngine(gdp_weight=0.4, unemployment_weight=-0.3)
    
    # ============================================================
    # INGEST FEDERAL BILLS (CONGRESS API)
    # ============================================================
    
    logger.info("Fetching Congressional bills...")
    congress_ingestor = CongressIngestor("fodYfBmI4cxpLigjhMdpY8jfEqUhbeSJKHKAKq4U")
    congress_bills = congress_ingestor.fetch_bills()
    
    if congress_bills is None or congress_bills.empty:
        logger.warning("No Congressional bills available")
        congress_bills = pd.DataFrame()
    else:
        logger.info(f"Fetched {len(congress_bills)} Congressional bills")
    
    # ============================================================
    # AI ENRICHMENT (LOCAL OLLAMA)
    # ============================================================
    
    if not congress_bills.empty:
        logger.info("Running Ollama policy classification...")
        
        sample_size = min(30, len(congress_bills))
        congress_sample = congress_bills.head(sample_size).copy()
        
        ai_outputs = congress_sample["title"].apply(interpret_bill_ollama)
        ai_df = pd.json_normalize(ai_outputs)
        
        congress_sample = pd.concat(
            [congress_sample.reset_index(drop=True), ai_df],
            axis=1
        )
        
        congress_bills = congress_sample
        logger.info(f"AI enrichment complete: {congress_bills.shape}")
    
    # ============================================================
    # SCORE BILLS
    # ============================================================
    
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
                "level": row.get("level", "federal")
            }
            
            result = score_bill(bill, analog_matcher, outcome_engine, scoring_engine)
            scored_results.append(result)
    
    else:
        logger.warning("No bills to score")
    
    # ============================================================
    # OUTPUT
    # ============================================================
    
    logger.info("\n" + "="*60)
    logger.info("SCORING COMPLETE")
    logger.info("="*60)
    
    if scored_results:
        logger.info(f"\nScored {len(scored_results)} bills:")
        for res in scored_results:
            logger.info(
                f"  {res['bill_id']}: score={res['net_score']:.3f} | "
                f"conf={res['confidence']:.3f} | matches={res['estimated_impacts']['num_analogs_matched']}"
            )
        
        return scored_results
    else:
        logger.warning("No results to return")
        return None


if __name__ == "__main__":
    results = build()
    if results:
        import json
        print("\n" + "="*60)
        print("FINAL OUTPUT")
        print("="*60)
        for res in results:
            print(json.dumps(res, indent=2, default=str))
