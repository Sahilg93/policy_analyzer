"""
Master Orchestrator: Policy Intelligence Pipeline
Routes bills through the historical-analog engine for scoring.
"""
import logging
from typing import Dict, Optional
import pandas as pd
import requests
import json

from pipeline.ingest_bea import fetch_gdp
from pipeline.bls_client import BLSClient
from pipeline.congress_ingest import CongressIngestor
from pipeline.policy_embeddings import AnalogMatcher
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


def interpret_bill_ollama(title: str) -> Dict:
    """
    Local LLM-based policy interpreter using Ollama.
    Returns structured policy classification.
    """
    prompt = f"""
You are a policy classification system.

Analyze the US congressional bill title and return ONLY valid JSON.

Title: {title}

JSON schema:
{{
  "policy_type": "tax | healthcare | education | regulation | spending | trade | other",
  "direction": "expansionary | contractionary | neutral",
  "intensity": "low | medium | high",
  "sector": "business | households | government | mixed"
}}
"""

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )

        text = r.json().get("response", "")
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("No JSON found")

        return json.loads(text[start:end])

    except Exception as e:
        logger.warning(f"Ollama classification failed: {e}. Using defaults.")
        return {
            "policy_type": "unknown",
            "direction": "neutral",
            "intensity": "low",
            "sector": "unknown"
        }


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
    Currently loads from processed dataset; could extend to full Congress archive.
    """
    logger.info("Loading historical bills...")
    
    try:
        dataset = pd.read_parquet("data/processed/policy_dataset.parquet")
        logger.info(f"Loaded historical bills: shape={dataset.shape}")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to load historical bills: {e}")
        return None


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
        "sector": bill.get("sector", "unknown")
    }
    
    analogs = analog_matcher.find_similar_bills(target_policy, top_k=5)
    
    if not analogs:
        logger.warning(f"No analogs found for {bill_id}")
        return {
            "bill_id": bill_id,
            "title": title,
            "policy_type": target_policy.get("policy_type"),
            "direction": target_policy.get("direction"),
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
    impacts = outcome_engine.estimate_directional_impacts(analogs)
    
    # Step 3: Calculate net score and confidence
    score_result = scoring_engine.calculate_net_score(impacts, analogs)
    
    # Format analog list for output
    analog_list = [
        {
            "bill_id": a.get("bill_id"),
            "title": a.get("title"),
            "similarity_score": round(a.get("similarity_score", 0.0), 3)
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
        
        sample_size = min(10, len(congress_bills))
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
                "introduced_date": row.get("introduced_date"),
                "state": "United States"
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
