#!/usr/bin/env python3
"""
Unified Ingestion & Auto-Vectorization Orchestrator
Idempotently downloads, deduplicates, and generates embeddings for state and federal bills.
"""
import os
import sys
import logging
import argparse
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Ensure project root is in the Python search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from pipeline.config import (
    OPENSTATES_API_KEY,
    CONGRESS_API_KEY,
    BLS_API_KEY,
    OLLAMA_HOST,
    OLLAMA_MODEL_EMBED,
    HISTORICAL_BILLS_PATH,
    EMBEDDINGS_PATH,
    PROCESSED_DIR
)
from pipeline.openstates_client import OpenStatesClient
from pipeline.congress_ingest import CongressIngestor
from pipeline.build_dataset import compress_dataframe

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[ingest_all] %(levelname)s: %(message)s"
)
logger = logging.getLogger("ingest_all")


def get_ollama_embedding(text: str) -> Optional[List[float]]:
    """Fetch vector embedding for text from local Ollama service."""
    if not text or not text.strip():
        return None
        
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {
        "model": OLLAMA_MODEL_EMBED,
        "prompt": text.strip()
    }
    
    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        vector = r.json().get("embedding")
        return vector
    except Exception as e:
        logger.warning(f"Ollama embedding failed for text snippet '{text[:30]}...': {e}")
        return None


def fetch_federal_bills(limit: int = 15) -> List[Dict]:
    """Ingest federal bills from Congress API and normalize to platform schema."""
    logger.info(f"Fetching federal bills (Limit: {limit})...")
    if not CONGRESS_API_KEY or CONGRESS_API_KEY == "fodYfBmI4cxpLigjhMdpY8jfEqUhbeSJKHKAKq4U":
        logger.warning("Congress.gov API key is using demo fallback. Fetch limits may be restricted.")
        
    ingestor = CongressIngestor(api_key=CONGRESS_API_KEY)
    df_fed = ingestor.fetch_bills(limit=limit)
    
    if df_fed.empty:
        logger.warning("No federal bills retrieved.")
        return []
        
    normalized = []
    for _, row in df_fed.iterrows():
        title = row.get("title", "").strip()
        introduced_date = row.get("introduced_date")
        
        session_year = None
        if introduced_date:
            try:
                session_year = int(str(introduced_date)[:4])
            except ValueError:
                pass
                
        bill_id = str(row.get("bill_id", "")).strip()
        
        normalized.append({
            "bill_id": f"fed_{bill_id}" if not bill_id.startswith("fed_") else bill_id,
            "title": title,
            "introduced_date": introduced_date,
            "enacted_date": None,
            "policy_type": "unknown",
            "direction": "neutral",
            "intensity": "low",
            "sector": "mixed",
            "state": "United States",
            "level": "federal",
            "text": title,
            "jurisdiction": "federal",
            "state_code": "US",
            "session_year": session_year,
            "enacted": False,
            "sponsor_party": "mixed",
            "bill_text_clean": title,
            "major_topic": "unknown"
        })
    logger.info(f"Retrieved and normalized {len(normalized)} federal bills.")
    return normalized


def fetch_state_bills(state_code: str, limit: int = 10) -> List[Dict]:
    """Ingest state bills via OpenStates paginated client."""
    logger.info(f"Fetching state bills for {state_code.upper()} (Limit: {limit})...")
    client = OpenStatesClient()
    df_state = client.fetch_state_bills_bulk(state_code=state_code, max_bills=limit)
    
    if df_state.empty:
        logger.warning(f"No state bills retrieved for {state_code.upper()}.")
        return []
        
    # OpenStatesClient already structures to state format, map topic/type fallbacks
    normalized = []
    for _, row in df_state.iterrows():
        bill_dict = row.to_dict()
        # Enforce canonical default mappings
        bill_dict["policy_type"] = bill_dict.get("policy_type") or "unknown"
        bill_dict["direction"] = bill_dict.get("direction") or "neutral"
        bill_dict["intensity"] = bill_dict.get("intensity") or "low"
        bill_dict["sector"] = bill_dict.get("sector") or "mixed"
        bill_dict["major_topic"] = bill_dict.get("major_topic") or "unknown"
        normalized.append(bill_dict)
        
    logger.info(f"Retrieved and normalized {len(normalized)} state bills for {state_code.upper()}.")
    return normalized


def run_unified_ingestion(states: List[str], limit_per_source: int = 15):
    """Run the complete ingestion pipeline idempotently with auto-embeddings."""
    logger.info("Initializing Unified Policy Ingestion Orchestrator...")
    
    # 1. Ensure processed directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Load existing historical databases for idempotency comparisons
    existing_bills = pd.DataFrame()
    existing_embeds = pd.DataFrame()
    
    if HISTORICAL_BILLS_PATH.exists():
        try:
            existing_bills = pd.read_parquet(HISTORICAL_BILLS_PATH)
            logger.info(f"Loaded existing bill database: {len(existing_bills)} records.")
        except Exception as e:
            logger.warning(f"Failed to read existing historical bills: {e}. Starting fresh.")
            
    if EMBEDDINGS_PATH.exists():
        try:
            existing_embeds = pd.read_parquet(EMBEDDINGS_PATH)
            logger.info(f"Loaded existing embeddings database: {len(existing_embeds)} records.")
        except Exception as e:
            logger.warning(f"Failed to read existing embeddings: {e}. Starting fresh.")
            
    # Track existing keys for instant lookup: (bill_id, jurisdiction)
    existing_keys = set()
    if not existing_bills.empty and "bill_id" in existing_bills.columns and "jurisdiction" in existing_bills.columns:
        for _, row in existing_bills.iterrows():
            existing_keys.add((str(row["bill_id"]).strip().lower(), str(row["jurisdiction"]).strip().lower()))
            
    # 3. Gather new candidate records
    candidates = []
    
    # Federal
    try:
        fed_records = fetch_federal_bills(limit=limit_per_source)
        candidates.extend(fed_records)
    except Exception as e:
        logger.error(f"Failed federal ingestion: {e}")
        
    # Selected States
    for st in states:
        try:
            state_records = fetch_state_bills(state_code=st, limit=limit_per_source)
            candidates.extend(state_records)
        except Exception as e:
            logger.error(f"Failed state ingestion for {st.upper()}: {e}")
            
    if not candidates:
        logger.info("No candidates retrieved from any source. Ingestion done.")
        return
        
    # 4. Filter duplicates (idempotency enforcement) and generate auto-embeddings
    new_bills_list = []
    new_embeds_list = []
    
    logger.info("Filtering duplicates and running auto-vectorization embedding pipelines...")
    
    for bill in candidates:
        bid = str(bill["bill_id"]).strip().lower()
        jur = str(bill["jurisdiction"]).strip().lower()
        
        if (bid, jur) in existing_keys:
            logger.info(f"Skipping duplicate bill: {bill['bill_id']} ({bill['jurisdiction'].upper()}) [Already Ingested]")
            continue
            
        # Unique bill found! Let's generate the semantic embedding immediately
        text_to_embed = bill.get("bill_text_clean") or bill.get("title", "")
        vector = get_ollama_embedding(text_to_embed)
        
        if vector is None:
            logger.warning(f"Skipping bill {bill['bill_id']} due to vector generation failure.")
            continue
            
        new_bills_list.append(bill)
        new_embeds_list.append({
            "bill_id": bill["bill_id"],
            "embedding": vector
        })
        
        # Add to keys to prevent duplicate within the same batch ingestion
        existing_keys.add((bid, jur))
        logger.info(f"Auto-embedded and registered new bill: {bill['bill_id']} ({bill['jurisdiction'].upper()})")
        
    if not new_bills_list:
        logger.info("All fetched bills were duplicates. No new records added to database.")
        return
        
    # 5. Append and serialize databases safely back to disk
    df_new_bills = pd.DataFrame(new_bills_list)
    df_new_embeds = pd.DataFrame(new_embeds_list)
    
    if not existing_bills.empty:
        df_final_bills = pd.concat([existing_bills, df_new_bills], ignore_index=True)
    else:
        df_final_bills = df_new_bills
        
    if not existing_embeds.empty:
        df_final_embeds = pd.concat([existing_embeds, df_new_embeds], ignore_index=True)
    else:
        df_final_embeds = df_new_embeds
        
    # Enforce memory optimizations
    df_final_bills = compress_dataframe(df_final_bills)
    
    # Save Parquet databases
    try:
        df_final_bills.to_parquet(HISTORICAL_BILLS_PATH, index=False)
        df_final_embeds.to_parquet(EMBEDDINGS_PATH, index=False)
        logger.info("="*60)
        logger.info(f"INGESTION COMPLETED SUCCESSFULLY!")
        logger.info(f"  - Appended {len(new_bills_list)} new unique bills and embeddings.")
        logger.info(f"  - Total bill database records: {len(df_final_bills)}")
        logger.info(f"  - Total embedding database records: {len(df_final_embeds)}")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Failed saving databases to disk: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Idempotent Policy Ingestion Orchestrator")
    parser.add_argument(
        "--states",
        type=str,
        default="CA,TX,NY,FL,OH,IL,PA,MI",
        help="Comma-separated list of target states (e.g. 'CA,FL')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit of bills fetched per source"
    )
    
    args = parser.parse_args()
    
    target_states = [s.strip().lower() for s in args.states.split(",") if s.strip()]
    run_unified_ingestion(states=target_states, limit_per_source=args.limit)
