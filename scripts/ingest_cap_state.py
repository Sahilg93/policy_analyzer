#!/usr/bin/env python3
"""
CAP Historical Ingestion Adapter
Parses downloaded state legislative .csv files from the Comparative Agendas Project (CAP)
and appends them to our standardized policy events corpus.
"""
import os
import sys
import argparse
import pandas as pd
import logging
from pathlib import Path

# Ensure project root is in the Python search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="[ingest_cap] %(levelname)s: %(message)s"
)
logger = logging.getLogger("ingest_cap")

# CAP Major Topic code string mappings
CAP_TOPIC_MAP = {
    1: "Macroeconomics",
    2: "Civil Rights & Liberties",
    3: "Healthcare",
    4: "Agriculture",
    5: "Labor & Employment",
    6: "Education",
    7: "Energy & Environment",
    8: "Energy & Environment",
    9: "Immigration",
    10: "Transportation",
    12: "Civil Rights & Liberties",
    13: "Other",
    14: "Other",
    15: "Macroeconomics",
    16: "Other",
    17: "Other",
    18: "Macroeconomics",
    19: "Other",
    20: "Government Operations",
    21: "Energy & Environment"
}

# Platform policy_type mapping based on economic topics
CAP_POLICY_TYPE_MAP = {
    1: "spending",      # Macroeconomics
    5: "regulation",    # Labor & Employment
    15: "trade"         # Domestic Commerce
}

def ingest_cap_csv(csv_path: str, state_name: str, state_code: str):
    """
    Parses a CAP state agenda CSV file and appends it to data/policy_events.csv.
    Filters strictly for economic core topics (major topics 1, 5, 15).
    """
    path = Path(csv_path)
    if not path.exists():
        logger.error(f"Target CAP CSV file not found: {csv_path}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Loaded CAP CSV: {len(df)} raw rows.")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        sys.exit(1)
        
    # Standardize column naming lookups dynamically
    col_mapping = {}
    for col in df.columns:
        c_low = col.lower()
        if c_low in {"major", "major_topic", "majortopic", "topic"}:
            col_mapping["major"] = col
        elif c_low in {"year", "session_year", "sessionyear"}:
            col_mapping["year"] = col
        elif c_low in {"title", "bill_title", "headline"}:
            col_mapping["title"] = col
        elif c_low in {"description", "summary", "abstract", "policy_change", "text"}:
            col_mapping["description"] = col
        elif c_low in {"id", "bill_id", "row_id", "rowid"}:
            col_mapping["id"] = col
            
    # Require core columns or use fallbacks
    if "major" not in col_mapping:
        logger.error("Could not find a major topic code column (e.g. 'major', 'major_topic') in the CSV.")
        sys.exit(1)
    if "year" not in col_mapping:
        logger.error("Could not find a year column (e.g. 'year') in the CSV.")
        sys.exit(1)
        
    major_col = col_mapping["major"]
    year_col = col_mapping["year"]
    title_col = col_mapping.get("title", "")
    desc_col = col_mapping.get("description", "")
    id_col = col_mapping.get("id", "")
    
    # 1. Filter strictly for economic core topics: 1 (Macroeconomics), 5 (Labor), 15 (Domestic Commerce)
    df[major_col] = pd.to_numeric(df[major_col], errors="coerce")
    df_econ = df[df[major_col].isin([1, 5, 15])].copy()
    
    if df_econ.empty:
        logger.warning("No economic core topic rows (major topics 1, 5, 15) found in the CAP CSV.")
        return
        
    logger.info(f"Filtered to {len(df_econ)} economic core topic rows.")
    
    # 2. Load existing policy events
    events_path = Path("data/policy_events.csv")
    if events_path.exists():
        try:
            df_events = pd.read_csv(events_path)
            logger.info(f"Loaded existing events: {len(df_events)} rows.")
        except Exception as e:
            logger.warning(f"Error loading existing policy events: {e}. Re-initializing.")
            df_events = pd.DataFrame()
    else:
        df_events = pd.DataFrame()
        
    # Ensure legacy files have the correct new primary columns
    if not df_events.empty:
        if "bill_id" not in df_events.columns:
            df_events["bill_id"] = [f"policy_event_{i+1}" for i in range(len(df_events))]
        if "level" not in df_events.columns:
            df_events["level"] = ["federal" if str(s).strip() in {"United States", "US", "Federal"} else "state" for s in df_events["state"]]
            
    # 3. Reformat and append rows
    new_rows = []
    for idx, row in df_econ.iterrows():
        year = pd.to_numeric(row[year_col], errors="coerce")
        if pd.isna(year):
            continue
        year = int(year)
        
        # Extrapolate title & description
        t_val = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""
        d_val = str(row[desc_col]).strip() if desc_col and pd.notna(row[desc_col]) else ""
        
        if not t_val and not d_val:
            continue
            
        policy_change = t_val if t_val else d_val[:50] + "..."
        description = d_val if d_val else t_val
        
        major_code = int(row[major_col])
        policy_type = CAP_POLICY_TYPE_MAP.get(major_code, "regulation")
        
        # Dynamic direction & intensity defaults
        direction = "neutral"
        intensity = "medium"
        sector = "mixed"
        
        # CAP isolated unique key
        cap_id = str(row[id_col]).strip() if id_col and pd.notna(row[id_col]) else f"{idx}"
        bill_id = f"cap_{state_code.lower()}_{cap_id}"
        
        new_rows.append({
            "year": year,
            "state": state_name.strip().title(),
            "policy_change": policy_change,
            "description": description,
            "policy_type": policy_type,
            "direction": direction,
            "intensity": intensity,
            "sector": sector,
            "bill_id": bill_id,
            "level": "state"
        })
        
    if not new_rows:
        logger.info("No valid economic rows structured. Ingestion done.")
        return
        
    df_new = pd.DataFrame(new_rows)
    
    if not df_events.empty:
        df_final = pd.concat([df_events, df_new], ignore_index=True)
    else:
        df_final = df_new
        
    # Safely drop duplicates based on bill_id (idempotent registry)
    df_final = df_final.drop_duplicates(subset=["bill_id"])
    
    # Save back to CSV
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(events_path, index=False)
        logger.info("="*60)
        logger.info(f"CAP HISTORICAL INGESTION COMPLETED!")
        logger.info(f"  - Appended unique economic rows for {state_name.upper()} ({state_code.upper()}).")
        logger.info(f"  - Total events in data/policy_events.csv: {len(df_final)}")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Failed to write events CSV back to disk: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Comparative Agendas Project (CAP) State Legislative Data")
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the downloaded CAP state dataset CSV file"
    )
    parser.add_argument(
        "--state",
        type=str,
        required=True,
        help="Target state name (e.g. 'Pennsylvania')"
    )
    parser.add_argument(
        "--code",
        type=str,
        required=True,
        help="Target state code (e.g. 'PA')"
    )
    
    args = parser.parse_args()
    ingest_cap_csv(csv_path=args.csv_path, state_name=args.state, state_code=args.code)
