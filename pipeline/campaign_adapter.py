"""
Campaign Platform Ingestion Adapter: Rhetoric Distillation Layer
"""
import logging
import hashlib
import os
import requests
import json
import pandas as pd
from typing import Dict, List, Optional
from pipeline.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

CAMPAIGN_REGISTRY_PATH = PROCESSED_DIR / "campaign_distilled_registry.parquet"

def get_distilled_from_cache(text_hash: str) -> Optional[str]:
    """Retrieves distilled abstract from persistent disk parquet cache."""
    if CAMPAIGN_REGISTRY_PATH.exists():
        try:
            df = pd.read_parquet(CAMPAIGN_REGISTRY_PATH)
            match = df[df["hash"] == text_hash]
            if not match.empty:
                return str(match.iloc[0]["distilled_text"])
        except Exception as e:
            logger.warning(f"Failed to read campaign registry: {e}")
    return None

def save_distilled_to_cache(text_hash: str, distilled_text: str):
    """Saves distilled abstract to persistent disk parquet cache."""
    try:
        CAMPAIGN_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CAMPAIGN_REGISTRY_PATH.exists():
            df = pd.read_parquet(CAMPAIGN_REGISTRY_PATH)
        else:
            df = pd.DataFrame(columns=["hash", "distilled_text"])
            
        new_row = pd.DataFrame([{"hash": text_hash, "distilled_text": distilled_text}])
        df = df[df["hash"] != text_hash]
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_parquet(CAMPAIGN_REGISTRY_PATH, index=False)
    except Exception as e:
        logger.warning(f"Failed to save to campaign registry: {e}")

import re

def clean_conversational_preamble(text: str, original_text: str, allow_re_prompt: bool = True) -> str:
    """Strips conversational preambles, chatty headers, and echoes from distilled abstracts."""
    s = text.strip()
    
    lower_s = s.lower()
    # Explicitly check for requested variations
    has_variation = (
        "here is a" in lower_s or
        "2-sentence dry summary" in lower_s or
        "abstract:" in lower_s or
        "the proposed policy" in lower_s
    )
    
    if has_variation:
        if ":" in s:
            # Dynamic partition string splitting using split(":")[-1] to completely remove preamble
            s = s.split(":")[-1].strip()
        else:
            # If no colon, use regex to strip out the leading conversational preamble
            s = re.sub(
                r"(?i)^(?:here is a|2-sentence dry summary|abstract|the proposed policy)\s*(?:is|of|to|establishing|allocating)?\s*",
                "",
                s
            ).strip()
            
    # 1. Clean other common conversational preambles
    patterns_to_remove = [
        r"(?i)^here is (?:a|the)?\s*(?:dry|neutral|2-sentence)?\s*(?:economic|administrative|economist)?\s*(?:summary|abstract|proposal|analysis|mechanics|statement)?.*?:",
        r"(?i)^abstract\s*:",
        r"(?i)^summary\s*:",
        r"(?i)^2-sentence\s*dry\s*economic\s*abstract\s*:",
        r"(?i)^economist\s*summary\s*:"
    ]
    for pattern in patterns_to_remove:
        s = re.sub(pattern, "", s).strip()
        
    # If the text has a colon in the first 60 characters and starts with common chatty words, split it
    if ":" in s[:60]:
        first_part = s.split(":")[0].lower()
        if any(word in first_part for word in ["here", "summary", "abstract", "proposed", "economist", "dry", "policy"]):
            s = s.split(":", 1)[-1].strip()
            
    # Strip leading/trailing quotes
    s = s.strip('"').strip("'").strip()
    
    # Capitalize the first letter if it got lowercased by pruning
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    
    # 2. Check if distilled abstract matches original statement exactly or is too long (echoing)
    if allow_re_prompt and (s.lower() == original_text.strip().lower() or len(s) > 0.85 * len(original_text)):
        logger.info("[LLM OVERRIDE] Abstract matched original text or is too long. Re-prompting with strict zero-shot system constraint.")
        
        prompt = f"""
You are a public finance economist. Write ONLY the 2-sentence dry administrative abstract of the tax, fiscal, or regulatory mechanism in this policy.
Do NOT echo the original text. Do NOT write any introduction or preamble. Do NOT write "Here is a". Write ONLY the 2-sentence mechanism.

Policy:
{original_text}

ONLY 2-sentence dry administrative abstract:
"""
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 64
            }
        }
        try:
            r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=20)
            r.raise_for_status()
            s_new = r.json().get("response", "").strip()
            # Prune again recursively without re-prompting
            s_new = clean_conversational_preamble(s_new, original_text, allow_re_prompt=False)
            if s_new and s_new.lower() != original_text.strip().lower() and len(s_new) < 0.8 * len(original_text):
                s = s_new
        except Exception as e:
            logger.warning(f"Override re-prompt failed: {e}")
            
    return s

def distill_campaign_rhetoric(raw_proposal_text: str) -> str:
    """
    Distills political campaign rhetoric into dry, neutral, 2-sentence administrative abstracts.
    Utilizes a local Ollama llama3 instance with strict economist public finance system constraints.
    """
    text_clean = str(raw_proposal_text).strip()
    if not text_clean:
        return ""
        
    text_hash = hashlib.sha256(text_clean.encode("utf-8")).hexdigest()
    
    # Check cache first
    cached_val = get_distilled_from_cache(text_hash)
    if cached_val is not None:
        logger.info("Campaign distillation cache hit!")
        # Clean cached value on retrieval to strip any stored conversational preambles
        return clean_conversational_preamble(cached_val, text_clean)

    # Strict public finance economist system prompt
    prompt = f"""
You are a strict, completely objective, and unbiased public finance economist.
Your task is to analyze the following campaign policy statement, completely strip all partisan buzzwords, emotional appeals, and ideological rhetoric, and output a dry, neutral, 2-sentence administrative abstract focusing exclusively on the underlying fiscal, tax, or regulatory mechanism.

Do not introduce any personal commentary, judgment, or analysis. Write ONLY the 2-sentence dry summary of the mechanical economic policy.

Campaign Statement:
{text_clean}

2-Sentence Dry Economic Abstract:
"""

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 64
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        distilled = response.json().get("response", "").strip()
        
        # Apply output pruning
        distilled = clean_conversational_preamble(distilled, text_clean)
        
        # Save cache
        if distilled:
            save_distilled_to_cache(text_hash, distilled)
            return distilled
    except Exception as e:
        logger.warning(f"Ollama campaign distillation failed: {e}. Falling back to raw text truncation.")
        
    # Fallback to truncated text
    return text_clean[:180] + "..." if len(text_clean) > 180 else text_clean

def load_and_distill_campaign_policies(jurisdiction: str) -> pd.DataFrame:
    """
    Loads data/campaign_policies.csv, filters for target jurisdiction,
    and runs semantic distillation to return a schema-safe DataFrame.
    """
    csv_path = "data/campaign_policies.csv"
    if not os.path.exists(csv_path):
        logger.error(f"Campaign database file {csv_path} not found.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(csv_path)
        required_cols = ["candidate_id", "candidate_name", "party", "jurisdiction", "policy_title", "raw_proposal_text"]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required campaign column: {col}")
                return pd.DataFrame()
                
        # Filter for active jurisdiction
        df_filtered = df[df["jurisdiction"].str.strip().str.lower() == jurisdiction.strip().lower()].copy()
        if df_filtered.empty:
            logger.warning(f"No campaign policies found for target jurisdiction: {jurisdiction}")
            return pd.DataFrame()
            
        logger.info(f"Loaded {len(df_filtered)} campaign proposals for {jurisdiction}. Starting rhetoric distillation...")
        
        abstracts = []
        for _, row in df_filtered.iterrows():
            abstract = distill_campaign_rhetoric(row["raw_proposal_text"])
            abstracts.append(abstract)
            
        df_filtered["distilled_abstract"] = abstracts
        return df_filtered
    except Exception as e:
        logger.error(f"Failed to load or distill campaign policies: {e}")
        return pd.DataFrame()
