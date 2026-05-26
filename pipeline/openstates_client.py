"""
Open States API v3 Client
Fetches legislative bills with pagination, rate-limiting, and clean summaries.
Supports dynamic jurisdictional routing without hardcoded query parameters.
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
                if attempt == self.max_retries - 1:
                    break
                time.sleep(delay)
                delay *= self.backoff_factor
        return None

    def fetch_state_bills_bulk(
        self, 
        state_code: str, 
        jurisdiction_name: Optional[str] = None, 
        max_bills: int = 20
    ) -> pd.DataFrame:
        """
        Fetches state legislative bills utilizing paginated requests with rate-limiting resilience.
        Dynamically passes proper casing jurisdiction parameters to the Open States v3 gateway.
        """
        st_lower = state_code.strip().lower()
        endpoint = f"{self.base_url}/bills"
        
        bills = []
        page = 1
        per_page = min(50, max_bills)
        
        state_map = {
            "ca": "California", "tx": "Texas", "ny": "New York", "fl": "Florida",
            "oh": "Ohio", "il": "Illinois", "pa": "Pennsylvania", "mi": "Michigan"
        }
        
        # Determine the clean proper name for Open States v3 mapping
        if jurisdiction_name and jurisdiction_name.strip().lower() != "federal":
            target_jurisdiction = jurisdiction_name.strip().title()
        else:
            target_jurisdiction = state_map.get(st_lower, "United States")
            
        logger.info(f"Starting bulk ingest for {target_jurisdiction} ({state_code.upper()}) [Max: {max_bills} bills]...")
        
        while len(bills) < max_bills:
            # Fully compliant Open States v3 parameters
            params = {
                "jurisdiction": target_jurisdiction,     # Dynamic case-sensitive proper noun string ("Florida", "Ohio", etc.)
                "per_page": per_page,
                "page": page,
                "include": ["sponsorships", "actions"]   # Passes as native list for clean requests array serialization
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
                    if classifications and "executive-signature" in classifications:
                        enacted = True
                        break
                
                # 3. Clean Text Summary and Process
                title = b.get("title", "").strip()
                abstract = b.get("abstract", "") or ""
                bill_text_clean = f"{title} | {abstract}".strip() if abstract else title
                
                introduced_date = b.get("first_action_date") or b.get("created_at")
                session_year = None
                if introduced_date:
                    try:
                        session_year = int(str(introduced_date)[:4])
                    except ValueError:
                        pass
                
                bills.append({
                    "bill_id": b.get("identifier", f"state_{b.get('id')[-8:]}"),
                    "title": title,
                    "introduced_date": introduced_date,
                    "enacted_date": b.get("latest_action_date") if enacted else None,
                    "policy_type": "unknown",
                    "direction": "neutral",
                    "intensity": "low",
                    "sector": "mixed",
                    "state": target_jurisdiction,
                    "level": "state",
                    "text": bill_text_clean,
                    "jurisdiction": target_jurisdiction.lower(),
                    "state_code": state_code.upper(),
                    "session_year": session_year,
                    "enacted": enacted,
                    "sponsor_party": sponsor_party,
                    "bill_text_clean": bill_text_clean,
                    "major_topic": "unknown"
                })
                
            logger.info(f"Retrieved page {page} | Total bills matched: {len(bills)}")
            page += 1
            time.sleep(0.5)  # General rate-limiting grace delay
            
        logger.info(f"Bulk Ingest Completed for {target_jurisdiction}. Total records: {len(bills)}")
        return pd.DataFrame(bills)