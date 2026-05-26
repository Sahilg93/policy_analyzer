"""
Open States API v3 Client
Fetches legislative bills with pagination, rate-limiting, and clean summaries.
Supports dynamic jurisdictional routing without hardcoded query parameters.
"""
import time
import requests
import pandas as pd
import logging
import threading
import random
from typing import Dict, List, Optional
from pipeline.config import OPENSTATES_API_KEY

logger = logging.getLogger(__name__)

ECONOMIC_KEYWORDS = [
    "appropriation", "tax", "revenue", "budget", "fiscal", "funding",
    "commerce", "workforce", "incentive", "credit", "finance",
    "economic", "subsidy", "bonds", "expenditure"
]


class TokenBucketLimiter:
    def __init__(self, max_capacity: float = 5.0, fill_rate: float = 0.5):
        """
        Thread-safe Token Bucket Limiter.
        max_capacity: float, maximum number of tokens in the bucket.
        fill_rate: float, replenishment rate in tokens per second (0.5 tokens/sec = 30 RPM).
        """
        self.capacity = max_capacity
        self.fill_rate = fill_rate
        self.tokens = max_capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        """Blocks until 1 token is acquired."""
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                
                needed = 1.0 - self.tokens
                sleep_duration = needed / self.fill_rate
            
            time.sleep(sleep_duration)


class OpenStatesClient:
    # Shares a single rate-limiting bucket across all instances of the client globally.
    _limiter = TokenBucketLimiter(max_capacity=5.0, fill_rate=0.5)

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 5, backoff_factor: float = 1.5):
        self.api_key = api_key or OPENSTATES_API_KEY
        self.base_url = "https://v3.openstates.org"
        self.headers = {"X-API-KEY": self.api_key}
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _request_with_retry(self, url: str, params: Dict) -> Optional[Dict]:
        """Performs GET requests with token bucket rate-limiting and defensive retry protection."""
        page = params.get("page", 1)
        
        for attempt in range(self.max_retries):
            # 1. Thread-safe Token acquisition
            self._limiter.acquire()
            
            try:
                r = requests.get(url, headers=self.headers, params=params, timeout=15)
                
                # Check for 429 Rate Limiting
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep_time = float(retry_after)
                        except ValueError:
                            sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                    else:
                        sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                        
                    logger.warning(
                        f"[RATE LIMIT SHIELD] HTTP 429 caught on Page {page}. "
                        f"Backing off for {sleep_time:.2f} seconds..."
                    )
                    time.sleep(sleep_time)
                    continue
                    
                r.raise_for_status()
                return r.json()
                
            except requests.exceptions.RequestException as e:
                sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                logger.error(
                    f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {sleep_time:.2f} seconds..."
                )
                if attempt == self.max_retries - 1:
                    break
                time.sleep(sleep_time)
                
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
        Filters bills upstream using keyword matching and enforces a strict page guardrail limit.
        """
        st_lower = state_code.strip().lower()
        endpoint = f"{self.base_url}/bills"
        
        bills = []
        page = 1
        per_page = min(50, max_bills)
        page_limit = 4  # hard ceiling guardrail to prevent runaway pagination loops
        
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
        
        try:
            while len(bills) < max_bills and page <= page_limit:
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
                        
                    title = b.get("title", "").strip()
                    abstract = b.get("abstract", "") or ""
                    description = b.get("description", "") or ""
                    
                    # Upstream Economic Gate: Only accept bills containing at least one core economic keyword
                    # in their title, abstract, or description (raw text streams) to avoid skipping valid legislation.
                    combined_text_lower = f"{title} {abstract} {description}".lower()
                    if not any(kw in combined_text_lower for kw in ECONOMIC_KEYWORDS):
                        continue
                        
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
                    
                    # 3. Clean Text Summary and Process prioritizing abstract/description
                    if abstract.strip():
                        bill_text_clean = abstract.strip()
                    elif description.strip():
                        bill_text_clean = description.strip()
                    else:
                        bill_text_clean = title
                    
                    bill_text_clean = bill_text_clean.strip()
                    
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
                
        except Exception as err:
            logger.error(f"[GRACEFUL BOUNDARY SHIELD] Unexpected error caught during pagination: {err}. Returning partial dataset.")
            
        logger.info(f"Bulk Ingest Completed for {target_jurisdiction}. Total records: {len(bills)}")
        if not bills:
            logger.warning(f"No state bills matched the keyword filter for {target_jurisdiction} ({state_code.upper()}).")
            return pd.DataFrame(columns=[
                "bill_id", "title", "introduced_date", "enacted_date", "policy_type", 
                "direction", "intensity", "sector", "state", "level", "text", 
                "jurisdiction", "state_code", "session_year", "enacted", "sponsor_party", 
                "bill_text_clean", "major_topic"
            ])
        return pd.DataFrame(bills)