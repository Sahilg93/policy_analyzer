"""
Analog Matcher: semantic historical policy similarity search via Ollama.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


class AnalogMatcher:
    """
    Matches current bills to historical analogs using local semantic embeddings.
    """

    def __init__(self, historical_bills_df: Optional[pd.DataFrame] = None):
        """
        Initialize AnalogMatcher with historical bill corpus and load cached vectors if available.
        
        Args:
            historical_bills_df: DataFrame with columns:
                - bill_id, title, introduced_date, enacted_date (optional)
                - policy_type, direction, intensity (from Ollama classification)
        """
        self.history = historical_bills_df.copy() if historical_bills_df is not None else None
        self.embedding_cache = {}
        
        if self.history is None or self.history.empty:
            logger.info("AnalogMatcher initialized with 0 historical bills")
            return

        # Index by bill_id to enable O(1) metadata lookups
        self.history = self.history.set_index("bill_id", drop=False)
        logger.info(f"AnalogMatcher initialized with {len(self.history)} historical bills")
        self._build_embedding_cache()
    
    def find_similar_bills(
        self,
        target_policy: Dict,
        min_threshold: float = 0.75,
        state_to_state_penalty: float = 0.15,
        fed_to_state_penalty: float = 0.30
    ) -> List[Dict]:
        """
        Find every historical analog meeting the semantic similarity threshold,
        applying cross-jurisdiction matching penalties using vectorized NumPy matrix math.
        
        Args:
            target_policy: Dict with keys:
                - "title": str
                - "bill_text_clean": str (optional summary)
                - "jurisdiction": str ('federal' or state name)
                - "policy_type": str (tax, healthcare, education, etc.)
                - "direction": str (expansionary, contractionary, neutral)
                - "intensity": str (low, medium, high)
                - "sector": str (business, households, government, mixed)
                - "level": str (federal or state)
            min_threshold: Minimum similarity score required for inclusion.
            state_to_state_penalty: Mismatch penalty between different states.
            fed_to_state_penalty: Mismatch penalty between federal and state levels.
        
        Returns:
            List of dicts, each containing similarity_score and metadata.
        """
        if self.history is None or self.history.empty:
            logger.warning("No historical bills available")
            return []

        if not self.embedding_cache:
            logger.warning("No historical embeddings available")
            return []
        
        # Use full-text clean summary if available, fallback to title
        target_text = str(target_policy.get("bill_text_clean", "") or target_policy.get("title", "") or "").strip()
        target_jur = str(target_policy.get("jurisdiction", "federal")).strip().lower()
        
        logger.info(
            "Searching for analogs (Target Jur: %s) above %.2f similarity: %s",
            target_jur.upper(),
            min_threshold,
            target_text[:50] or "?"
        )

        target_vector = self._get_embedding(target_text)
        if target_vector is None:
            logger.warning("Unable to embed target bill text")
            return []
        
        threshold = max(0.0, min(1.0, min_threshold))
        
        # 1. Build vectorized array metrics from self.history matching embedding_cache
        bill_ids = []
        embeddings = []
        jurisdictions = []
        
        for bid, hist_vector in self.embedding_cache.items():
            if bid in self.history.index:
                bill_ids.append(bid)
                embeddings.append(hist_vector)
                jurisdictions.append(str(self.history.at[bid, "jurisdiction"]).strip().lower())
                
        if not embeddings:
            logger.warning("No matching cache embeddings found in historical bills")
            return []
            
        M = np.array(embeddings)  # shape: (N, 768)
        
        # 2. Vectorized Cosine Similarity
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            return []
            
        matrix_norms = np.linalg.norm(M, axis=1)
        matrix_norms[matrix_norms == 0] = 1.0  # avoid division by zero
        
        # Calculate raw similarities using BLAS dot product
        raw_similarities = np.dot(M, target_vector) / (matrix_norms * target_norm)
        
        # 3. Vectorized Match Penalties
        # Evaluated IMMEDIATELY to fail fast and trigger threshold checks
        penalties = np.zeros(len(bill_ids))
        jur_array = np.array(jurisdictions)
        
        mismatch_mask = jur_array != target_jur
        fed_mask = mismatch_mask & ((jur_array == "federal") | (target_jur == "federal"))
        state_mask = mismatch_mask & ~fed_mask
        
        penalties[fed_mask] = fed_to_state_penalty
        penalties[state_mask] = state_to_state_penalty
        
        # Subtract matching penalties and clamp immediately
        penalized_similarities = np.clip(raw_similarities - penalties, 0.0, 1.0)
        
        # 4. Fail-fast filtering using numpy selection
        above_threshold_indices = np.where(penalized_similarities >= threshold)[0]
        
        analogs = []
        for index in above_threshold_indices:
            bid = bill_ids[index]
            sim = float(penalized_similarities[index])
            hist_bill = self.history.loc[bid]
            
            # Construct structural analogs dictionary only for items surviving threshold gate
            analog = {
                "bill_id": bid,
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
                "jurisdiction": hist_bill.get("jurisdiction", "federal"),
                "state_code": hist_bill.get("state_code", "US"),
                "session_year": hist_bill.get("session_year"),
                "enacted": hist_bill.get("enacted", True),
                "sponsor_party": hist_bill.get("sponsor_party", "mixed"),
                "bill_text_clean": hist_bill.get("bill_text_clean", ""),
                "major_topic": hist_bill.get("major_topic", "Macroeconomics"),
                "is_fallback": False
            }
            analogs.append(analog)

        # Smart jurisdiction fallback logic:
        # If target is a state, and we find 0 analogs above the threshold,
        # we dynamically fallback to allowing federal-level analogs with a relaxed federal-to-state penalty.
        if target_jur != "federal" and len(analogs) == 0:
            logger.info("No state-level analogs survived threshold filters. Engaging smart federal-fallback logic...")
            # Relax federal-to-state penalty to half (scope leakage is bounded) and lower threshold slightly
            relaxed_fed_penalty = fed_to_state_penalty * 0.5
            penalties = np.zeros(len(bill_ids))
            
            penalties[fed_mask] = relaxed_fed_penalty
            penalties[state_mask] = state_to_state_penalty
            
            penalized_similarities = np.clip(raw_similarities - penalties, 0.0, 1.0)
            relaxed_threshold = max(0.5, threshold - 0.1) # Safe lower boundary
            
            above_threshold_indices = np.where(penalized_similarities >= relaxed_threshold)[0]
            
            for index in above_threshold_indices:
                bid = bill_ids[index]
                sim = float(penalized_similarities[index])
                hist_bill = self.history.loc[bid]
                
                analog = {
                    "bill_id": bid,
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
                    "jurisdiction": hist_bill.get("jurisdiction", "federal"),
                    "state_code": hist_bill.get("state_code", "US"),
                    "session_year": hist_bill.get("session_year"),
                    "enacted": hist_bill.get("enacted", True),
                    "sponsor_party": hist_bill.get("sponsor_party", "mixed"),
                    "bill_text_clean": hist_bill.get("bill_text_clean", ""),
                    "major_topic": hist_bill.get("major_topic", "Macroeconomics"),
                    "is_fallback": True
                }
                analogs.append(analog)
            
        analogs.sort(key=lambda analog: analog["similarity_score"], reverse=True)
        logger.info("Found %d threshold-qualified analogs (vectorized search complete)", len(analogs))
        
        return analogs

    def _build_embedding_cache(self) -> None:
        """
        Pre-compute embeddings for the historical corpus or load from persistent storage.
        """
        import os
        cache_path = "data/processed/historical_embeddings.parquet"
        
        # Try loading persistent parquet vector storage
        if os.path.exists(cache_path):
            try:
                cache_df = pd.read_parquet(cache_path)
                for _, row in cache_df.iterrows():
                    bid = row["bill_id"]
                    vec = row["embedding"]
                    if isinstance(vec, (list, np.ndarray)) and len(vec) > 0:
                        self.embedding_cache[bid] = np.array(vec, dtype=float)
                logger.info(
                    "Loaded %d historical embeddings from persistent cache file: %s",
                    len(self.embedding_cache),
                    cache_path
                )
            except Exception as e:
                logger.warning("Failed to load persistent embedding cache from disk: %s", e)
        
        # Check if there are missing vectors
        missing_rows = []
        for idx, row in self.history.iterrows():
            bill_id = row.get("bill_id", f"hist_{idx}")
            if bill_id not in self.embedding_cache:
                # Use clean summary text first for high-fidelity caching, fallback to title
                text = str(row.get("bill_text_clean", "") or row.get("title", "") or "").strip()
                if text:
                    missing_rows.append((bill_id, text))
        
        if missing_rows:
            logger.info("Fetching %d missing embeddings via local Ollama API...", len(missing_rows))
            newly_fetched = 0
            for bill_id, text in missing_rows:
                embedding = self._get_embedding(text)
                if embedding is not None:
                    self.embedding_cache[bill_id] = embedding
                    newly_fetched += 1
            
            logger.info("Successfully fetched %d/%d new embeddings", newly_fetched, len(missing_rows))
            
            # Serialize the entire cache back to the local persistent cache file
            if newly_fetched > 0:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    # Convert embedding dictionary back to standard format
                    serializable_rows = [
                        {"bill_id": bid, "embedding": vec.tolist()}
                        for bid, vec in self.embedding_cache.items()
                    ]
                    df_cache = pd.DataFrame(serializable_rows)
                    df_cache.to_parquet(cache_path, index=False)
                    logger.info("Successfully serialized and updated persistent embedding cache on disk")
                except Exception as e:
                    logger.warning("Failed to write persistent embedding cache to disk: %s", e)
        else:
            logger.info("All historical embeddings were successfully loaded from disk cache.")

        logger.info(
            "Cached %d/%d historical embeddings",
            len(self.embedding_cache),
            len(self.history)
        )

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Fetch a local Ollama embedding vector for text.
        """
        if not text:
            return None

        payload = {
            "model": OLLAMA_EMBEDDING_MODEL,
            "prompt": text
        }

        try:
            response = requests.post(
                OLLAMA_EMBEDDING_URL,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            vector = response.json().get("embedding")
            if not vector:
                return None
            return np.array(vector, dtype=float)
        except (
            requests.exceptions.RequestException,
            ValueError,
            TypeError
        ) as e:
            logger.warning("Ollama embedding failed: %s", e)
            return None

    def _cosine_similarity(self, v1, v2) -> Optional[float]:
        """
        Compute cosine similarity between two embedding vectors.
        """
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return None
        return float(np.dot(v1, v2) / norm_product)
