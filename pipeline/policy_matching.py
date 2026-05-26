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

        logger.info(f"AnalogMatcher initialized with {len(self.history)} historical bills")
        self._build_embedding_cache()
    
    def find_similar_bills(
        self,
        target_policy: Dict,
        min_threshold: float = 0.75
    ) -> List[Dict]:
        """
        Find every historical analog meeting the semantic similarity threshold.
        
        Args:
            target_policy: Dict with keys:
                - "title": str
                - "policy_type": str (tax, healthcare, education, etc.)
                - "direction": str (expansionary, contractionary, neutral)
                - "intensity": str (low, medium, high)
                - "sector": str (business, households, government, mixed)
                - "level": str (federal or state)
            min_threshold: Minimum similarity score required for inclusion.
        
        Returns:
            List of dicts, each containing:
                - bill_id, title, introduced_date, enacted_date
                - policy_type, direction, intensity, sector, level
                - similarity_score (0 to 1)
        """
        if self.history is None or self.history.empty:
            logger.warning("No historical bills available")
            return []

        if not self.embedding_cache:
            logger.warning("No historical embeddings available")
            return []
        
        target_title = str(target_policy.get("title", "") or "").strip()
        logger.info(
            "Searching for analogs above %.2f similarity: %s",
            min_threshold,
            target_title or "?"
        )

        target_vector = self._get_embedding(target_title)
        if target_vector is None:
            logger.warning("Unable to embed target bill title")
            return []
        
        threshold = max(0.0, min(1.0, min_threshold))
        analogs = []
        
        for idx, hist_bill in self.history.iterrows():
            bill_id = hist_bill.get("bill_id", f"hist_{idx}")
            hist_vector = self.embedding_cache.get(bill_id)
            if hist_vector is None:
                continue

            sim = self._cosine_similarity(target_vector, hist_vector)
            if sim is None or sim < threshold:
                continue
            
            analog = {
                "bill_id": bill_id,
                "title": hist_bill.get("title", ""),
                "introduced_date": hist_bill.get("introduced_date"),
                "enacted_date": hist_bill.get("enacted_date"),
                "policy_type": hist_bill.get("policy_type", "unknown"),
                "direction": hist_bill.get("direction", "neutral"),
                "intensity": hist_bill.get("intensity", "low"),
                "sector": hist_bill.get("sector", "mixed"),
                "similarity_score": sim,
                "state": hist_bill.get("state", "United States"),
                "level": hist_bill.get("level", "federal")
            }
            analogs.append(analog)
        
        analogs.sort(key=lambda analog: analog["similarity_score"], reverse=True)
        logger.info("Found %d threshold-qualified analogs", len(analogs))
        
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
                title = str(row.get("title", "") or "").strip()
                if title:
                    missing_rows.append((bill_id, title))
        
        if missing_rows:
            logger.info("Fetching %d missing embeddings via local Ollama API...", len(missing_rows))
            newly_fetched = 0
            for bill_id, title in missing_rows:
                embedding = self._get_embedding(title)
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
