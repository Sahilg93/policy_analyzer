import logging
from typing import Dict, List, Optional
import os
import numpy as np
import pandas as pd
import requests

from pipeline.config import OLLAMA_HOST, OLLAMA_MODEL_EMBED

logger = logging.getLogger(__name__)

OLLAMA_EMBEDDING_URL = f"{OLLAMA_HOST}/api/embeddings"
OLLAMA_EMBEDDING_MODEL = OLLAMA_MODEL_EMBED


class AnalogMatcher:
    """
    Matches current bills to historical analogs using local semantic embeddings.
    """

    def __init__(self, historical_bills_df: Optional[pd.DataFrame] = None, rebuild_embeddings: bool = False):
        """
        Initialize AnalogMatcher with historical bill corpus and load cached vectors if available.
        """
        self.history = historical_bills_df.copy() if historical_bills_df is not None else None
        self.embedding_cache = {}
        
        # Precomputed vectors & matrix properties for O(1) searches
        self.bill_ids = []
        self.embeddings_matrix = None
        self.jurisdictions_array = None
        
        if self.history is None or self.history.empty:
            logger.info("AnalogMatcher initialized with 0 historical bills")
            return

        # Index by bill_id to enable O(1) metadata lookups
        self.history = self.history.set_index("bill_id", drop=False)
        logger.info(f"AnalogMatcher initialized with {len(self.history)} historical bills (Rebuild: {rebuild_embeddings})")
        self._build_embedding_cache(rebuild=rebuild_embeddings)
        self._precompute_matrices()
        
    def _precompute_matrices(self) -> None:
        """Precomputes and caches normalized embedding matrices and jurisdiction lookup arrays."""
        if self.history is None or self.history.empty or not self.embedding_cache:
            self.bill_ids = []
            self.embeddings_matrix = None
            self.jurisdictions_array = None
            return
            
        bill_ids = []
        embeddings = []
        jurisdictions = []
        
        for bid, hist_vector in self.embedding_cache.items():
            if bid in self.history.index:
                bill_ids.append(bid)
                embeddings.append(hist_vector)
                jurisdictions.append(str(self.history.at[bid, "jurisdiction"]).strip().lower())
                
        if not embeddings:
            self.bill_ids = []
            self.embeddings_matrix = None
            self.jurisdictions_array = None
            return
            
        self.bill_ids = bill_ids
        M = np.array(embeddings)  # shape: (N, 768)
        
        # Pre-normalize row vectors to L2 norm = 1.0
        row_norms = np.linalg.norm(M, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        self.embeddings_matrix = M / row_norms
        self.jurisdictions_array = np.array(jurisdictions)
        logger.info(f"Precomputed normalized similarity matrix: shape={self.embeddings_matrix.shape}")
    
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
        """
        if self.history is None or self.history.empty:
            logger.warning("No historical bills available")
            return []

        if self.embeddings_matrix is None or not self.bill_ids:
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
            logger.warning("Unable to embed target bill bill_text_clean")
            return []
            
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            return []
            
        # Target vector normalized to L2 norm = 1.0
        target_vector_normed = target_vector / target_norm
        threshold = max(0.0, min(1.0, min_threshold))
        
        # 1. Cosine similarity via single BLAS matrix-vector dot product
        raw_similarities = np.dot(self.embeddings_matrix, target_vector_normed)
        
        # In-memory Min-Max Scalar to stretch the local distribution across a clean [0, 1] coordinate boundary
        sim_min = raw_similarities.min()
        sim_max = raw_similarities.max()
        norm_similarities = (raw_similarities - sim_min) / (sim_max - sim_min + 1e-9)
        
        # 2. Vectorized Match Penalties
        penalties = np.zeros(len(self.bill_ids))
        mismatch_mask = self.jurisdictions_array != target_jur
        fed_mask = mismatch_mask & ((self.jurisdictions_array == "federal") | (target_jur == "federal"))
        state_mask = mismatch_mask & ~fed_mask
        
        penalties[fed_mask] = fed_to_state_penalty
        penalties[state_mask] = state_to_state_penalty
        
        # Subtract matching penalties and clamp immediately
        penalized_similarities = np.clip(norm_similarities - penalties, 0.0, 1.0)
        
        # 3. Fail-fast filtering using numpy selection
        above_threshold_indices = np.where(penalized_similarities >= threshold)[0]
        
        analogs = []
        for index in above_threshold_indices:
            bid = self.bill_ids[index]
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
                "is_fallback": False
            }
            analogs.append(analog)

        # Smart jurisdiction fallback logic
        if target_jur != "federal" and len(analogs) == 0:
            logger.info("No state-level analogs survived threshold filters. Engaging smart federal-fallback logic...")
            
            # Use raw_similarities directly for fallback bounds
            penalized_similarities = np.clip(raw_similarities - penalties, 0.0, 1.0)
            fallback_floor = 0.55
            
            above_threshold_indices = np.where(penalized_similarities >= fallback_floor)[0]
            
            for index in above_threshold_indices:
                bid = self.bill_ids[index]
                sim = float(penalized_similarities[index])
                hist_bill = self.history.loc[bid]
                
                # Only accept federal-level analogs in this fallback pathway
                if str(hist_bill.get("jurisdiction")).strip().lower() == "federal":
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
            
            if len(analogs) == 0:
                logger.info("Federal fallback analogs failed to clear the strict 0.60 floor threshold. Skipping base score generation.")
                return []
            
        analogs.sort(key=lambda analog: analog["similarity_score"], reverse=True)
        logger.info("Found %d threshold-qualified analogs (vectorized search complete)", len(analogs))
        
        return analogs

    def _build_embedding_cache(self, rebuild: bool = False) -> None:
        """
        Pre-compute embeddings for the historical corpus or load from persistent storage.
        """
        import os
        cache_path = "data/processed/historical_embeddings.parquet"
        
        # If rebuild is True, clear the cache to force a fresh regeneration
        if rebuild and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.info(f"Rebuild requested: Cleared historical embeddings cache file at {cache_path}")
            except Exception as e:
                logger.warning(f"Rebuild requested: Failed to delete cache file {cache_path}: {e}")
        
        # Try loading persistent parquet vector storage
        if not rebuild and os.path.exists(cache_path):
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
        
        # Detect index drift: any keys in embedding_cache that are NOT in self.history represent lingering/deleted bills
        history_ids = set(self.history.index)
        cache_ids = set(self.embedding_cache.keys())
        if cache_ids and (cache_ids != history_ids):
            drift_keys = cache_ids - history_ids
            logger.warning(
                "[INDEX DRIFT DETECTED] Mismatch between history (%d keys) and embedding cache (%d keys). "
                "Lingering/deleted keys in cache: %s. Filtering cache to enforce inner-join intersection.",
                len(history_ids), len(cache_ids), list(drift_keys)[:5]
            )
            # Filter embedding cache to only keys in self.history (inner-join intersection)
            self.embedding_cache = {bid: vec for bid, vec in self.embedding_cache.items() if bid in history_ids}

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
        else:
            logger.info("All historical embeddings were successfully loaded from disk cache.")

        # Final alignment validation: enforce strict 1:1 inner-join intersection to align indexing
        final_history_ids = set(self.history.index)
        final_cache_ids = set(self.embedding_cache.keys())
        if final_history_ids != final_cache_ids:
            intersection = final_history_ids.intersection(final_cache_ids)
            logger.warning(
                "[FINAL INDEX ALIGNMENT] Enforcing strict 1:1 inner-join intersection. "
                "Retaining only %d aligned records.", len(intersection)
            )
            self.history = self.history.loc[list(intersection)]
            self.embedding_cache = {bid: vec for bid, vec in self.embedding_cache.items() if bid in intersection}

        # Always serialize the synchronized embedding cache back to the local persistent cache file
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            serializable_rows = [
                {"bill_id": bid, "embedding": vec.tolist()}
                for bid, vec in self.embedding_cache.items()
            ]
            df_cache = pd.DataFrame(serializable_rows)
            df_cache.to_parquet(cache_path, index=False)
            logger.info("Successfully serialized synchronized embedding cache on disk")
        except Exception as e:
            logger.warning("Failed to write persistent embedding cache to disk: %s", e)

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
