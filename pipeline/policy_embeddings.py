"""
Analog Matcher: Historical Bill Similarity Search
Finds similar historical bills to a target policy.
"""
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AnalogMatcher:
    """
    Matches current bills to historical analogs based on policy characteristics.
    """

    def __init__(self, historical_bills_df: Optional[pd.DataFrame] = None):
        """
        Initialize AnalogMatcher with historical bill corpus.
        
        Args:
            historical_bills_df: DataFrame with columns:
                - bill_id, title, introduced_date, enacted_date (optional)
                - policy_type, direction, intensity (from Ollama classification)
        """
        self.history = historical_bills_df.copy() if historical_bills_df is not None else None
        
        if self.history is not None:
            logger.info(f"AnalogMatcher initialized with {len(self.history)} historical bills")
    
    def find_similar_bills(
        self,
        target_policy: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find top-K historical analogs to target policy.
        
        Args:
            target_policy: Dict with keys:
                - "title": str
                - "policy_type": str (tax, healthcare, education, etc.)
                - "direction": str (expansionary, contractionary, neutral)
                - "intensity": str (low, medium, high)
                - "sector": str (business, households, government, mixed)
            top_k: Number of analog bills to return
        
        Returns:
            List of dicts, each containing:
                - bill_id, title, introduced_date, enacted_date
                - policy_type, direction, intensity, sector
                - similarity_score (0 to 1)
        """
        if self.history is None or self.history.empty:
            logger.warning("No historical bills available")
            return []
        
        logger.info(f"Searching for {top_k} analogs to: {target_policy.get('title', '?')}")
        
        # Compute similarity scores based on policy attributes
        scores = []
        
        for idx, hist_bill in self.history.iterrows():
            sim = self._compute_similarity(target_policy, hist_bill)
            scores.append((idx, sim))
        
        # Sort by similarity (descending) and take top K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:top_k]]
        
        analogs = []
        for i, idx in enumerate(top_indices):
            bill_row = self.history.loc[idx]
            
            analog = {
                "bill_id": bill_row.get("bill_id", f"hist_{idx}"),
                "title": bill_row.get("title", ""),
                "introduced_date": bill_row.get("introduced_date"),
                "enacted_date": bill_row.get("enacted_date"),
                "policy_type": bill_row.get("policy_type", "unknown"),
                "direction": bill_row.get("direction", "neutral"),
                "intensity": bill_row.get("intensity", "low"),
                "sector": bill_row.get("sector", "mixed"),
                "similarity_score": scores[i][1],
                "state": bill_row.get("state", "United States")
            }
            analogs.append(analog)
        
        logger.info(f"Found {len(analogs)} analogs (avg sim: {np.mean([a['similarity_score'] for a in analogs]):.3f})")
        
        return analogs
    
    def _compute_similarity(self, target: Dict, hist_row) -> float:
        """
        Compute policy similarity (0 to 1) based on attributes.
        
        Exact matches on policy_type, direction, and intensity increase score.
        """
        base_score = 0.0
        
        # Policy type match (40% weight)
        if target.get("policy_type") == hist_row.get("policy_type"):
            base_score += 0.4
        elif target.get("policy_type") and hist_row.get("policy_type"):
            # Same family but not identical
            base_score += 0.15
        
        # Direction match (30% weight)
        if target.get("direction") == hist_row.get("direction"):
            base_score += 0.3
        
        # Intensity match (20% weight)
        if target.get("intensity") == hist_row.get("intensity"):
            base_score += 0.2
        
        # Sector match (10% weight)
        if target.get("sector") == hist_row.get("sector"):
            base_score += 0.1
        
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, base_score))
        
        return similarity