"""
Scoring Engine: Calculate net policy impact score and confidence
Uses directional macro impacts to produce bounded policy scores.
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Converts directional macro impacts into a bounded net score with confidence.
    """

    def __init__(self, gdp_weight: float = 0.4, unemployment_weight: float = -0.3):
        """
        Initialize ScoringEngine with macro signal weights.
        
        Args:
            gdp_weight: Relative importance of GDP effect (positive = good)
            unemployment_weight: Relative importance of unemployment effect
                                (negative = lower unemployment is better)
        """
        self.w_gdp = gdp_weight
        self.w_unemp = unemployment_weight
        logger.info(f"ScoringEngine initialized: GDP={gdp_weight}, Unemp={unemployment_weight}")
    
    def calculate_net_score(
        self,
        impacts_dict: Dict[str, float],
        analog_bills: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate bounded net policy score and confidence from impacts.
        
        Args:
            impacts_dict: Dict with keys:
                - "gdp_effect": float (-1 to 1)
                - "unemployment_effect": float (-1 to 1)
                - "num_analogs_matched": int
                - "avg_similarity": float (0 to 1)
            analog_bills: List of analog bills (for confidence calculation)
        
        Returns:
            Dict with keys:
                - "net_score": float (-1.0 to 1.0)
                - "confidence": float (0.0 to 1.0)
                - "gdp_component": float
                - "unemployment_component": float
        """
        gdp_eff = impacts_dict.get("gdp_effect", 0.0)
        unemp_eff = impacts_dict.get("unemployment_effect", 0.0)
        avg_sim = impacts_dict.get("avg_similarity", 0.0)
        num_matched = impacts_dict.get("num_analogs_matched", 0)
        
        # Apply analytical weights to macro signals
        gdp_component = self.w_gdp * gdp_eff
        unemp_component = self.w_unemp * unemp_eff
        
        # Sum weighted components
        raw_score = gdp_component + unemp_component
        
        # Clamp to [-1.0, 1.0]
        net_score = max(-1.0, min(1.0, raw_score))
        
        # Confidence based on:
        # 1. Average similarity of matched analogs
        # 2. Number of analogs (more = more confident, with diminishing returns)
        confidence = self._compute_confidence(avg_sim, num_matched)
        
        logger.info(
            f"Score calculation: GDP_comp={gdp_component:.3f}, "
            f"Unemp_comp={unemp_component:.3f}, Net={net_score:.3f}, Conf={confidence:.3f}"
        )
        
        return {
            "net_score": net_score,
            "confidence": confidence,
            "gdp_component": gdp_component,
            "unemployment_component": unemp_component
        }
    
    def _compute_confidence(self, avg_similarity: float, num_analogs: int) -> float:
        """
        Compute confidence score (0 to 1) based on analog quality and quantity.
        
        Confidence = similarity_score * saturation_factor
        where saturation_factor increases with num_analogs (sublinearly).
        """
        # Similarity directly contributes to confidence
        sim_conf = max(0.0, min(1.0, avg_similarity))
        
        # Number of analogs contributes via log scale (diminishing returns)
        # 1 analog = 0.5, 5 analogs = 0.85, 20+ analogs = ~1.0
        if num_analogs == 0:
            qty_conf = 0.0
        else:
            qty_conf = min(1.0, (1.0 + num_analogs) / (1.0 + num_analogs + 5.0))
        
        # Combine: both similarity and quantity matter
        confidence = (sim_conf + qty_conf) / 2.0
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence