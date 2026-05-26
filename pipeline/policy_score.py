"""
Scoring Engine: Calculate net policy impact score and confidence
Uses directional macro impacts to produce bounded policy scores.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Converts directional macro impacts into a bounded net score with confidence.
    """

    def __init__(
        self,
        gdp_weight: float = 0.4,
        unemployment_weight: float = -0.3,
        gdp_mean: Optional[float] = None,
        gdp_std: Optional[float] = None,
        unemp_mean: Optional[float] = None,
        unemp_std: Optional[float] = None,
        macro_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize ScoringEngine with macro signal weights and historical moments.
        Ingests baseline macro data to automatically pre-calculate standard moments
        on year-over-year growth rate scales to prevent unit distortion.
        """
        self.w_gdp = gdp_weight
        self.w_unemp = unemployment_weight
        
        # Determine moments from macro_df if provided, otherwise fallback to passed arguments
        if macro_df is not None and not macro_df.empty:
            # If baseline GDP contains absolute dollar values, convert into fractional percentage growth rates first
            # We group by state boundary to compute YoY percentage change deltas properly per jurisdiction
            gdp_series = macro_df.groupby("state")["gdp"].pct_change() if "gdp" in macro_df.columns else pd.Series()
            
            # If gdp_series contains absolute monetary values, we standard-growth-scale it.
            # (pct_change naturally produces a year-over-year fractional decimal change, e.g. 0.03 for 3% growth).
            self.gdp_mean = float(gdp_series.mean()) if not gdp_series.empty and not gdp_series.isna().all() else 0.0
            self.gdp_std = float(gdp_series.std()) if not gdp_series.empty and not gdp_series.isna().all() else 1.0
            
            # Convert Unemployment rate into first differences (year-over-year rate point change) per state boundary
            unemp_series = macro_df.groupby("state")["unemployment_rate"].diff() if "unemployment_rate" in macro_df.columns else pd.Series()
            self.unemp_mean = float(unemp_series.mean()) if not unemp_series.empty and not unemp_series.isna().all() else 0.0
            self.unemp_std = float(unemp_series.std()) if not unemp_series.empty and not unemp_series.isna().all() else 1.0
        else:
            self.gdp_mean = gdp_mean if gdp_mean is not None else 0.0
            self.gdp_std = gdp_std if gdp_std is not None else 1.0
            self.unemp_mean = unemp_mean if unemp_mean is not None else 0.0
            self.unemp_std = unemp_std if unemp_std is not None else 1.0

        # Replace standard deviations that are extremely close to zero or nan to prevent division by zero
        if pd.isna(self.gdp_mean):
            self.gdp_mean = 0.0
        if pd.isna(self.gdp_std) or abs(self.gdp_std) < 1e-9:
            self.gdp_std = 1.0
        if pd.isna(self.unemp_mean):
            self.unemp_mean = 0.0
        if pd.isna(self.unemp_std) or abs(self.unemp_std) < 1e-9:
            self.unemp_std = 1.0
            
        logger.info(
            f"ScoringEngine initialized: GDP={gdp_weight} (mean={self.gdp_mean:.6f}, std={self.gdp_std:.6f}), "
            f"Unemp={unemployment_weight} (mean={self.unemp_mean:.6f}, std={self.unemp_std:.6f})"
        )
    
    def calculate_net_score(
        self,
        impacts_dict: Dict[str, float],
        analog_bills: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate bounded net policy score and confidence from impacts,
        incorporating similarity variance and fallback status.
        
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
        
        # Dimensionless Z-score standardization layer using historical baseline moments
        z_gdp = (gdp_eff - self.gdp_mean) / (self.gdp_std + 1e-9)
        z_unemp = (unemp_eff - self.unemp_mean) / (self.unemp_std + 1e-9)
        
        # Apply analytical priority weights to standardized standard scores
        gdp_component = self.w_gdp * z_gdp
        unemp_component = self.w_unemp * z_unemp
        
        # Sum weighted dimensionless standard scores
        raw_score = gdp_component + unemp_component
        
        # Clamp to [-1.0, 1.0]
        net_score = max(-1.0, min(1.0, raw_score))
        
        # Structural Scale Denormalization (policy-level scaling applied to inverse-transform)
        policy_scale = 0.01
        true_gdp_delta = ((z_gdp * self.gdp_std) + self.gdp_mean) * policy_scale
        true_unemp_delta = ((z_unemp * self.unemp_std) + self.unemp_mean) * policy_scale
        
        # Enhanced Confidence based on quality, quantity, variance, and fallback status
        sim_scores = [a.get("similarity_score", 0.0) for a in analog_bills]
        variance = float(np.var(sim_scores)) if len(sim_scores) > 1 else 0.0
        
        # Check if fallback logic was triggered
        has_fallback = any(a.get("is_fallback", False) for a in analog_bills)
        
        confidence = self._compute_confidence(avg_sim, num_matched, variance, has_fallback)
        
        logger.info(
            f"Score calculation: GDP_comp={gdp_component:.3f}, "
            f"Unemp_comp={unemp_component:.3f}, Net={net_score:.3f}, Conf={confidence:.3f} "
            f"(Matched={num_matched}, Var={variance:.4f}, Fallback={has_fallback})"
        )
        
        return {
            "net_score": net_score,
            "confidence": confidence,
            "gdp_component": gdp_component,
            "unemployment_component": unemp_component,
            "true_gdp_delta": true_gdp_delta,
            "true_unemployment_delta": true_unemp_delta
        }
    
    def _compute_confidence(self, avg_similarity: float, num_analogs: int, variance: float = 0.0, has_fallback: bool = False) -> float:
        """
        Compute robust confidence score (0 to 1) based on analog quality, quantity, variance, and fallback scope.
        """
        if num_analogs == 0:
            return 0.0

        sim_conf = max(0.0, min(1.0, avg_similarity))
        
        # Quantity confidence saturates gradually as the analog sample grows.
        qty_conf = min(1.0, num_analogs / 5.0)
        
        base_confidence = sim_conf * qty_conf
        
        # Variance penalty: high volatility indicates unstable analog predictions
        variance_penalty = min(0.25, variance * 2.0)
        
        # Scope fallback penalty: federal fallbacks reduce state scoping confidence
        fallback_penalty = 0.20 if has_fallback else 0.0
        
        confidence = base_confidence - variance_penalty - fallback_penalty
        
        # Defensive single-sample check
        if num_analogs == 1:
            confidence *= 0.5
            
        confidence = max(0.0, min(1.0, confidence))
        return confidence
