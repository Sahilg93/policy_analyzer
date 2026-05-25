"""
Outcome Engine: Historical Analog Impact Analysis
Deterministically calculates directional macro impacts based on similar historical bills.
"""
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class OutcomeEngine:
    """
    Loads macro data and computes directional impacts for bills based on historical analogs.
    """

    def __init__(self, macro_df: Optional[pd.DataFrame] = None, macro_path: Optional[str] = None):
        """
        Initialize OutcomeEngine with macro data.
        
        Args:
            macro_df: Pre-loaded DataFrame with columns [state, year, gdp, unemployment_rate]
            macro_path: Path to parquet file if macro_df not provided
        """
        self.macro = None
        
        if macro_df is not None:
            self.macro = macro_df.copy()
        elif macro_path:
            try:
                self.macro = pd.read_parquet(macro_path)
                logger.info(f"Loaded macro data from {macro_path}: shape={self.macro.shape}")
            except Exception as e:
                logger.error(f"Failed to load macro data from {macro_path}: {e}")
                self.macro = None
        
        if self.macro is not None:
            self._validate_macro_structure()
    
    def _validate_macro_structure(self):
        """Ensure macro data has required columns."""
        required = {"state", "year", "gdp", "unemployment_rate"}
        missing = required - set(self.macro.columns)
        if missing:
            logger.warning(f"Macro data missing columns: {missing}")
    
    def estimate_directional_impacts(
        self,
        analog_bills: List[Dict],
        window_months: int = 12
    ) -> Dict[str, float]:
        """
        Calculate weighted directional impacts from historical analog bills.
        
        Args:
            analog_bills: List of dicts with keys:
                - "bill_id": str
                - "title": str
                - "introduced_date": datetime or str (YYYY-MM-DD)
                - "enacted_date": datetime or str (YYYY-MM-DD), optional
                - "similarity_score": float (0-1)
                - "state": str, optional (defaults to "US" national)
            window_months: Months to measure macro change after bill date (default 12)
        
        Returns:
            Dict with keys:
                - "gdp_effect": float (-1 to 1)
                - "unemployment_effect": float (-1 to 1)
                - "num_analogs_matched": int
                - "avg_similarity": float
        """
        if self.macro is None or self.macro.empty:
            logger.warning("No macro data available; returning neutral impacts")
            return {
                "gdp_effect": 0.0,
                "unemployment_effect": 0.0,
                "num_analogs_matched": 0,
                "avg_similarity": 0.0
            }
        
        if not analog_bills:
            logger.warning("No analog bills provided")
            return {
                "gdp_effect": 0.0,
                "unemployment_effect": 0.0,
                "num_analogs_matched": 0,
                "avg_similarity": 0.0
            }
        
        logger.info(f"Estimating impacts from {len(analog_bills)} analogs")
        
        gdp_changes = []
        unemp_changes = []
        sim_scores = []
        
        for bill in analog_bills:
            try:
                # Extract date (prefer enacted, fallback to introduced)
                evt_date = bill.get("enacted_date") or bill.get("introduced_date")
                if not evt_date:
                    logger.debug(f"Skipping bill {bill.get('bill_id', '?')}: no date")
                    continue
                
                # Parse date if string
                if isinstance(evt_date, str):
                    evt_date = pd.to_datetime(evt_date)
                elif not isinstance(evt_date, (datetime, pd.Timestamp)):
                    continue
                
                evt_year = evt_date.year
                
                # Determine state (default to United States for federal level)
                st = bill.get("state", "United States").strip()
                if st.upper() in ["FEDERAL", "US"]:
                    st = "United States"
                
                # Lookup macro data at event year
                macro_yr = self.macro[
                    (self.macro["year"] == evt_year) & 
                    (self.macro["state"] == st)
                ]
                
                if macro_yr.empty:
                    logger.debug(f"No macro data for {st}/{evt_year}")
                    continue
                
                gdp_base = macro_yr["gdp"].iloc[0]
                unemp_base = macro_yr["unemployment_rate"].iloc[0]
                
                # Lookup macro data 12 months later (next year)
                future_year = evt_year + 1
                macro_fut = self.macro[
                    (self.macro["year"] == future_year) & 
                    (self.macro["state"] == st)
                ]
                
                if macro_fut.empty:
                    logger.debug(f"No future macro data for {st}/{future_year}")
                    continue
                
                gdp_future = macro_fut["gdp"].iloc[0]
                unemp_future = macro_fut["unemployment_rate"].iloc[0]
                
                # Handle missing values
                if pd.isna(gdp_base) or pd.isna(gdp_future):
                    logger.debug(f"Missing GDP data for {st} {evt_year}->{future_year}")
                    continue
                
                # Calculate percentage change (bounded -1 to 1)
                gdp_pct = 0.0
                if gdp_base != 0:
                    gdp_pct = (gdp_future - gdp_base) / gdp_base
                    gdp_pct = max(-1.0, min(1.0, gdp_pct))
                    gdp_changes.append(gdp_pct)
                
                # Calculate absolute change in unemployment (bounded -1 to 1)
                unemp_delta = 0.0
                if not pd.isna(unemp_base) and not pd.isna(unemp_future):
                    unemp_delta = unemp_future - unemp_base
                    # Clamp to [-1, 1] range
                    unemp_delta = max(-1.0, min(1.0, unemp_delta))
                    unemp_changes.append(unemp_delta)
                
                # Collect similarity score
                sim = bill.get("similarity_score", 0.5)
                sim = max(0.0, min(1.0, sim))  # Clamp to [0, 1]
                sim_scores.append(sim)
                
                logger.debug(
                    f"Bill {bill.get('bill_id', '?')}: "
                    f"GDP {gdp_pct:.3f}, Unemp {unemp_delta:.3f}, Sim {sim:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Error processing analog bill: {e}")
                continue
        
        # Aggregate using weighted averages
        num_matched = len(sim_scores)
        
        if num_matched == 0:
            logger.warning("No analogs could be matched to macro data")
            return {
                "gdp_effect": 0.0,
                "unemployment_effect": 0.0,
                "num_analogs_matched": 0,
                "avg_similarity": 0.0
            }
        
        # Weighted average by similarity score
        total_weight = sum(sim_scores)
        
        gdp_weighted = sum(
            g * s for g, s in zip(gdp_changes, sim_scores[:len(gdp_changes)])
        ) / total_weight if gdp_changes else 0.0
        
        unemp_weighted = sum(
            u * s for u, s in zip(unemp_changes, sim_scores[:len(unemp_changes)])
        ) / total_weight if unemp_changes else 0.0
        
        avg_sim = sum(sim_scores) / len(sim_scores)
        
        # Clamp final effects to [-1, 1]
        gdp_weighted = max(-1.0, min(1.0, gdp_weighted))
        unemp_weighted = max(-1.0, min(1.0, unemp_weighted))
        
        result = {
            "gdp_effect": gdp_weighted,
            "unemployment_effect": unemp_weighted,
            "num_analogs_matched": num_matched,
            "avg_similarity": avg_sim
        }
        
        logger.info(
            f"Impact estimation complete: "
            f"GDP={gdp_weighted:.3f}, Unemp={unemp_weighted:.3f}, "
            f"Matched={num_matched}, AvgSim={avg_sim:.3f}"
        )
        
        return result