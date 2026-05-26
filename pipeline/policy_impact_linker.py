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
        current_bill: Optional[Dict] = None,
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
                - "level": str, optional (federal or state)
            current_bill: Current bill metadata, including "level" when available.
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
        
        gdp_weighted_terms = []
        unemp_weighted_terms = []
        sim_scores = []
        current_level = self._normalize_level(
            (current_bill or {}).get("level", "federal")
        )
        
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
                
                # Lookup macro data at event year (Base Year)
                macro_yr = self.macro[
                    (self.macro["year"] == evt_year) & 
                    (self.macro["state"] == st)
                ]
                
                if macro_yr.empty:
                    logger.debug(f"No macro data for {st}/{evt_year}")
                    continue
                
                gdp_base = macro_yr["gdp"].iloc[0]
                unemp_base = macro_yr["unemployment_rate"].iloc[0]
                
                # Dynamic Historical Fallback for missing pre-2010 US unemployment rates
                US_HISTORICAL_UNEMP = {
                    1990: 5.6, 1991: 6.8, 1992: 7.5, 1993: 6.9, 1994: 6.1,
                    1995: 5.6, 1996: 5.4, 1997: 4.9, 1998: 4.5, 1999: 4.2,
                    2000: 4.0, 2001: 4.7, 2002: 5.8, 2003: 6.0, 2004: 5.5,
                    2005: 5.1, 2006: 4.6, 2007: 4.6, 2008: 5.8, 2009: 9.3
                }
                
                if st == "United States" and pd.isna(unemp_base) and evt_year in US_HISTORICAL_UNEMP:
                    unemp_base = US_HISTORICAL_UNEMP[evt_year]
                elif pd.isna(unemp_base):
                    # Fallback to United States federal unemployment rate for that year if state-specific is missing
                    macro_us = self.macro[
                        (self.macro["year"] == evt_year) & 
                        (self.macro["state"] == "United States")
                    ]
                    if not macro_us.empty:
                        unemp_base = macro_us["unemployment_rate"].iloc[0]
                    if pd.isna(unemp_base) and evt_year in US_HISTORICAL_UNEMP:
                        unemp_base = US_HISTORICAL_UNEMP[evt_year]
                
                if pd.isna(gdp_base) and pd.isna(unemp_base):
                    logger.debug(f"Base year macro data is entirely missing for {st}/{evt_year}")
                    continue
                
                # We evaluate a 3-year trailing horizon (T+1, T+2, T+3) relative to Base Year T
                # Category-indexed decay profiles map to eliminate scale distortion and support lag horizons
                DECAY_PROFILES = {
                    "Macroeconomics": [1.0, 0.50, 0.25],
                    "Domestic Commerce": [0.25, 1.0, 0.50],
                    "Labor": [0.50, 1.0, 0.25],
                    "Default": [1.0, 0.50, 0.25]
                }
                topic = current_bill.get("major_topic", "Default")
                profile = DECAY_PROFILES.get(topic, DECAY_PROFILES["Default"])
                decay_weights = {1: profile[0], 2: profile[1], 3: profile[2]}
                
                gdp_deltas = []
                gdp_weights = []
                unemp_deltas = []
                unemp_weights = []
                
                for k, weight in decay_weights.items():
                    future_year = evt_year + k
                    macro_fut = self.macro[
                        (self.macro["year"] == future_year) & 
                        (self.macro["state"] == st)
                    ]
                    if macro_fut.empty:
                        continue
                    
                    gdp_future = macro_fut["gdp"].iloc[0]
                    unemp_future = macro_fut["unemployment_rate"].iloc[0]
                    
                    if st == "United States" and pd.isna(unemp_future) and future_year in US_HISTORICAL_UNEMP:
                        unemp_future = US_HISTORICAL_UNEMP[future_year]
                    elif pd.isna(unemp_future):
                        # Fallback to United States federal unemployment rate for that year if state-specific is missing
                        macro_us_fut = self.macro[
                            (self.macro["year"] == future_year) & 
                            (self.macro["state"] == "United States")
                        ]
                        if not macro_us_fut.empty:
                            unemp_future = macro_us_fut["unemployment_rate"].iloc[0]
                        if pd.isna(unemp_future) and future_year in US_HISTORICAL_UNEMP:
                            unemp_future = US_HISTORICAL_UNEMP[future_year]
                    
                    # GDP calculation
                    if not pd.isna(gdp_base) and not pd.isna(gdp_future) and gdp_base != 0:
                        pct = (gdp_future - gdp_base) / gdp_base
                        pct = max(-1.0, min(1.0, pct))
                        gdp_deltas.append(pct)
                        gdp_weights.append(weight)
                    
                    # Unemployment calculation
                    if not pd.isna(unemp_base) and not pd.isna(unemp_future):
                        delta = unemp_future - unemp_base
                        delta = max(-1.0, min(1.0, delta))
                        unemp_deltas.append(delta)
                        unemp_weights.append(weight)
                
                # If we have no valid steps for GDP or unemployment, we skip.
                if not gdp_deltas and not unemp_deltas:
                    logger.debug(f"No future multi-year macro data found for {st} after {evt_year}")
                    continue
                
                # Calculate decay-weighted average for this specific analog bill
                regime = self._get_economic_regime(evt_year)
                
                gdp_pct = 0.0
                if gdp_deltas:
                    raw_gdp_pct = sum(d * w for d, w in zip(gdp_deltas, gdp_weights)) / sum(gdp_weights)
                    # De-bias cyclical recession drag & inflation distortions
                    if regime["is_recession"]:
                        gdp_pct = raw_gdp_pct + 0.015  # Add 1.5% drag correction
                    elif regime["is_high_inflation"]:
                        gdp_pct = raw_gdp_pct * 0.85   # Dampen by 15% to account for stagflation drag
                    else:
                        gdp_pct = raw_gdp_pct
                
                unemp_delta = 0.0
                if unemp_deltas:
                    raw_unemp_delta = sum(d * w for d, w in zip(unemp_deltas, unemp_weights)) / sum(unemp_weights)
                    # De-bias cyclical unemployment spikes during recessions
                    if regime["is_recession"]:
                        unemp_delta = raw_unemp_delta - 0.20  # Subtract 0.20% cyclical spike
                    else:
                        unemp_delta = raw_unemp_delta
                
                # Collect similarity score
                sim = bill.get("similarity_score", 0.5)
                sim = max(0.0, min(1.0, sim))  # Clamp to [0, 1]
                sim = self._apply_scope_penalty(
                    sim,
                    current_level=current_level,
                    analog_level=self._normalize_level(bill.get("level", "federal"))
                )
                sim_scores.append(sim)

                if gdp_deltas:
                    gdp_weighted_terms.append((gdp_pct, sim))

                if unemp_deltas:
                    unemp_weighted_terms.append((unemp_delta, sim))
                
                logger.debug(
                    f"Bill {bill.get('bill_id', '?')}: "
                    f"GDP {gdp_pct:.3f}, Unemp {unemp_delta:.3f}, Sim {sim:.3f} "
                    f"(Regime: Recession={regime['is_recession']}, Inflation={regime['is_high_inflation']})"
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
            change * weight for change, weight in gdp_weighted_terms
        ) / total_weight if gdp_weighted_terms else 0.0
        
        unemp_weighted = sum(
            change * weight for change, weight in unemp_weighted_terms
        ) / total_weight if unemp_weighted_terms else 0.0
        
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

    def _apply_scope_penalty(
        self,
        similarity: float,
        current_level: str,
        analog_level: str
    ) -> float:
        """
        Down-weight state-to-federal analogs to avoid scope leakage.
        """
        if current_level == "state" and analog_level == "federal":
            return similarity * 0.5
        return similarity

    def _normalize_level(self, level: Optional[str]) -> str:
        """Normalize level labels used by different ingestors."""
        value = str(level or "federal").strip().lower()
        if value in {"state", "states", "local"}:
            return "state"
        return "federal"

    def _get_economic_regime(self, year: int) -> dict:
        """Analyze the economic environment of a given historical year."""
        recession_years = {1929, 1930, 1931, 1932, 1933, 1974, 1975, 1980, 1981, 1982, 1990, 1991, 2001, 2008, 2009, 2020}
        inflation_years = {1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 2021, 2022}
        
        return {
            "is_recession": year in recession_years,
            "is_high_inflation": year in inflation_years
        }
