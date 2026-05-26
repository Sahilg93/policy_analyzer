#!/usr/bin/env python3
"""
Cross-Validation Backtesting Infrastructure
Performs a "leave-one-out" validation test across historical policy bills to
evaluate predictive grounding against real-world macro outcomes.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np

# Ensure project root is in the Python search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from pipeline.build_dataset import load_macro_data, load_historical_bills
from pipeline.policy_matching import AnalogMatcher
from pipeline.policy_impact_linker import OutcomeEngine
from pipeline.policy_score import ScoringEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[backtest] %(levelname)s: %(message)s"
)
logger = logging.getLogger("backtest")


def calculate_actual_macro_outcome(macro_df: pd.DataFrame, state: str, base_year: int) -> tuple[float, float, bool]:
    """
    Calculate the actual real-world macro outcomes (GDP & Unemployment) for a given state
    and year using the same decay-weighted 3-year trailing horizon (T+1, T+2, T+3).
    """
    st = state.strip()
    if st.upper() in ["FEDERAL", "US"]:
        st = "United States"
        
    macro_yr = macro_df[(macro_df["year"] == base_year) & (macro_df["state"] == st)]
    if macro_yr.empty:
        return 0.0, 0.0, False
        
    gdp_base = macro_yr["gdp"].iloc[0]
    unemp_base = macro_yr["unemployment_rate"].iloc[0]
    
    if pd.isna(gdp_base) and pd.isna(unemp_base):
        return 0.0, 0.0, False
        
    decay_weights = {1: 1.0, 2: 0.5, 3: 0.25}
    gdp_deltas = []
    gdp_weights = []
    unemp_deltas = []
    unemp_weights = []
    
    for k, weight in decay_weights.items():
        future_year = base_year + k
        macro_fut = macro_df[(macro_df["year"] == future_year) & (macro_df["state"] == st)]
        if macro_fut.empty:
            continue
            
        gdp_future = macro_fut["gdp"].iloc[0]
        unemp_future = macro_fut["unemployment_rate"].iloc[0]
        
        # GDP Calculation
        if not pd.isna(gdp_base) and not pd.isna(gdp_future) and gdp_base != 0:
            pct = (gdp_future - gdp_base) / gdp_base
            pct = max(-1.0, min(1.0, pct))
            gdp_deltas.append(pct)
            gdp_weights.append(weight)
            
        # Unemployment Calculation
        if not pd.isna(unemp_base) and not pd.isna(unemp_future):
            delta = unemp_future - unemp_base
            delta = max(-1.0, min(1.0, delta))
            unemp_deltas.append(delta)
            unemp_weights.append(weight)
            
    if not gdp_deltas and not unemp_deltas:
        return 0.0, 0.0, False
        
    actual_gdp = 0.0
    if gdp_deltas:
        actual_gdp = sum(d * w for d, w in zip(gdp_deltas, gdp_weights)) / sum(gdp_weights)
        
    actual_unemp = 0.0
    if unemp_deltas:
        actual_unemp = sum(d * w for d, w in zip(unemp_deltas, unemp_weights)) / sum(unemp_weights)
        
    return actual_gdp, actual_unemp, True


def run_backtest(sample_size: int = 15):
    """
    Run the leave-one-out cross-validation backtesting runner.
    """
    logger.info("Initializing Backtest Runner...")
    
    # 1. Load Data
    macro_data = load_macro_data()
    if macro_data is None or macro_data.empty:
        logger.error("No macroeconomic data available for backtesting.")
        return
        
    historical_bills = load_historical_bills()
    if historical_bills is None or historical_bills.empty:
        logger.error("No historical bills available for backtesting.")
        return
        
    # Filter historical bills with valid introduction/enactment dates and states
    valid_bills = []
    for idx, row in historical_bills.iterrows():
        evt_date = row.get("enacted_date") or row.get("introduced_date")
        if not evt_date:
            continue
        try:
            date_parsed = pd.to_datetime(evt_date)
            year = date_parsed.year
            state = row.get("state", "United States")
            
            # Check if actual macro ground truth exists
            _, _, has_truth = calculate_actual_macro_outcome(macro_data, state, year)
            if has_truth:
                valid_bills.append(row)
        except Exception:
            continue
            
    if not valid_bills:
        logger.error("No historical bills have valid dates and corresponding ground truth macro data.")
        return
        
    logger.info(f"Found {len(valid_bills)} historical bills with valid ground truth macro data.")
    
    # Sample subset for validation to avoid hitting the local API excessively
    df_valid = pd.DataFrame(valid_bills)
    df_sample = df_valid.sample(min(sample_size, len(df_valid)), random_state=42)
    logger.info(f"Selected {len(df_sample)} bills for Leave-One-Out validation.")
    
    results = []
    gdp_errors = []
    unemp_errors = []
    
    # 2. Leave-One-Out Loop
    for idx, test_bill in df_sample.iterrows():
        bill_id = test_bill["bill_id"]
        title = test_bill["title"]
        state = test_bill.get("state", "United States")
        
        evt_date = test_bill.get("enacted_date") or test_bill.get("introduced_date")
        date_parsed = pd.to_datetime(evt_date)
        year = date_parsed.year
        
        logger.info(f"Validating Bill {bill_id} | Year: {year} | State: {state} | Title: {title[:50]}...")
        
        # Calculate actual macro ground truth
        actual_gdp, actual_unemp, _ = calculate_actual_macro_outcome(macro_data, state, year)
        
        # Temporarily remove this bill from the historical corpus pool
        corpus_subset = historical_bills[historical_bills["bill_id"] != bill_id].copy()
        
        # Instantiate engines with the leave-one-out subset
        matcher = AnalogMatcher(historical_bills_df=corpus_subset)
        outcome_engine = OutcomeEngine(macro_df=macro_data)
        scoring_engine = ScoringEngine()
        
        # Run test bill through live pipeline
        target_policy = {
            "title": title,
            "policy_type": test_bill.get("policy_type", "unknown"),
            "direction": test_bill.get("direction", "neutral"),
            "intensity": test_bill.get("intensity", "low"),
            "sector": test_bill.get("sector", "unknown"),
            "level": test_bill.get("level", "federal"),
            "macro_relevance": True
        }
        
        analogs = matcher.find_similar_bills(target_policy, min_threshold=0.7)
        
        projected_gdp = 0.0
        projected_unemp = 0.0
        net_score = 0.0
        
        if analogs:
            impacts = outcome_engine.estimate_directional_impacts(analogs, current_bill=target_policy)
            score_res = scoring_engine.calculate_net_score(impacts, analogs)
            
            projected_gdp = impacts.get("gdp_effect", 0.0)
            projected_unemp = impacts.get("unemployment_effect", 0.0)
            net_score = score_res.get("net_score", 0.0)
            
        # Compute Errors
        err_gdp = abs(projected_gdp - actual_gdp)
        err_unemp = abs(projected_unemp - actual_unemp)
        
        gdp_errors.append(err_gdp)
        unemp_errors.append(err_unemp)
        
        results.append({
            "bill_id": bill_id,
            "year": year,
            "actual_gdp": actual_gdp,
            "projected_gdp": projected_gdp,
            "err_gdp": err_gdp,
            "actual_unemp": actual_unemp,
            "projected_unemp": projected_unemp,
            "err_unemp": err_unemp,
            "net_score": net_score,
            "analogs_matched": len(analogs)
        })
        
        logger.info(
            f"Result: GDP Err: {err_gdp:.4f} (Act: {actual_gdp:.4f}, Proj: {projected_gdp:.4f}) | "
            f"Unemp Err: {err_unemp:.4f} (Act: {actual_unemp:.4f}, Proj: {projected_unemp:.4f})"
        )
        
    # 3. Calculate and Print MAE
    mae_gdp = np.mean(gdp_errors)
    mae_unemp = np.mean(unemp_errors)
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION BACKTESTING RESULTS (LEAVE-ONE-OUT)")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results[["bill_id", "year", "actual_gdp", "projected_gdp", "err_gdp", "actual_unemp", "projected_unemp", "err_unemp", "net_score", "analogs_matched"]])
    
    print("\n" + "-"*80)
    print(f"FINAL MEAN ABSOLUTE ERROR (MAE) INDEX METRICS:")
    print(f"  * GDP Growth Effect MAE:             {mae_gdp:.6f}")
    print(f"  * Unemployment Delta Effect MAE:     {mae_unemp:.6f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # If a specific sample size is provided via command line args
    size = 15
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
        except ValueError:
            pass
    run_backtest(sample_size=size)
