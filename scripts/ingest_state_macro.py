"""
State Macroeconomic Data Ingestor
Downloads real GDP and Unemployment indicators directly from BEA and BLS APIs,
and compiles them into data/processed/policy_dataset.parquet.
"""
import os
import requests
import pandas as pd
import numpy as np

BEA_API_KEY = "84DF9CAA-34FB-4555-BDF0-130FEA791DA2"
BLS_API_KEY = "71ca07a939aa4e71a82ae2f88ac8ad1e"


def fetch_real_state_gdp(api_key: str) -> pd.DataFrame:
    """Fetch real state-level GDP (Annual) from the BEA Regional GDP API."""
    url = "https://apps.bea.gov/api/data"
    params = {
        "UserID": api_key,
        "Method": "GetData",
        "DataSetName": "Regional",
        "TableName": "SAGDP2",  # Annual GDP by State
        "LineCode": "1",        # All industries total
        "GeoFips": "STATE",     # All states
        "Year": "2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025",
        "ResultFormat": "json"
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        results = data.get("BEAAPI", {}).get("Results", {}).get("Data", [])
        rows = []
        for d in results:
            val_str = d.get("DataValue")
            if val_str:
                val_str = val_str.replace(",", "")
            
            try:
                gdp_val = float(val_str) if val_str else None
            except ValueError:
                gdp_val = None
                
            rows.append({
                "year": int(d.get("TimePeriod")),
                "state": d.get("GeoName"),
                "gdp": gdp_val
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error fetching BEA GDP: {e}")
        return pd.DataFrame()


def fetch_real_state_unemployment(api_key: str, state_fips_map: dict) -> pd.DataFrame:
    """Fetch real state unemployment rate histories from BLS LAUS API."""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-Type": "application/json"}
    
    # Compile BLS LAUS timeseries IDs: LASST<FIPS>00000000003 (Seasonally Adjusted Rate)
    series_ids = []
    id_to_state = {}
    for fips, state in state_fips_map.items():
        series_id = f"LASST{fips:02d}00000000003"
        series_ids.append(series_id)
        id_to_state[series_id] = state
        
    payload = {
        "seriesid": series_ids,
        "startyear": "2010",
        "endyear": "2025",
        "registrationkey": api_key
    }
    
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        rows = []
        for series in data.get("Results", {}).get("series", []):
            series_id = series.get("seriesID")
            state = id_to_state.get(series_id)
            for d in series.get("data", []):
                year = int(d.get("year"))
                try:
                    val = float(d.get("value"))
                except ValueError:
                    val = None
                
                if val is not None:
                    rows.append({
                        "year": year,
                        "state": state,
                        "unemployment_rate": val
                    })
        # BLS provides monthly data; aggregate to annual averages
        df = pd.DataFrame(rows)
        if not df.empty:
            return df.groupby(["year", "state"])["unemployment_rate"].mean().reset_index()
        return df
    except Exception as e:
        print(f"Error fetching BLS Unemployment: {e}")
        return pd.DataFrame()


def integrate_real_macro_data():
    """Compiles BEA state GDP and BLS state unemployment data into the Parquet store."""
    parquet_path = "data/processed/policy_dataset.parquet"
    
    state_fips = {
        6: "California", 48: "Texas", 36: "New York", 12: "Florida",
        39: "Ohio", 17: "Illinois", 42: "Pennsylvania", 26: "Michigan"
    }
    
    print("Downloading live state GDP from BEA...")
    df_gdp = fetch_real_state_gdp(BEA_API_KEY)
    
    print("Downloading live state Unemployment from BLS...")
    df_unemp = fetch_real_state_unemployment(BLS_API_KEY, state_fips)
    
    if df_gdp.empty and df_unemp.empty:
        print("Data download failed. Check API credentials or network connections.")
        return
        
    # Merge GDP and Unemployment
    if not df_gdp.empty and not df_unemp.empty:
        df_merged = pd.merge(df_gdp, df_unemp, on=["year", "state"], how="outer")
    elif not df_gdp.empty:
        df_merged = df_gdp.copy()
        df_merged["unemployment_rate"] = np.nan
    else:
        df_merged = df_unemp.copy()
        df_merged["gdp"] = np.nan
        
    # Integrate into policy_dataset.parquet
    if os.path.exists(parquet_path):
        df_old = pd.read_parquet(parquet_path)
        # Exclude matching state records in old DB to prevent duplicates
        states_to_exclude = list(state_fips.values())
        df_old_clean = df_old[~df_old["state"].isin(states_to_exclude)]
        df_final = pd.concat([df_old_clean, df_merged], ignore_index=True)
    else:
        df_final = df_merged
        
    # Re-calculate state-level GDP growth
    df_final = df_final.sort_values(by=["state", "year"]).reset_index(drop=True)
    for state in df_final["state"].unique():
        mask = df_final["state"] == state
        df_state = df_final[mask]
        gdp_series = df_state["gdp"]
        gdp_prev = gdp_series.shift(1)
        
        # Calculate growth curve
        gdp_growth = (gdp_series - gdp_prev) / gdp_prev
        df_final.loc[mask, "gdp_growth"] = gdp_growth
        
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df_final.to_parquet(parquet_path, index=False)
    print(f"Data Integration Complete! Updated database shape: {df_final.shape}")


if __name__ == "__main__":
    integrate_real_macro_data()
