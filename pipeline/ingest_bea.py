from pipeline.bea_client import BEAClient
import pandas as pd


def fetch_gdp(api_key):
    client = BEAClient(api_key)

    raw = client.get_gdp_by_state("SAGDP1")

    results = raw["BEAAPI"]["Results"].get("Data", [])

    rows = []
    for r in results:
        rows.append({
            "year": int(r["TimePeriod"]),
            "state": r["GeoName"],
            "gdp": float(r["DataValue"].replace(",", "")) if r["DataValue"] != "(NA)" else None
        })

    return pd.DataFrame(rows)