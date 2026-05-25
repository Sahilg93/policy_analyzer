import pandas as pd


def normalize_bea_response(bea_json, metric_name="gdp"):
    results = bea_json.get("BEAAPI", {}).get("Results", {})

    # 🚨 HANDLE API ERROR CASE
    if "Data" not in results:
        print("BEA API returned no Data:", results.get("Error", results))
        return pd.DataFrame(columns=["state", "year", "metric", "value"])

    data = results["Data"]

    rows = []

    for item in data:
        state = item.get("GeoName")
        year = item.get("TimePeriod")
        value = item.get("DataValue")

        if not state or not year or not value:
            continue

        try:
            value = float(str(value).replace(",", ""))
            year = int(year)
        except:
            continue

        rows.append({
            "state": state,
            "year": year,
            "metric": metric_name,
            "value": value
        })

    return pd.DataFrame(rows)