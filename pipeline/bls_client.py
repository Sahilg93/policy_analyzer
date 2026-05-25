import requests
import pandas as pd


class BLSClient:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def fetch_series(self, series_ids, start_year=2010, end_year=2025):

        payload = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }

        if self.api_key:
            payload["registrationkey"] = self.api_key

        r = requests.post(self.base_url, json=payload)
        data = r.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(f"BLS API error: {data}")

        results = data.get("Results", {})
        series_list = results.get("series", [])

        if not series_list:
            raise ValueError(f"No BLS series returned: {data}")

        rows = []

        for series in series_list:
            sid = series.get("seriesID")

            for item in series.get("data", []):
                raw_value = item.get("value", "-")

                if raw_value in ["-", ""]:
                    continue

                try:
                    value = float(raw_value)
                except:
                    continue

                rows.append({
                    "series": sid,
                    "year": int(item["year"]),
                    "value": value
                })

        return pd.DataFrame(rows)