import requests
import pandas as pd


class CongressIngestor:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_bills(self, congress=119, limit=20):

        url = f"https://api.congress.gov/v3/bill/{congress}"

        params = {
            "api_key": self.api_key,
            "limit": limit
        }

        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print("Congress API request failed:", e)
            return pd.DataFrame()

        # Debug safety (only runs when needed)
        if r.status_code != 200:
            print("Congress API error:", r.status_code)
            print(r.text[:500])
            return pd.DataFrame()

        data = r.json()

        bills_raw = []

        # Congress API responses vary in structure
        if isinstance(data, dict):

            # common case 1
            if "bills" in data:
                bills_raw = data["bills"]

                # sometimes nested
                if isinstance(bills_raw, dict) and "bill" in bills_raw:
                    bills_raw = bills_raw["bill"]

            # fallback case 2 (API sometimes wraps differently)
            elif "bill" in data:
                bills_raw = data["bill"]

            else:
                print("Unexpected Congress API shape:", list(data.keys()))
                return pd.DataFrame()

        bills = []

        for b in bills_raw or []:
            bills.append({
                "bill_id": b.get("number"),
                "title": b.get("title"),
                "state": "US",
                "level": "federal",
                "introduced_date": b.get("introducedDate"),
                "status": (b.get("latestAction") or {}).get("text", "") if isinstance(b.get("latestAction"), dict) else "",
                "text": b.get("title", "")
            })

        return pd.DataFrame(bills)
