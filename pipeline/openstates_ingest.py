import requests
import pandas as pd


class OpenStatesIngestor:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_bills(self, state="ohio", limit=20):

        url = "https://v3.openstates.org/bills"

        headers = {
            "X-API-KEY": self.api_key
        }

        params = {
            "jurisdiction": state,
            "per_page": limit
        }

        r = requests.get(url, headers=headers, params=params)

        if r.status_code != 200:
            print("OpenStates error:", r.status_code, r.text[:300])
            return pd.DataFrame()

        data = r.json()

        bills = []

        for b in data.get("results", []):
            bills.append({
                "bill_id": b.get("identifier"),
                "title": b.get("title"),
                "state": state,
                "level": "state",
                "introduced_date": b.get("created_at"),
                "status": b.get("classification", []),
                "text": b.get("title", "")
            })

        return pd.DataFrame(bills)