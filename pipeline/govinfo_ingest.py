import requests


class GovInfoIngestor:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_bill_text(self, package_id: str):

        url = f"https://api.govinfo.gov/packages/{package_id}/summary"

        params = {"api_key": self.api_key}

        r = requests.get(url, params=params)
        return r.json()