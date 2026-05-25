import requests


class BEAClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://apps.bea.gov/api/data"

    # -----------------------------
    # STEP 1: discover LineCodes
    def get_linecodes(self, table_name="SAGDP1"):
        params = {
            "UserID": self.api_key,
            "method": "GetParameterValues",
            "datasetname": "Regional",
            "ParameterName": "LineCode",
            "TableName": table_name,
            "ResultFormat": "JSON"
        }

        r = requests.get(self.base_url, params=params)



        return r.json()

    # -----------------------------
    # STEP 2: extract best LineCode
    # -----------------------------
    def pick_linecode(self, meta_json):
        results = meta_json.get("BEAAPI", {}).get("Results", {})

        values = None

        if "ParamValue" in results:
            values = results["ParamValue"]
        elif "ParameterValues" in results:
            values = results["ParameterValues"].get("ParamValue", [])
        elif isinstance(results, list):
            values = results
        else:
            values = []

        if not values:
            return None

        # FIRST: search GDP explicitly
        for v in values:
            desc = str(v.get("Desc", "")).lower()
            if "gross domestic product" in desc or "gdp" in desc:
                return v.get("Key") or v.get("LineCode")

        # SECOND: fallback
        return values[0].get("Key") or values[0].get("LineCode")
    # -----------------------------
    # STEP 3: fetch GDP
    # -----------------------------
    def get_gdp_by_state(self, table_name="SAGDP1"):
        meta = self.get_linecodes(table_name)
        linecode = self.pick_linecode(meta)

        if not linecode:
            raise ValueError(f"No valid LineCode found for table {table_name}")

        params = {
            "UserID": self.api_key,
            "method": "GetData",
            "datasetname": "Regional",
            "TableName": table_name,
            "GeoFips": "STATE",
            "LineCode": linecode,
            "Year": "ALL",
            "ResultFormat": "JSON"
        }

        r = requests.get(self.base_url, params=params)
        data = r.json()

        # -----------------------------
        # HARD VALIDATION (IMPORTANT)
        # -----------------------------
        if "BEAAPI" not in data:
            raise ValueError(f"Invalid BEA response: {data}")

        if "Error" in data["BEAAPI"].get("Results", {}):
            raise ValueError(f"BEA API Error: {data}")

        if "Data" not in data["BEAAPI"].get("Results", {}):
            return {"BEAAPI": {"Results": {"Data": []}}}

        return data