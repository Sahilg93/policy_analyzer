import pandas as pd
from datetime import datetime


class PolicyIngestor:
    def __init__(self):
        pass

    def load_from_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return self._normalize(df)

    def load_from_json(self, path: str) -> pd.DataFrame:
        df = pd.read_json(path)
        return self._normalize(df)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

        df["state"] = df.get("state", "US")

        df["text"] = (
            df.get("text", "") + " " + df.get("title", "")
        ).str.lower()

        return df[[
            "bill_id",
            "title",
            "date",
            "year",
            "state",
            "text"
        ]]