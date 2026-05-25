import pandas as pd


class PolicyNormalizer:
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:

        if df is None or len(df) == 0:
            return pd.DataFrame()

        df = df.copy()

        # ensure required columns exist
        if "title" not in df.columns:
            df["title"] = ""

        if "text" not in df.columns:
            df["text"] = ""

        if "state" not in df.columns:
            df["state"] = "US"

        if "introduced_date" in df.columns:
            df["year"] = pd.to_datetime(
                df["introduced_date"],
                errors="coerce"
            ).dt.year
        else:
            df["year"] = None

        df["text"] = (
            df["title"].fillna("") + " " + df["text"].fillna("")
        ).str.lower()

        return df