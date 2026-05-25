import pandas as pd


class PolicyMapper:
    """
    Converts policy text into structured economic exposure signals.
    """

    def __init__(self):
        # extremely simple keyword model (upgrade later to embeddings)
        self.rules = {
            "tax": ["tax", "irs", "revenue", "income tax"],
            "labor": ["wage", "labor", "minimum wage", "union"],
            "healthcare": ["health", "medicaid", "insurance"],
            "infrastructure": ["infrastructure", "road", "bridge", "transport"],
            "tech": ["ai", "technology", "semiconductor", "chip", "data"],
            "education": ["school", "education", "student", "loan"],
            "energy": ["oil", "gas", "energy", "electricity", "renewable"]
        }

    def map_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for sector, keywords in self.rules.items():
            df[sector + "_exposure"] = df["text"].apply(
                lambda x: sum(1 for k in keywords if k in x)
            )

        # normalize exposure score
        exposure_cols = [c for c in df.columns if c.endswith("_exposure")]
        df["total_exposure"] = df[exposure_cols].sum(axis=1)

        return df