import pandas as pd

def compute_policy_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # simple baseline model
    df["policy_score"] = (
        df["gdp_growth"] * 0.6
        - df["unemployment_rate"] * 0.4
    )

    return df