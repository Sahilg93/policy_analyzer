import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["state", "year"])

    df["gdp_growth"] = df.groupby("state")["gdp"].pct_change()

    # optional smoothing
    df["unemployment_rate"] = df["unemployment_rate"].astype(float)

    return df