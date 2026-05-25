import pandas as pd


class PolicyImpactLinker:
    """
    Joins policy events with macroeconomic state-year dataset.
    """

    def merge(self, policy_df: pd.DataFrame, macro_df: pd.DataFrame):

        # ensure merge keys exist
        if "state" not in macro_df.columns:
            macro_df["state"] = "US"

        if "state" not in policy_df.columns:
            policy_df["state"] = "US"

        if "year" not in macro_df.columns or "year" not in policy_df.columns:
            raise ValueError("Both datasets must contain 'year' column")

        merged = policy_df.merge(
            macro_df,
            on=["state", "year"],
            how="left"
        )

        return merged