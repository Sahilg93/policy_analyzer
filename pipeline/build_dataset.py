from pipeline.ingest_bea import fetch_gdp
from pipeline.bls_client import BLSClient
from pipeline.policy_ingest import PolicyIngestor
from pipeline.policy_mapper import PolicyMapper
from pipeline.policy_impact_linker import PolicyImpactLinker
from pipeline.congress_ingest import CongressIngestor
from pipeline.openstates_ingest import OpenStatesIngestor
import pandas as pd
import numpy as np
import requests
import json


API_KEY = "84DF9CAA-34FB-4555-BDF0-130FEA791DA2"

def interpret_bill_ollama(title: str):
    """
    Local LLM-based policy interpreter using Ollama.
    """

    prompt = f"""
You are a policy classification system.

Analyze the US congressional bill title and return ONLY valid JSON.

Title: {title}

JSON schema:
{{
  "policy_type": "tax | healthcare | education | regulation | spending | trade | other",
  "direction": "expansionary | contractionary | neutral",
  "intensity": "low | medium | high",
  "sector": "business | households | government | mixed"
}}
"""

    try:
        r = requests.post(
        
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )

        text = r.json().get("response", "")

        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("No JSON found")

        return json.loads(text[start:end])

    except Exception:
        return {
            "policy_type": "unknown",
            "direction": "neutral",
            "intensity": "low",
            "sector": "unknown"
        }


def build():

    # ----------------------
    # BEA GDP
    # ----------------------
    gdp = fetch_gdp(API_KEY)

    if gdp is None or gdp.empty:
        print("BEA returned empty dataset")
        return None

    # ----------------------
    # BLS UNEMPLOYMENT
    # ----------------------
    bls = BLSClient(api_key="71ca07a939aa4e71a82ae2f88ac8ad1e")

    unemp = bls.fetch_series(
        series_ids=["LNS14000000"],
        start_year=2010,
        end_year=2025
    )

    unemp = unemp.groupby("year")["value"].mean().reset_index()
    unemp.columns = ["year", "unemployment_rate"]

    # ----------------------
    # MERGE MACRO DATA
    # ----------------------
    df = gdp.merge(unemp, on="year", how="left")
    df = df.copy()

    df = df.sort_values(["state", "year"])
    df["gdp_growth"] = df.groupby("state")["gdp"].pct_change()

    print("Pipeline built:", df.shape)

    # ======================================================
    # POLICY CSV LAYER (optional local file)
    # ======================================================
    try:
        ingestor = PolicyIngestor()
        mapper = PolicyMapper()
        linker = PolicyImpactLinker()

        policy = ingestor.load_from_csv("data/raw/policies.csv")
        policy = mapper.map_policy(policy)

        df = linker.merge(policy, df)

        print("Policy CSV layer applied:", df.shape)

    except Exception:
        print("Policy CSV layer skipped")

    # ======================================================
    # FEDERAL + STATE POLICY INGESTION
    # ======================================================

    congress = CongressIngestor(
        "fodYfBmI4cxpLigjhMdpY8jfEqUhbeSJKHKAKq4U"
    ).fetch_bills()

    openstates = OpenStatesIngestor(
        "c9426a2c-debd-4870-9304-616b5e463ea3"
    ).fetch_bills()

    # safety
    if congress is None:
        congress = pd.DataFrame()

    if openstates is None:
        openstates = pd.DataFrame()

    print("Congress:", congress.shape)
    print("OpenStates:", openstates.shape)

    # ----------------------
    # AI POLICY ENRICHMENT (LOCAL OLLAMA)
    # ----------------------

    if congress is not None and not congress.empty:

        congress_sample = congress.head(10).copy()

        ai_outputs = congress_sample["title"].apply(interpret_bill_ollama)

        ai_df = pd.json_normalize(ai_outputs)

        congress_sample = pd.concat(
            [congress_sample.reset_index(drop=True), ai_df],
            axis=1
        )

        congress = congress_sample

        print("AI enrichment complete:", congress.shape)
        print("\n=== AI POLICY OUTPUT SAMPLE ===")

        print(congress[[
            "title",
            "policy_type",
            "direction",
            "intensity",
            "sector"
        ]].head(20))

    else:
        print("Skipping AI layer (no congress data)")

    # ----------------------
    # TAG SOURCES
    # ----------------------
    if not congress.empty:
        congress["source"] = "federal"
        congress["state"] = "federal"

    if not openstates.empty:
        openstates["source"] = "state"

    policy_all = pd.concat(
        [congress, openstates],
        ignore_index=True
    )

    if policy_all.empty:
        print("No policy data available")
    else:

        # ----------------------
        # FEDERAL POLICY INDEX (REPLACEMENT)
        # ----------------------

        if not congress.empty:

            congress["introduced_date"] = pd.to_datetime(
                congress.get("introduced_date"),
                errors="coerce"
            )

            congress["year"] = congress["introduced_date"].dt.year

            congress = congress.dropna(subset=["year"])
            congress["year"] = congress["year"].astype(int)

            policy_agg = congress.groupby("year").agg(
                federal_policy_count=("bill_id", "count"),
                expansionary_policy=("direction", lambda x: (x == "expansionary").sum()),
                contractionary_policy=("direction", lambda x: (x == "contractionary").sum()),
                high_intensity_policy=("intensity", lambda x: (x == "high").sum())
            ).reset_index()

            df = df.merge(policy_agg, on="year", how="left")

            print("Federal policy index merged:", df.shape)

        else:
            print("No congressional policy data available")

    # ----------------------
    # FINAL OUTPUT
    # ----------------------
    print("\n=== FINAL DATA SAMPLE ===")
    print(df.head(10))
    print("\nColumns:", df.columns.tolist())
    print("\nShape:", df.shape)

    return df


if __name__ == "__main__":
    build()