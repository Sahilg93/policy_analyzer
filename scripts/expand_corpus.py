"""
Expand the historical policy analog corpus from CAP US Public Laws data.

This utility downloads the official Comparative Agendas Project public laws
CSV, filters it to modern economically relevant laws, maps CAP topic codes
onto the pipeline taxonomy, and writes data/policy_events.csv.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


CAP_PUBLIC_LAWS_URL = (
    "https://www.comparativeagendas.net/files/cap_us_public_laws_v2023.csv"
)
CAP_PUBLIC_LAWS_FALLBACK_URL = (
    "https://minio.la.utexas.edu/compagendas/datasetfiles/"
    "US-Legislative-public_laws_20.1_8.csv"
)
OUTPUT_PATH = Path("data/policy_events.csv")
TARGET_ROWS = 110

ECONOMIC_TOPIC_MAP = {
    1: "tax",
    3: "healthcare",
    4: "other",
    5: "regulation",
    6: "education",
    7: "other",
    8: "other",
    13: "other",
    14: "other",
    15: "regulation",
    17: "other",
    18: "trade",
    20: "spending",
}
ECONOMIC_TOPIC_CODES = set(ECONOMIC_TOPIC_MAP)

EXPANSIONARY_TERMS = (
    "appropriation",
    "authorize",
    "benefit",
    "credit",
    "cut",
    "development",
    "emergency",
    "expand",
    "extension",
    "funding",
    "grant",
    "incentive",
    "investment",
    "loan",
    "relief",
    "stimulus",
    "subsidy",
    "tax cut",
)
CONTRACTIONARY_TERMS = (
    "cap",
    "cutback",
    "deficit reduction",
    "limitation",
    "limit",
    "reduction",
    "rescission",
    "restriction",
    "spending cut",
    "tax increase",
)
HIGH_INTENSITY_TERMS = (
    "appropriation",
    "budget",
    "comprehensive",
    "emergency",
    "major",
    "omnibus",
    "reform",
    "stimulus",
)


def main() -> None:
    raw = fetch_cap_public_laws()
    events = transform_cap_public_laws(raw)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(events)} policy events to {OUTPUT_PATH}")


def fetch_cap_public_laws() -> pd.DataFrame:
    """Download the CAP public laws CSV and parse it into a DataFrame."""
    errors = []
    for url in (CAP_PUBLIC_LAWS_URL, CAP_PUBLIC_LAWS_FALLBACK_URL):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except requests.exceptions.RequestException as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Unable to download CAP public laws CSV from configured sources. "
        + " | ".join(errors)
    )


def transform_cap_public_laws(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter and standardize CAP public laws for the analog pipeline."""
    df = raw.copy()
    df.columns = [str(col).strip() for col in df.columns]

    year_col = find_first_column(df, ("year", "Year"))
    topic_col = find_first_column(
        df,
        (
            "majortopic",
            "major_topic",
            "MajorTopic",
            "major",
            "topic",
            "Topic",
        ),
    )
    description_col = find_first_column(
        df,
        (
            "title",
            "Title",
            "lawtitle",
            "LawTitle",
            "name",
            "Name",
            "description",
            "Description",
            "summary",
            "Summary",
        ),
    )

    if not year_col or not topic_col or not description_col:
        raise ValueError(
            "CAP CSV missing required year, topic, or description columns. "
            f"Columns found: {df.columns.tolist()}"
        )

    df["_year"] = pd.to_numeric(df[year_col], errors="coerce")
    df["_topic"] = pd.to_numeric(df[topic_col], errors="coerce")
    df["_description"] = df[description_col].fillna("").astype(str).str.strip()

    df = df[df["_year"].ge(1990)]
    df = df[df["_topic"].isin(ECONOMIC_TOPIC_CODES)]
    df = df[df["_description"].ne("")]

    commemorative_col = find_first_column(
        df,
        ("filter_commemorative", "commemorative", "filterCommemorative"),
    )
    if commemorative_col:
        commemorative = pd.to_numeric(df[commemorative_col], errors="coerce")
        df = df[commemorative.fillna(0).eq(0)]

    significance_col = find_significance_column(df)
    if significance_col:
        significant = filter_significant_laws(df, significance_col)
        if len(significant) >= 50:
            df = significant

    df = df.drop_duplicates(subset=["_year", "_description"])
    df["_direction"] = df["_description"].apply(infer_direction)
    df["_intensity"] = df["_description"].apply(infer_intensity)
    df["_ranking_score"] = df.apply(rank_policy_event, axis=1)
    df = df.sort_values(
        ["_ranking_score", "_year", "_topic", "_description"],
        ascending=[False, False, True, True],
    )
    df = df.head(TARGET_ROWS)
    df = df.sort_values(["_year", "_topic", "_description"])

    events = pd.DataFrame(
        {
            "year": df["_year"].astype(int),
            "state": "United States",
            "policy_change": df["_description"].apply(short_policy_change),
            "description": df["_description"],
            "policy_type": df["_topic"].astype(int).map(ECONOMIC_TOPIC_MAP),
            "direction": df["_direction"],
            "intensity": df["_intensity"],
            "sector": df["_topic"].astype(int).apply(map_sector),
        }
    )

    return events[
        [
            "year",
            "state",
            "policy_change",
            "description",
            "policy_type",
            "direction",
            "intensity",
            "sector",
        ]
    ]


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first matching column, case-insensitively."""
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower_to_original.get(candidate.lower())
        if match:
            return match
    return None


def find_significance_column(df: pd.DataFrame) -> str | None:
    """Find an optional CAP significance/major-law indicator column."""
    candidates = (
        "major",
        "maj_law",
        "majorlaw",
        "significant",
        "significance",
        "important",
        "landmark",
    )
    for col in df.columns:
        normalized = str(col).lower().replace("_", "")
        if any(candidate.replace("_", "") == normalized for candidate in candidates):
            return col
    return None


def filter_significant_laws(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Keep rows marked as major/significant when CAP exposes that field."""
    values = df[column]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return df[numeric.fillna(0).gt(0)]

    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return df[normalized.isin({"1", "true", "yes", "major", "significant"})]


def infer_direction(description: str) -> str:
    """Infer macro direction from clear action words in the law title."""
    text = description.lower()
    if contains_any(text, EXPANSIONARY_TERMS):
        return "expansionary"
    if contains_any(text, CONTRACTIONARY_TERMS):
        return "contractionary"
    return "neutral"


def infer_intensity(description: str) -> str:
    """Infer policy intensity from landmark-style wording."""
    text = description.lower()
    if contains_any(text, HIGH_INTENSITY_TERMS):
        return "high"
    return "medium"


def rank_policy_event(row: pd.Series) -> int:
    """Prefer high-signal macro laws when CAP lacks a major-law flag."""
    description = str(row["_description"]).lower()
    score = 0
    if row["_intensity"] == "high":
        score += 4
    if row["_direction"] != "neutral":
        score += 2
    if contains_any(description, ("budget", "reconciliation", "appropriation")):
        score += 3
    if contains_any(description, ("tax", "health", "labor", "trade", "transportation")):
        score += 1
    return score


def map_sector(topic_code: int) -> str:
    """Map CAP major topics to broad affected economic sectors."""
    if topic_code in {1, 15, 18, 20}:
        return "business"
    if topic_code == 3:
        return "households"
    if topic_code == 6:
        return "government"
    return "mixed"


def short_policy_change(description: str) -> str:
    """Create a compact label while preserving the official description."""
    clean = " ".join(description.split())
    if len(clean) <= 80:
        return clean
    return clean[:77].rstrip() + "..."


def contains_any(text: str, terms: Iterable[str]) -> bool:
    """Case-normalized containment helper."""
    return any(term in text for term in terms)


if __name__ == "__main__":
    main()
