import pandas as pd
import os

folder = "data/gdp_raw"
all_data = []

for file in os.listdir(folder):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(folder, file)

    # -----------------------------
    # 1. Load raw file (no assumptions)
    # -----------------------------
    raw = pd.read_csv(path, header=None, dtype=str)

    header_row = None

    # -----------------------------
    # 2. Detect header row robustly
    # -----------------------------
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).tolist()

        has_description = any("Description" in cell for cell in row)
        has_year = any(cell.strip().isdigit() for cell in row)

        if has_description and has_year:
            header_row = i
            break

    if header_row is None:
        continue

    # -----------------------------
    # 3. Re-read with correct header
    # -----------------------------
    df = pd.read_csv(path, header=header_row)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Extract state from filename
    state = file.split("_")[1]
    # keep only real state abbreviations (2-letter codes)
    if len(state) != 2:
        continue

    # -----------------------------
    # 4. Clean description field
    # -----------------------------
    if "Description" not in df.columns:
        continue

    df["Description"] = df["Description"].astype(str).str.strip()

    # -----------------------------
    # 5. Keep ONLY total GDP row
    # (strict match to avoid duplicates)
    # -----------------------------
    df = df[df["Description"] == "All industry total"]

    if df.empty:
        continue

    # -----------------------------
    # 6. Identify year columns
    # -----------------------------
    year_cols = [c for c in df.columns if str(c).isdigit()]

    if len(year_cols) == 0:
        continue

    # -----------------------------
    # 7. Convert wide → long
    # -----------------------------
    df = df.melt(
        id_vars=["Description"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp"
    )

    # -----------------------------
    # 8. Add state + clean values
    # -----------------------------
    df["state"] = state
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gdp"] = pd.to_numeric(df["gdp"], errors="coerce")

    df = df.dropna()

    df = df[["state", "year", "gdp"]]

    all_data.append(df)

# -----------------------------
# 9. Combine all states
# -----------------------------
final_df = pd.concat(all_data, ignore_index=True)

# -----------------------------
# 10. Remove duplicates (robust state-year aggregation)
# -----------------------------
final_df = (
    final_df
    .groupby(["state", "year"], as_index=False)
    .agg({"gdp": "mean"})
)

# -----------------------------
# 11. Save clean panel dataset
# -----------------------------
final_df.to_csv("data/state_gdp_clean.csv", index=False)

print("Done:", final_df.shape)
print(final_df.head())