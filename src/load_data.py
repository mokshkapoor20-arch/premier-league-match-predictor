"""
load_data.py
------------
Combine every season file in data/raw into one clean, date-sorted table.

Changes in this version:
  * reads rows safely (older files sometimes have stray extra columns; the
    old method silently DROPPED those matches)
  * keeps the Bet365 pre-match odds columns (B365H / B365D / B365A)
"""
import csv
import glob

import pandas as pd

WANTED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HC", "AC",
    "HF", "AF",
    "HY", "AY", "HR", "AR",
    "Referee",
    "B365H", "B365D", "B365A",     # Bet365 odds: home win / draw / away win
]
NUMERIC = ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC",
           "HF", "AF", "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A"]


def read_season_file(path):
    """Read a CSV without losing rows that have extra or missing fields."""
    with open(path, encoding="latin-1", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in rows[1:] if any(c.strip() for c in r)]
    return pd.DataFrame(body, columns=header).replace("", pd.NA)


def parse_dates(series):
    # Seasons use different formats: dd/mm/yy, dd/mm/YYYY and YYYY-mm-dd
    parsed = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    for fmt in ("%d/%m/%y", "%Y-%m-%d"):
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(series[missing], format=fmt, errors="coerce")
    return parsed


files = sorted(glob.glob("data/raw/season-*.csv"))
label = lambda f: f.split("season-")[1].replace(".csv", "")
print(f"Found {len(files)} season files ({label(files[0])} to {label(files[-1])})")

frames = []
for f in files:
    df = read_season_file(f)
    df = df[[c for c in WANTED_COLUMNS if c in df.columns]].copy()
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = parse_dates(df["Date"])
    df["Season"] = f.split("season-")[1].replace(".csv", "")
    frames.append(df)

matches = pd.concat(frames, ignore_index=True, sort=False)
before = len(matches)
matches = matches.dropna(subset=["Date", "FTHG", "FTAG"])
print(f"Dropped {before - len(matches)} unusable row(s) (no date or no score)")

matches = matches.sort_values("Date", kind="mergesort").reset_index(drop=True)

# Recompute the result from the goals rather than trusting the file
computed = pd.Series("D", index=matches.index)
computed[matches["FTHG"] > matches["FTAG"]] = "H"
computed[matches["FTHG"] < matches["FTAG"]] = "A"
print("FTR mismatches:", int((computed != matches["FTR"]).sum()))
matches["FTR"] = computed

print(matches.shape)
print(matches["Season"].value_counts().sort_index())
if "B365H" in matches.columns:
    have = matches["B365H"].notna().sum()
    print(f"Matches with Bet365 odds: {have} of {len(matches)}")

matches.to_csv("data/processed/matches_combined.csv", index=False)
print("Saved to data/processed/matches_combined.csv")
