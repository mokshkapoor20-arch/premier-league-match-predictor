"""
update_data.py
--------------
1. Download the latest results for the current season.
2. Download upcoming fixtures (with Bet365 odds when published).
"""
import io
import os

import pandas as pd
import requests

HEADERS = {"User-Agent": "PLMatchPredictorSchoolProject/1.0 (personal learning project)"}

SEASON_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

RAW_PATH = "data/raw/season-2026-2027.csv"
TMP_PATH = "data/raw/_download_tmp.csv"
FIXTURES_PATH = "data/processed/fixtures.csv"


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.content


# ---------- 1. Latest results ----------
with open(TMP_PATH, "wb") as f:
    f.write(fetch(SEASON_URL))

new_df = pd.read_csv(TMP_PATH, encoding="latin-1").dropna(subset=["HomeTeam", "AwayTeam"])
if os.path.exists(RAW_PATH):
    old_rows = len(pd.read_csv(RAW_PATH, encoding="latin-1").dropna(subset=["HomeTeam", "AwayTeam"]))
else:
    old_rows = 0

print(f"Matches you had: {old_rows} | Matches downloaded: {len(new_df)}")
if len(new_df) >= old_rows:
    os.replace(TMP_PATH, RAW_PATH)
    print(f"Updated {RAW_PATH} (+{len(new_df) - old_rows} new matches)")
else:
    os.remove(TMP_PATH)
    print("Downloaded file has FEWER matches than yours, so I kept your existing file.")

# ---------- 2. Upcoming fixtures (and their odds) ----------
raw = fetch(FIXTURES_URL)
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    text = raw.decode("latin-1")

fx = pd.read_csv(io.StringIO(text))
fx.columns = [c.strip().lstrip("\ufeff") for c in fx.columns]

if "Div" not in fx.columns:
    print("Unexpected fixtures format. First 300 characters:")
    print(text[:300])
else:
    fx = fx[fx["Div"] == "E0"]
    cols = [c for c in ["Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"] if c in fx.columns]
    fx = fx[cols]
    fx.to_csv(FIXTURES_PATH, index=False)
    if len(fx) == 0:
        print("File downloaded fine, but it contains no Premier League fixtures right now.")
    else:
        has_odds = int(fx["B365H"].notna().sum()) if "B365H" in fx.columns else 0
        print(f"Saved {len(fx)} upcoming fixtures ({has_odds} with odds) to {FIXTURES_PATH}")
        print(fx.head(10).to_string(index=False))
