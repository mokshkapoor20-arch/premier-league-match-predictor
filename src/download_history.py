"""
download_history.py
-------------------
Download older Premier League seasons (2002-03 to 2013-14) from
football-data.co.uk into data/raw. More history = more matches to learn
from, and these seasons all include Bet365 odds.
Safe to re-run: seasons you already have are skipped.
"""
import os
import time

import requests

HEADERS = {"User-Agent": "PLMatchPredictorSchoolProject/1.0 (personal learning project)"}
FIRST_YEAR, LAST_YEAR = 2002, 2013          # season START years

os.makedirs("data/raw", exist_ok=True)
saved = 0

for y in range(FIRST_YEAR, LAST_YEAR + 1):
    label = f"{y}-{y + 1}"
    path = f"data/raw/season-{label}.csv"
    if os.path.exists(path):
        print(f"{label}: already have it, skipping")
        continue
    code = f"{y % 100:02d}{(y + 1) % 100:02d}"
    url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        text = r.content.decode("latin-1")
        header = text.splitlines()[0] if text else ""
        rows = text.count("\n")
        if "HomeTeam" not in header or rows < 300:
            print(f"{label}: downloaded but it doesn't look like a season file, skipped")
        else:
            with open(path, "wb") as f:
                f.write(r.content)
            saved += 1
            print(f"{label}: saved ({rows} rows)")
    except Exception as e:
        print(f"{label}: FAILED ({e})")
    time.sleep(1)

print(f"\nDone. {saved} new season file(s) added to data/raw.")
