"""
fetch_fixtures.py
-----------------
Download the whole season's fixture list (all 38 matchweeks, with kick-off times, live status and
scores) from the official Fantasy Premier League feed and save it. The app also does this by itself
every 30 seconds while it is open; this script is the manual version used by refresh_all.py.
"""
import sys

sys.path.insert(0, "src")
from live import FIXTURES_PATH, _write_file, fetch_feed

try:
    fixtures = fetch_feed()
except Exception as e:
    print(f"Could not read the fixture feed ({e}).")
    print("Your existing fixtures file (if any) was left unchanged.")
    raise SystemExit(1)

_write_file(fixtures)

weeks = sorted({f["event"] for f in fixtures if f["event"] is not None})
played = sum(1 for f in fixtures if f["finished"])
live = sum(1 for f in fixtures if f["started"] and not f["finished"])
print(f"Saved {len(fixtures)} fixtures across {len(weeks)} matchweeks to {FIXTURES_PATH} "
      f"({played} played, {live} live right now).")

upcoming_weeks = [w for w in weeks if any(not f["finished"] for f in fixtures if f["event"] == w)]
if upcoming_weeks:
    nxt = upcoming_weeks[0]
    print(f"\nCurrent / next matchweek: {nxt}")
    for f in sorted((f for f in fixtures if f["event"] == nxt), key=lambda f: f["kickoff"] or ""):
        state = "FT " if f["finished"] else ("LIVE" if f["started"] else "    ")
        score = f"{f['hs']}-{f['as']}" if f["hs"] is not None else "   "
        print(f"  {state} {f['kickoff'] or 'TBC':<22} {f['home']} {score} {f['away']}")

try:
    import pandas as pd
    names = {f["home"] for f in fixtures} | {f["away"] for f in fixtures}
    known = set(pd.read_csv("data/processed/matches_combined.csv")["HomeTeam"])
    odd = sorted(n for n in names if n not in known)
    print("\nClub names not in your match data (spelling may differ):", odd if odd else "none")
except Exception:
    pass
