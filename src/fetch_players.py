"""
fetch_players.py
----------------
Download every Premier League squad from the official Fantasy Premier League
data feed (free, no login). For each player it gives: position, goals, expected
goals (xG), minutes, starts, and injury/availability news.
This replaces the hand-typed squads, and covers all 20 clubs.
"""
import json
import os
from datetime import datetime

import requests
from club_names import POSITIONS, TEAM_FIX

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PLMatchPredictorSchoolProject/1.0; personal learning project)"}
OUT = "data/processed/players.json"

def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


try:
    r = requests.get(URL, headers=HEADERS, timeout=30)
    data = r.json()
except Exception as e:
    status = locals().get("r") and r.status_code
    print(f"Could not read the player feed ({e}). HTTP status: {status}")
    if locals().get("r"):
        print("First 200 characters of the reply:", r.text[:200])
    print("Your existing players file (if any) was left unchanged.")
    raise SystemExit(1)

teams = {t["id"]: TEAM_FIX.get(t["name"], t["name"]) for t in data["teams"]}
squads = {name: [] for name in teams.values()}

for p in data["elements"]:
    team = teams.get(p.get("team"))
    if team is None:
        continue
    squads[team].append({
        "name": p.get("web_name") or p.get("second_name") or "Unknown",
        "pos": POSITIONS.get(p.get("element_type"), "MID"),
        "goals": int(number(p.get("goals_scored"))),
        "xg": round(number(p.get("expected_goals")), 2),
        "minutes": int(number(p.get("minutes"))),
        "starts": int(number(p.get("starts"))),
        "status": p.get("status") or "a",
        "chance": p.get("chance_of_playing_next_round"),
        "news": p.get("news") or "",
    })

os.makedirs("data/processed", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"fetched_at": datetime.now().isoformat(timespec="minutes"), "teams": squads},
              f, ensure_ascii=False, indent=1)

print(f"Saved {sum(len(v) for v in squads.values())} players for {len(squads)} clubs to {OUT}\n")
print(f"{'Club':<18}{'Players':>8}  Top scorers (goals)")
for team in sorted(squads):
    top = sorted(squads[team], key=lambda p: -p["goals"])[:3]
    print(f"{team:<18}{len(squads[team]):>8}  " + ", ".join(f"{p['name']} ({p['goals']})" for p in top))

# Check the club names match your match data
try:
    import pandas as pd
    known = set(pd.read_csv("data/processed/matches_combined.csv")["HomeTeam"])
    odd = [t for t in squads if t not in known]
    if odd:
        print("\nThese club names are NOT in your match data (the spelling may differ):", odd)
    else:
        print("\nAll club names match your match data.")
except Exception:
    pass
