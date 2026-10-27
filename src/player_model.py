"""
player_model.py
---------------
Chance of each player scoring, for one match.

How it works:
  1. Each player gets a scoring RATE (goals per 90 minutes), built from this
     season's goals and expected goals, shrunk toward a typical rate for their
     position so a few matches can't mislead it.
  2. That rate is multiplied by how much they actually play (share of minutes)
     and by availability (injured = 0, doubtful = partial).
  3. The team's expected goals for the match (which already reflects how weak or
     strong the opponent's defence is) are split between players in proportion.
  4. Chance of scoring at least once = 1 - exp(-player's expected goals).
"""
import json
import math
import os

PLAYERS_PATH = "data/processed/players.json"
MANUAL_PATH = "data/processed/squads_manual.json"

# Typical goals per 90 for a regular in each position. Rough starting guesses,
# not fitted from data. They only matter early in the season.
PRIOR_PER90 = {"FWD": 0.30, "MID": 0.11, "DEF": 0.04, "GK": 0.002}
PRIOR_NINETIES = 5.0       # how many "virtual" matches of typical form to blend in


def load_squads():
    if os.path.exists(PLAYERS_PATH):
        with open(PLAYERS_PATH, encoding="utf-8") as f:
            return {"format": "fpl", "teams": json.load(f)["teams"]}
    if os.path.exists(MANUAL_PATH):                       # old hand-typed squads
        with open(MANUAL_PATH, encoding="utf-8") as f:
            return {"format": "manual", "teams": json.load(f)}
    return {"format": "none", "teams": {}}


def _availability(p):
    status, chance = p.get("status", "a"), p.get("chance")
    if status in ("i", "s", "u", "n"):                    # injured / suspended / unavailable
        return 0.0
    if chance is not None:
        return max(0.0, min(1.0, float(chance) / 100.0))
    return 0.5 if status == "d" else 1.0                  # doubtful with no figure -> 50%


def _from_feed(team_xg, squad):
    games = max(1, math.ceil(max(p["minutes"] for p in squad) / 90.0))   # matches the club has played
    started = any(p["minutes"] > 0 for p in squad)
    rows = []
    for p in squad:
        pos = p["pos"]
        perf = 0.5 * p["goals"] + 0.5 * p["xg"]
        rate = (perf + PRIOR_PER90.get(pos, 0.05) * PRIOR_NINETIES) / (p["minutes"] / 90.0 + PRIOR_NINETIES)
        if started:
            plays = min(1.0, p["minutes"] / (90.0 * games))
        else:                                              # before a ball is kicked
            plays = 0.0 if pos == "GK" else 0.5
        avail = _availability(p)
        rows.append((p, rate * plays * avail, avail))
    total = sum(w for _, w, _ in rows)
    if total <= 0:
        return []
    out = []
    for p, w, avail in rows:
        player_xg = team_xg * w / total
        out.append({
            "name": p["name"],
            "position": p["pos"] + (" · doubtful" if 0 < avail < 1 else ""),
            "prob_score": 1 - math.exp(-player_xg),
        })
    return sorted(out, key=lambda r: -r["prob_score"])


def _from_manual(team_xg, squad):
    total = sum(p["attack_weight"] for p in squad)
    if total == 0:
        return []
    out = [{"name": p["name"], "position": p["position"],
            "prob_score": 1 - math.exp(-team_xg * p["attack_weight"] / total)} for p in squad]
    return sorted(out, key=lambda r: -r["prob_score"])


def player_scoring_probabilities(team, team_expected_goals, squads):
    squad = squads.get("teams", {}).get(team, [])
    if not squad:
        return []
    if squads.get("format") == "fpl":
        return _from_feed(team_expected_goals, squad)
    return _from_manual(team_expected_goals, squad)
