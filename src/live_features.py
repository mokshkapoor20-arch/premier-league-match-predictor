"""
live_features.py
----------------
The numbers the model needs to predict a match that has NOT been played yet.

Training rows use "form going into a match" (the 5 matches BEFORE it). For a
future match, "form going in" = each team's latest 5 matches INCLUDING the most
recent one. This file computes exactly that, using the same definitions as
build_features.py (so training and live use match perfectly):

  * last-5 points / goals for / goals against  (sums)
  * last-5 shots, shots on target, corners     (averages)
  * home team's average points in its last 5 HOME matches
    / away team's average points in its last 5 AWAY matches (needs 3, else uses overall form)
  * home team's average points in the last 3 meetings of these two clubs
"""
import pandas as pd

MIN_MATCHES = 5


class NotEnoughHistory(Exception):
    pass


def team_log(matches, team):
    """One row per match for this team, oldest first, from the team's point of view."""
    home = matches[matches["HomeTeam"] == team]
    away = matches[matches["AwayTeam"] == team]
    h = pd.DataFrame({"Date": home["Date"], "GF": home["FTHG"], "GA": home["FTAG"],
                      "S": home["HS"], "ST": home["HST"], "C": home["HC"], "IsHome": 1})
    a = pd.DataFrame({"Date": away["Date"], "GF": away["FTAG"], "GA": away["FTHG"],
                      "S": away["AS"], "ST": away["AST"], "C": away["AC"], "IsHome": 0})
    log = pd.concat([h, a]).sort_values("Date", kind="mergesort").reset_index(drop=True)
    log["Pts"] = (log["GF"] > log["GA"]) * 3 + (log["GF"] == log["GA"]) * 1
    return log


def snapshot(log):
    """Form over a team's latest 5 matches (including the most recent one)."""
    last = log.tail(5)
    return {"Pts": float(last["Pts"].sum()), "GF": float(last["GF"].sum()), "GA": float(last["GA"].sum()),
            "S": float(last["S"].mean()), "ST": float(last["ST"].mean()), "C": float(last["C"].mean())}


def venue_form(log, is_home):
    v = log[log["IsHome"] == is_home].tail(5)["Pts"]
    return float(v.mean()) if len(v) >= 3 else float(log.tail(5)["Pts"].mean())


def head_to_head(matches, home, away):
    """Home team's average points over the last 3 meetings (either venue); 1.0 if they have never met."""
    meet = matches[((matches["HomeTeam"] == home) & (matches["AwayTeam"] == away)) |
                   ((matches["HomeTeam"] == away) & (matches["AwayTeam"] == home))].tail(3)
    if meet.empty:
        return 1.0
    pts = []
    for _, r in meet.iterrows():
        gf, ga = (r["FTHG"], r["FTAG"]) if r["HomeTeam"] == home else (r["FTAG"], r["FTHG"])
        pts.append(3 if gf > ga else (1 if gf == ga else 0))
    return float(sum(pts) / len(pts))


def current_features(matches, home, away):
    """Model inputs for home vs away, plus the two snapshots and each club's most recent match date."""
    hl, al = team_log(matches, home), team_log(matches, away)
    for team, log in ((home, hl), (away, al)):
        if len(log) < MIN_MATCHES:
            raise NotEnoughHistory(f"{team} has only played {len(log)} matches in the data so far "
                                   f"(needs {MIN_MATCHES}).")
    h, a = snapshot(hl), snapshot(al)
    feats = {
        "H_L5_Pts": h["Pts"], "H_L5_GF": h["GF"], "H_L5_GA": h["GA"],
        "A_L5_Pts": a["Pts"], "A_L5_GF": a["GF"], "A_L5_GA": a["GA"],
        "HomeForm": venue_form(hl, 1), "AwayForm": venue_form(al, 0),
        "H2H_HomePts": head_to_head(matches, home, away),
        "H_L5_S": h["S"], "H_L5_ST": h["ST"], "H_L5_C": h["C"],
        "A_L5_S": a["S"], "A_L5_ST": a["ST"], "A_L5_C": a["C"],
    }
    return feats, h, a, hl["Date"].iloc[-1], al["Date"].iloc[-1]
