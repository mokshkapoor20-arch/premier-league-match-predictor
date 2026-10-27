"""
tracker.py
----------
Your prediction record.

  * The FIRST prediction saved for a fixture is the one that counts (so you can't
    quietly replace a prediction after seeing the result).
  * Results fill in automatically from the final scores in fixtures_all.json.
  * A prediction only counts if it was saved BEFORE kick-off.
"""
import json
import os
from datetime import datetime, timedelta

import pandas as pd

LOG_PATH = "data/processed/prediction_log.csv"
FIXTURES_PATH = "data/processed/fixtures_all.json"

COLUMNS = ["id", "logged_at", "home_team", "away_team", "p_home", "p_draw", "p_away",
           "predicted_outcome", "actual_outcome", "correct", "source", "confidence"]
TEXT_COLUMNS = {"logged_at": str, "actual_outcome": str, "correct": str, "source": str, "confidence": str}


def load_log():
    if os.path.exists(LOG_PATH):
        # keep_default_na=False: an empty "actual_outcome" stays "" instead of turning into NaN
        log = pd.read_csv(LOG_PATH, keep_default_na=False, dtype=TEXT_COLUMNS)
        for c in COLUMNS:
            if c not in log.columns:
                log[c] = ""
        return log[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def log_prediction(home_team, away_team, p_home, p_draw, p_away, source="model", confidence=""):
    """Save a prediction. Returns the new row, or None if this fixture is already in the record."""
    log = load_log()
    already = log[(log["home_team"] == home_team) & (log["away_team"] == away_team) & (log["actual_outcome"] == "")]
    if len(already) > 0:
        return None

    outcomes = {"H": float(p_home), "D": float(p_draw), "A": float(p_away)}
    row = {
        "id": int(log["id"].astype(int).max()) + 1 if len(log) else 1,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "home_team": home_team,
        "away_team": away_team,
        "p_home": round(outcomes["H"], 3),
        "p_draw": round(outcomes["D"], 3),
        "p_away": round(outcomes["A"], 3),
        "predicted_outcome": max(outcomes, key=outcomes.get),
        "actual_outcome": "",
        "correct": "",
        "source": source,
        "confidence": confidence,
    }
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)
    return row


def record_result(prediction_id, actual_outcome):
    log = load_log()
    idx = log.index[log["id"].astype(int) == int(prediction_id)]
    if len(idx) == 0:
        return False
    i = idx[0]
    log.loc[i, "actual_outcome"] = actual_outcome
    log.loc[i, "correct"] = "True" if log.loc[i, "predicted_outcome"] == actual_outcome else "False"
    log.to_csv(LOG_PATH, index=False)
    return True


def compute_stats():
    log = load_log()
    resolved = log[log["actual_outcome"] != ""]
    pending = len(log) - len(resolved)
    if len(resolved) == 0:
        return {"total": 0, "correct": 0, "accuracy": None, "pending": pending}
    correct = int((resolved["correct"] == "True").sum())
    return {"total": len(resolved), "correct": correct, "accuracy": correct / len(resolved), "pending": pending}


def auto_resolve():
    """Mark pending predictions right or wrong using finished matches. Returns how many were resolved."""
    if not os.path.exists(FIXTURES_PATH):
        return 0
    try:
        with open(FIXTURES_PATH, encoding="utf-8") as f:
            fixtures = json.load(f)["fixtures"]
    except Exception:
        return 0
    finished = {(g["home"], g["away"]): g for g in fixtures
                if g.get("finished") and g.get("hs") is not None and g.get("as") is not None}

    log = load_log()
    pending = log[log["actual_outcome"] == ""]
    if pending.empty or not finished:
        return 0

    offset = datetime.now().astimezone().utcoffset() or timedelta(0)     # this computer's clock vs UTC
    changed = 0
    for _, row in pending.iterrows():
        g = finished.get((row["home_team"], row["away_team"]))
        if not g or not g.get("kickoff"):
            continue
        try:
            kickoff = datetime.fromisoformat(g["kickoff"].replace("Z", "+00:00")).replace(tzinfo=None)
            saved_at = datetime.strptime(row["logged_at"], "%Y-%m-%d %H:%M") - offset
        except Exception:
            continue
        if saved_at >= kickoff:                   # saved after kick-off: doesn't count
            continue
        hs, aws = int(g["hs"]), int(g["as"])
        outcome = "H" if hs > aws else ("A" if hs < aws else "D")
        if record_result(int(row["id"]), outcome):
            changed += 1
    return changed


def breakdown(by="confidence"):
    """How the resolved picks did, grouped by confidence level (or by source)."""
    log = load_log()
    done = log[log["actual_outcome"] != ""].copy()
    if done.empty:
        return pd.DataFrame()
    done[by] = done[by].replace("", "Not recorded")
    done["hit"] = done["correct"] == "True"
    out = done.groupby(by)["hit"].agg(["count", "sum"]).reset_index()
    out.columns = [by.capitalize(), "Resolved", "Correct"]
    out["Right"] = (out["Correct"] / out["Resolved"] * 100).round().astype(int).astype(str) + "%"
    order = {"High": 0, "Medium": 1, "Low": 2, "market": 0, "model": 1}
    return out.sort_values(by.capitalize(), key=lambda col: col.map(lambda v: order.get(v, 9))).reset_index(drop=True)
