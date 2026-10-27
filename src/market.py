"""
market.py
---------
Use the bookmakers' pre-match odds when they exist for a fixture.

Why: in a walk-forward test over 12 seasons, the bookmakers' implied
probabilities were right about 55% of the time vs about 52% for the form-only
model. Odds are public before kickoff, so using them respects the "only
information available before kickoff" rule.

For pairings with no published odds, the form model is used on its own and the
app says so. Either way, the goals and scorelines are fitted to the headline probabilities.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from player_model import load_squads, player_scoring_probabilities

FIXTURES_PATH = "data/processed/fixtures.csv"
TABLE_PATH = "data/processed/confidence_table.json"

# How often the top pick was right, by how confident it was. These defaults come
# from a 2007-2019 test; running model_6_walkforward.py replaces them with
# numbers measured on your own data.
DEFAULT_TABLE = {
    "market": [{"lo": 0.0, "hi": 0.4, "n": 707, "hit": 0.37}, {"lo": 0.4, "hi": 0.5, "n": 1316, "hit": 0.46},
               {"lo": 0.5, "hi": 0.6, "n": 881, "hit": 0.57}, {"lo": 0.6, "hi": 0.7, "n": 722, "hit": 0.68},
               {"lo": 0.7, "hi": 1.01, "n": 645, "hit": 0.80}],
    "model": [{"lo": 0.0, "hi": 0.4, "n": 471, "hit": 0.38}, {"lo": 0.4, "hi": 0.5, "n": 1404, "hit": 0.43},
              {"lo": 0.5, "hi": 0.6, "n": 1131, "hit": 0.50}, {"lo": 0.6, "hi": 0.7, "n": 840, "hit": 0.66},
              {"lo": 0.7, "hi": 1.01, "n": 425, "hit": 0.74}],
}

_squads = None


def market_probs(home, away):
    """De-vigged probabilities from the fixture's odds, or None if unavailable."""
    if not os.path.exists(FIXTURES_PATH):
        return None
    fx = pd.read_csv(FIXTURES_PATH)
    if fx.empty or not {"B365H", "B365D", "B365A"}.issubset(fx.columns):
        return None
    row = fx[(fx["HomeTeam"] == home) & (fx["AwayTeam"] == away)]
    if row.empty:
        return None
    try:
        odds = [float(row.iloc[0][c]) for c in ("B365H", "B365D", "B365A")]
    except (TypeError, ValueError):
        return None
    if any(np.isnan(o) or o <= 1.0 for o in odds):
        return None
    inv = [1.0 / o for o in odds]
    total = sum(inv)                                  # > 1: the bookmaker's margin
    return {"H": inv[0] / total, "D": inv[1] / total, "A": inv[2] / total}


def _grid(lh, la, n=10):
    grid = np.outer(poisson.pmf(np.arange(n + 1), lh), poisson.pmf(np.arange(n + 1), la))
    return grid / grid.sum()


def align_goals(model_home_xg, model_away_xg, target):
    """Shift expected goals between the teams so the scoreline grid agrees with
    the headline probabilities (the model's total goals are kept)."""
    total = max(0.8, min(5.0, model_home_xg + model_away_xg))

    def loss(s):
        g = _grid((total + s) / 2, (total - s) / 2)
        return (np.tril(g, -1).sum() - target["H"]) ** 2 + (np.triu(g, 1).sum() - target["A"]) ** 2

    s = minimize_scalar(loss, bounds=(-total + 0.3, total - 0.3), method="bounded").x
    lh, la = (total + s) / 2, (total - s) / 2
    g = _grid(lh, la)
    top = sorted(((i, j, float(g[i, j])) for i in range(7) for j in range(7)), key=lambda x: -x[2])[:5]
    return lh, la, top


def _table(source):
    if os.path.exists(TABLE_PATH):
        try:
            with open(TABLE_PATH) as f:
                return json.load(f)[source]
        except Exception:
            pass
    return DEFAULT_TABLE[source]


def confidence(top, source):
    label = "High" if top >= 0.60 else ("Medium" if top >= 0.45 else "Low")
    hit = n = None
    for b in _table(source):
        if b["lo"] <= top < b["hi"]:
            hit, n = b["hit"], b["n"]
    return {"label": label, "top": float(top), "hit_rate": hit, "n": n}


def apply_market(result, home, away):
    global _squads
    result = dict(result)
    model = {"H": float(result["p_home"]), "D": float(result["p_draw"]), "A": float(result["p_away"])}
    mk = market_probs(home, away)
    final, source = (mk, "market") if mk else (model, "model")

    result["model_probs"], result["market_probs"], result["source"] = model, mk, source
    result["p_home"], result["p_draw"], result["p_away"] = final["H"], final["D"], final["A"]

    who = {"H": home, "D": "a draw", "A": away}
    top_key = max(final, key=final.get)
    result["confidence"] = confidence(final[top_key], source)

    notes = []
    if mk:
        mt, ml = max(mk, key=mk.get), max(model, key=model.get)
        if mt == ml:
            notes.append(f"The form model and the bookmaker odds agree: {who[mt]} ({mk[mt]*100:.0f}% by odds, {model[ml]*100:.0f}% by the form model).")
        else:
            notes.append(f"The form model and the bookmaker odds disagree: odds favour {who[mt]} ({mk[mt]*100:.0f}%), the form model favours {who[ml]} ({model[ml]*100:.0f}%).")
    else:
        notes.append("No bookmaker odds are published for this pairing, so this is the form-only model.")

    # Keep expected goals, scorelines and player chances consistent with the headline
    # probabilities (otherwise the page could say Liverpool are favourites while showing
    # Chelsea with more expected goals).
    try:
        lh, la, top = align_goals(float(result["exp_home_goals"]), float(result["exp_away_goals"]), final)
        result["exp_home_goals"], result["exp_away_goals"], result["top_scorelines"] = lh, la, top
        if _squads is None:
            _squads = load_squads()
        result["home_players"] = player_scoring_probabilities(home, lh, _squads)
        result["away_players"] = player_scoring_probabilities(away, la, _squads)
    except Exception:
        pass
    result["key_factors"] = notes + list(result.get("key_factors", []))
    return result
