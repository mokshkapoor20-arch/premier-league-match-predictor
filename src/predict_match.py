"""
predict_match.py
----------------
The prediction engine.

  1. A logistic regression (trained on every completed match with full pre-match
     features) gives win / draw / loss probabilities.
  2. A Poisson goals model gives expected goals and likely scorelines.
  3. Player scoring chances come from player_model.py.

For a match that hasn't been played, each team's "form" is taken from its latest
5 matches INCLUDING the most recent one (see live_features.py).
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from live_features import NotEnoughHistory, current_features  # noqa: F401  (NotEnoughHistory is re-exported)
from player_model import load_squads, player_scoring_probabilities

FEATURES = ["H_L5_Pts", "H_L5_GF", "H_L5_GA", "A_L5_Pts", "A_L5_GF", "A_L5_GA",
            "HomeForm", "AwayForm", "H2H_HomePts",
            "H_L5_S", "H_L5_ST", "H_L5_C", "A_L5_S", "A_L5_ST", "A_L5_C"]

squads = load_squads()
matches = pd.read_csv("data/processed/matches_combined.csv", parse_dates=["Date"])
train = pd.read_csv("data/processed/features_v2.csv").dropna(subset=FEATURES)

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
clf.fit(train[FEATURES], train["FTR"])

league_avg_goals = train[["FTHG", "FTAG"]].values.mean()
home_advantage = train["FTHG"].mean() / train["FTAG"].mean()

SHRINKAGE_GAMES = 8      # blend each team's 5-match goal rates with 8 "virtual" league-average matches


def shrink(rate_sum, games, league_avg):
    return (rate_sum + league_avg * SHRINKAGE_GAMES) / (games + SHRINKAGE_GAMES)


def check_staleness(team, last_match_date, max_days=400):
    days_since = (pd.Timestamp.now() - pd.Timestamp(last_match_date)).days
    if days_since > max_days:
        return (f"⚠️ Warning: {team}'s most recent match in the data was {days_since} days ago "
                f"({pd.Timestamp(last_match_date).date()}). They may no longer be in the Premier League, "
                "or this data needs updating.")
    return None


def predict_match(home_team, away_team):
    feats, h, a, h_last, a_last = current_features(matches, home_team, away_team)
    warnings = [w for w in (check_staleness(home_team, h_last), check_staleness(away_team, a_last)) if w]

    probs = clf.predict_proba(pd.DataFrame([feats])[FEATURES])[0]
    wdl = dict(zip(clf.classes_, probs))

    home_attack = shrink(h["GF"], 5, league_avg_goals)
    home_defense = shrink(h["GA"], 5, league_avg_goals)
    away_attack = shrink(a["GF"], 5, league_avg_goals)
    away_defense = shrink(a["GA"], 5, league_avg_goals)
    exp_home = league_avg_goals * (home_attack / league_avg_goals) * (away_defense / league_avg_goals) * home_advantage
    exp_away = league_avg_goals * (away_attack / league_avg_goals) * (home_defense / league_avg_goals)

    grid = np.outer([poisson.pmf(i, exp_home) for i in range(7)], [poisson.pmf(i, exp_away) for i in range(7)])
    flat = sorted(((i, j, grid[i, j]) for i in range(7) for j in range(7)), key=lambda x: -x[2])

    factors = []
    if a["GA"] / 5 > league_avg_goals * 1.15:
        factors.append(f"{away_team} have been leaky defensively lately")
    if h["GA"] / 5 > league_avg_goals * 1.15:
        factors.append(f"{home_team} have been leaky defensively lately")
    if h["GF"] / 5 > league_avg_goals * 1.15:
        factors.append(f"{home_team} have scored heavily in recent games")
    if a["GF"] / 5 > league_avg_goals * 1.15:
        factors.append(f"{away_team} have scored heavily in recent games")
    if feats["H2H_HomePts"] > 2.0:
        factors.append(f"{home_team} have historically dominated this fixture")
    elif feats["H2H_HomePts"] < 1.0:
        factors.append(f"{away_team} have historically dominated this fixture")
    if not factors:
        factors.append("No single factor stands out. A fairly even matchup on the numbers")

    return {
        "home_team": home_team,
        "away_team": away_team,
        "p_home": wdl.get("H", 0),
        "p_draw": wdl.get("D", 0),
        "p_away": wdl.get("A", 0),
        "exp_home_goals": exp_home,
        "exp_away_goals": exp_away,
        "top_scorelines": flat[:5],
        "warnings": warnings,
        "key_factors": factors,
        "home_players": player_scoring_probabilities(home_team, exp_home, squads),
        "away_players": player_scoring_probabilities(away_team, exp_away, squads),
        "form": {"home": h, "away": a},          # last-5 snapshot, used by the form chart
    }


if __name__ == "__main__":
    print(predict_match("Chelsea", "Arsenal"))
