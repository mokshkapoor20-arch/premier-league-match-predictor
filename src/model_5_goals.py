import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss

m = pd.read_csv("data/processed/features_v2.csv")
m = m.dropna(subset=["H_L5_GF", "H_L5_GA", "A_L5_GF", "A_L5_GA"])
m = m[m["Season"] != "2026-2027"]

train = m[m["Season"] < "2025-2026"]
test = m[m["Season"] == "2025-2026"]

# League-wide average goals per team per match, calculated from TRAINING
# data only — this is the baseline every team gets compared against.
league_avg_goals = train[["FTHG", "FTAG"]].values.mean()
print("League average goals per team per match:", round(league_avg_goals, 3))

home_goals_avg = train["FTHG"].mean()
away_goals_avg = train["FTAG"].mean()
home_advantage = home_goals_avg / away_goals_avg
print("Home advantage multiplier:", round(home_advantage, 3))

def expected_goals(row):
    home_attack = row.H_L5_GF / 5
    home_defense = row.H_L5_GA / 5
    away_attack = row.A_L5_GF / 5
    away_defense = row.A_L5_GA / 5

    exp_home = league_avg_goals * (home_attack / league_avg_goals) * (away_defense / league_avg_goals) * home_advantage
    exp_away = league_avg_goals * (away_attack / league_avg_goals) * (home_defense / league_avg_goals)
    return exp_home, exp_away

test = test.copy()
test[["ExpHomeGoals", "ExpAwayGoals"]] = test.apply(lambda r: pd.Series(expected_goals(r)), axis=1)

def scoreline_matrix(exp_home, exp_away, max_goals=6):
    home_probs = [poisson.pmf(i, exp_home) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, exp_away) for i in range(max_goals + 1)]
    return np.outer(home_probs, away_probs)   # rows = home goals, cols = away goals

def match_outcome_probs(matrix):
    p_home = np.sum(np.tril(matrix, -1))   # home goals > away goals
    p_draw = np.sum(np.diag(matrix))       # equal
    p_away = np.sum(np.triu(matrix, 1))    # away goals > home goals
    return p_home, p_draw, p_away

# ---- Show one example in detail ----
row = test.iloc[0]
matrix = scoreline_matrix(row.ExpHomeGoals, row.ExpAwayGoals)
p_h, p_d, p_a = match_outcome_probs(matrix)

print(f"\n{row.HomeTeam} vs {row.AwayTeam}")
print(f"Expected goals: {row.ExpHomeGoals:.2f} - {row.ExpAwayGoals:.2f}")
print(f"P(Home)={p_h:.2f}  P(Draw)={p_d:.2f}  P(Away)={p_a:.2f}")

flat = [(i, j, matrix[i, j]) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
flat.sort(key=lambda x: -x[2])
print("Top 5 most likely scorelines:")
for i, j, p in flat[:5]:
    print(f"  {i}-{j}: {p:.3f}")

# ---- Evaluate across the WHOLE test season ----
all_probs, all_preds = [], []
for _, r in test.iterrows():
    mat = scoreline_matrix(r.ExpHomeGoals, r.ExpAwayGoals)
    p_h, p_d, p_a = match_outcome_probs(mat)
    all_probs.append([p_a, p_d, p_h])   # order A, D, H
    all_preds.append(["A", "D", "H"][np.argmax([p_a, p_d, p_h])])

all_probs = np.array(all_probs)
all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)   # force each row to sum to exactly 1

acc = accuracy_score(test["FTR"], all_preds)
ll = log_loss(test["FTR"], all_probs, labels=["A", "D", "H"])

print(f"\nPoisson goals model — Accuracy: {acc:.3f}  Log loss: {ll:.3f}")
print("Logistic Regression (Model 2) was: 0.489 accuracy, 1.037 log loss")