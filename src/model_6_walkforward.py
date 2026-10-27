"""
model_6_walkforward.py
----------------------
The honest accuracy test.

Instead of judging on ONE season (about 380 matches, where luck swings accuracy
by +-5 points), this tests season after season: train on everything before a
season, predict that season, move on. Then it pools every prediction.

Compares: always-pick-the-usual-result, the form model, the bookmakers'
own probabilities, and the form model + odds together.
Also measures how often a pick is right at each confidence level.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

FORM = ["H_L5_Pts", "H_L5_GF", "H_L5_GA", "A_L5_Pts", "A_L5_GF", "A_L5_GA",
        "HomeForm", "AwayForm", "H2H_HomePts",
        "H_L5_S", "H_L5_ST", "H_L5_C", "A_L5_S", "A_L5_ST", "A_L5_C"]
ODDS = ["B365H", "B365D", "B365A"]
LABELS = ["A", "D", "H"]
CURRENT_SEASON = "2026-2027"

m = pd.read_csv("data/processed/features_v2.csv")
missing = [c for c in ODDS if c not in m.columns]
if missing:
    print("The odds columns are missing from features_v2.csv.")
    print("Run: python src\\load_data.py   then   python src\\build_features.py")
    raise SystemExit(1)

m = m[m["Season"] != CURRENT_SEASON].dropna(subset=FORM + ODDS)
m = m[(m[ODDS] > 1.0).all(axis=1)].reset_index(drop=True)

inv = 1.0 / m[ODDS].values
market = inv / inv.sum(axis=1, keepdims=True)                     # remove the bookmaker's margin
m["mH"], m["mD"], m["mA"] = market[:, 0], market[:, 1], market[:, 2]
m["lo_HA"] = np.log(m["mH"] / m["mA"])
m["lo_D"] = np.log(m["mD"] / np.sqrt(m["mH"] * m["mA"]))

seasons = sorted(m["Season"].unique())
first_test = 5 if len(seasons) > 6 else 1
test_seasons = seasons[first_test:]
print(f"{len(m)} matches, {len(seasons)} seasons. Testing on {test_seasons[0]} to {test_seasons[-1]} "
      f"({len(test_seasons)} seasons), always training only on earlier seasons.\n")


def fit_predict(train, test, feats):
    scaler = StandardScaler().fit(train[feats])
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(train[feats]), train["FTR"])
    p = clf.predict_proba(scaler.transform(test[feats]))
    cls = list(clf.classes_)
    return p[:, [cls.index(c) for c in LABELS]]


NAMES = ["baseline", "form", "bookmaker", "odds+form"]
P = {k: [] for k in NAMES}
Y, rows = [], []
for s in test_seasons:
    train, test = m[m["Season"] < s], m[m["Season"] == s]
    freq = train["FTR"].value_counts(normalize=True)
    p = {
        "baseline": np.tile([freq.get(c, 0) for c in LABELS], (len(test), 1)),
        "form": fit_predict(train, test, FORM),
        "bookmaker": test[["mA", "mD", "mH"]].values,
        "odds+form": fit_predict(train, test, FORM + ["lo_HA", "lo_D"]),
    }
    y = test["FTR"].values
    Y.append(y)
    row = {"season": s}
    for k in NAMES:
        P[k].append(p[k])
        row[k] = accuracy_score(y, np.array(LABELS)[p[k].argmax(1)])
    rows.append(row)

print("Accuracy by season:")
print(pd.DataFrame(rows).set_index("season").round(3).to_string())

Y = np.concatenate(Y)
print(f"\nAll {len(Y)} test matches pooled:")
print(f"{'model':<12}{'accuracy':>9}{'log loss':>10}{'  +-95%':>9}")
summary = {}
for k in NAMES:
    pp = np.vstack(P[k])
    acc = accuracy_score(Y, np.array(LABELS)[pp.argmax(1)])
    ll = log_loss(Y, pp, labels=LABELS)
    print(f"{k:<12}{acc:>9.3f}{ll:>10.3f}{1.96 * np.sqrt(acc * (1 - acc) / len(Y)):>9.3f}")
    summary[k] = {"acc": round(float(acc), 4), "ll": round(float(ll), 4)}

bands = [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]
table = {}
for key, name in [("market", "bookmaker"), ("model", "form")]:
    pp = np.vstack(P[name])
    top, pick = pp.max(1), np.array(LABELS)[pp.argmax(1)]
    table[key] = []
    print(f"\nHow often the {key}'s top pick is right, by confidence:")
    for lo, hi in bands:
        sel = (top >= lo) & (top < hi)
        if sel.sum():
            hit = float(np.mean(pick[sel] == Y[sel]))
            table[key].append({"lo": lo, "hi": hi, "n": int(sel.sum()), "hit": round(hit, 3)})
            print(f"  top probability {lo:.1f}-{min(hi, 1):.1f}: {int(sel.sum()):5} matches, right {hit:.0%}")

with open("data/processed/confidence_table.json", "w") as f:
    json.dump(table, f, indent=1)
with open("data/processed/walkforward_seasons.json", "w") as f:
    json.dump({"seasons": [{k: (round(float(v), 4) if k != "season" else v) for k, v in row.items()} for row in rows]}, f, indent=1)
with open("data/processed/walkforward_summary.json", "w") as f:
    json.dump({"first": test_seasons[0], "last": test_seasons[-1], "matches": int(len(Y)), "models": summary}, f, indent=1)
print("\nSaved confidence_table.json, walkforward_summary.json and walkforward_seasons.json (the app uses all three).")
