import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

m = pd.read_csv("data/processed/features_v1.csv")
m = m.dropna(subset=["H_L5_Pts", "A_L5_Pts"])   # drop matches with no form history
m = m[m["Season"] != "2026-2027"]               # unfinished season

# Split by TIME: learn from the past, test on a season the model never saw
train = m[m["Season"] < "2025-2026"]
test = m[m["Season"] == "2025-2026"]
print("Train matches:", len(train), "| Test matches:", len(test))

features = ["H_L5_Pts", "H_L5_GF", "H_L5_GA", "A_L5_Pts", "A_L5_GF", "A_L5_GA"]
model = LogisticRegression(max_iter=1000)
model.fit(train[features], train["FTR"])

pred = model.predict(test[features])
probs = model.predict_proba(test[features])

acc = accuracy_score(test["FTR"], pred)
baseline = (test["FTR"] == "H").mean()
ll = log_loss(test["FTR"], probs, labels=model.classes_)
print("\nModel 1 accuracy:", round(acc, 3))
print("Baseline (always Home) on same matches:", round(baseline, 3))
print("Log loss:", round(ll, 3), "(coin-flip among 3 outcomes = 1.099, lower is better)")

print("\nConfusion matrix (rows = actual, columns = predicted, order A, D, H):")
print(confusion_matrix(test["FTR"], pred, labels=["A", "D", "H"]))

print("\nFirst test match:", test[["HomeTeam", "AwayTeam", "FTR"]].iloc[0].to_dict())
print("Model probabilities:", dict(zip(model.classes_, probs[0].round(2))))