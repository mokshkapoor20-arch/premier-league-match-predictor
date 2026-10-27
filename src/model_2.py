import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

m = pd.read_csv("data/processed/features_v2.csv")

features = ["H_L5_Pts", "H_L5_GF", "H_L5_GA", "A_L5_Pts", "A_L5_GF", "A_L5_GA",
            "HomeForm", "AwayForm", "H2H_HomePts",
            "H_L5_S", "H_L5_ST", "H_L5_C", "A_L5_S", "A_L5_ST", "A_L5_C"]

m = m.dropna(subset=features)
m = m[m["Season"] != "2026-2027"]

train = m[m["Season"] < "2025-2026"]
test = m[m["Season"] == "2025-2026"]
print("Train matches:", len(train), "| Test matches:", len(test))

model = LogisticRegression(max_iter=1000)
model.fit(train[features], train["FTR"])

pred = model.predict(test[features])
probs = model.predict_proba(test[features])

acc = accuracy_score(test["FTR"], pred)
baseline = (test["FTR"] == "H").mean()
ll = log_loss(test["FTR"], probs, labels=model.classes_)

print("\nModel 2 accuracy:", round(acc, 3))
print("Model 1 accuracy was: 0.474   (compare against this, not the baseline)")
print("Baseline (always Home) on same matches:", round(baseline, 3))
print("Log loss:", round(ll, 3), "  (Model 1 was 1.058)")

print("\nConfusion matrix (rows=actual, cols=predicted, order A, D, H):")
print(confusion_matrix(test["FTR"], pred, labels=["A", "D", "H"]))