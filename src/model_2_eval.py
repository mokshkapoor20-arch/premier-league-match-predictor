import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, brier_score_loss
from sklearn.calibration import calibration_curve

m = pd.read_csv("data/processed/features_v2.csv")

features = ["H_L5_Pts", "H_L5_GF", "H_L5_GA", "A_L5_Pts", "A_L5_GF", "A_L5_GA",
            "HomeForm", "AwayForm", "H2H_HomePts",
            "H_L5_S", "H_L5_ST", "H_L5_C", "A_L5_S", "A_L5_ST", "A_L5_C"]

m = m.dropna(subset=features)
m = m[m["Season"] != "2026-2027"]

train = m[m["Season"] < "2025-2026"]
test = m[m["Season"] == "2025-2026"]

scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["FTR"])

pred = model.predict(X_test)
probs = model.predict_proba(X_test)

print("=== Classification report ===")
print(classification_report(test["FTR"], pred, target_names=["Away", "Draw", "Home"]))

home_idx = list(model.classes_).index("H")
actual_home = (test["FTR"] == "H").astype(int)
predicted_prob_home = probs[:, home_idx]

frac_actual, mean_predicted = calibration_curve(actual_home, predicted_prob_home, n_bins=8)

print("\n=== Calibration (Home win) ===")
for pred_p, act_p in zip(mean_predicted, frac_actual):
    print(f"Predicted ~{pred_p:.2f}  ->  Actually happened {act_p:.2f}")

brier = brier_score_loss(actual_home, predicted_prob_home)
print(f"\nBrier score (Home): {brier:.3f}  (0 = perfect, 0.25 = random guessing at 50%)")

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
plt.plot(mean_predicted, frac_actual, marker="o", label="Model 2")
plt.xlabel("Predicted probability of Home win")
plt.ylabel("Actual fraction of Home wins")
plt.title("Calibration curve — Home win predictions")
plt.legend()
plt.savefig("data/processed/calibration_home.png")
print("Saved plot to data/processed/calibration_home.png")

print("\n=== Feature importance (Home-win direction) ===")
coefs = pd.Series(model.coef_[home_idx], index=features).sort_values(key=abs, ascending=False)
print(coefs)