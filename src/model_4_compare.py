import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

m = pd.read_csv("data/processed/features_v2.csv")

features = ["H_L5_Pts", "H_L5_GF", "H_L5_GA", "A_L5_Pts", "A_L5_GF", "A_L5_GA",
            "HomeForm", "AwayForm", "H2H_HomePts",
            "H_L5_S", "H_L5_ST", "H_L5_C", "A_L5_S", "A_L5_ST", "A_L5_C"]

m = m.dropna(subset=features)
m = m[m["Season"] != "2026-2027"]

train = m[m["Season"] < "2025-2026"]
test = m[m["Season"] == "2025-2026"]

# XGBoost needs numeric labels, not letters
label_map = {"A": 0, "D": 1, "H": 2}
y_train = train["FTR"].map(label_map)
y_test = test["FTR"].map(label_map)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              eval_metric="mlogloss", random_state=1),
}

print(f"{'Model':<22}{'Accuracy':<12}{'Log loss'}")
for name, mdl in models.items():
    mdl.fit(train[features], y_train)
    pred = mdl.predict(test[features])
    probs = mdl.predict_proba(test[features])
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, probs)
    print(f"{name:<22}{acc:<12.3f}{ll:.3f}")