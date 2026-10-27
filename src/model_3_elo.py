import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

m = pd.read_csv("data/processed/features_v3.csv")

base_features = ["H_L5_Pts", "H_L5_GF", "H_L5_GA", "A_L5_Pts", "A_L5_GF", "A_L5_GA",
                  "HomeForm", "AwayForm", "H2H_HomePts",
                  "H_L5_S", "H_L5_ST", "H_L5_C", "A_L5_S", "A_L5_ST", "A_L5_C"]
elo_features = base_features + ["EloDiff"]

m = m.dropna(subset=elo_features)
m = m[m["Season"] != "2026-2027"]

train = m[m["Season"] < "2025-2026"]
test = m[m["Season"] == "2025-2026"]

def run(features, label):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train["FTR"])
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(test["FTR"], pred)
    ll = log_loss(test["FTR"], probs, labels=model.classes_)
    print(f"{label:<25} accuracy: {acc:.3f}   log loss: {ll:.3f}")

run(base_features, "Model 2 (no Elo)")
run(elo_features, "Model 3 (with Elo)")

run(["EloDiff"], "Elo alone")