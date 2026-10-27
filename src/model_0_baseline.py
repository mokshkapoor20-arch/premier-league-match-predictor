import pandas as pd
from sklearn.metrics import accuracy_score

matches = pd.read_csv("data/processed/matches_combined.csv")

# Only use completed matches — drop the current in-progress season for now,
# since some of that data might be genuinely upcoming fixtures
matches = matches[matches["Season"] != "2026-2027"]

matches["Prediction"] = "H"  # naive rule: always predict home win

accuracy = accuracy_score(matches["FTR"], matches["Prediction"])
print(f"Model 0 (always predict home win) accuracy: {accuracy:.3f}")
print(matches["FTR"].value_counts(normalize=True))