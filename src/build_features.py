import pandas as pd
m = pd.read_csv("data/processed/matches_combined.csv", parse_dates=["Date"])
m = m.reset_index().rename(columns={"index": "MatchID"})

# one row per TEAM per match
home = pd.DataFrame({"MatchID": m.MatchID, "Date": m.Date, "Team": m.HomeTeam, "GF": m.FTHG, "GA": m.FTAG})
away = pd.DataFrame({"MatchID": m.MatchID, "Date": m.Date, "Team": m.AwayTeam, "GF": m.FTAG, "GA": m.FTHG})
log = pd.concat([home, away]).sort_values(["Team", "Date", "MatchID"]).reset_index(drop=True)
log["Pts"] = (log.GF > log.GA) * 3 + (log.GF == log.GA) * 1

# shift(1) = exclude the current match. This line is what prevents leakage.
for c in ["Pts", "GF", "GA"]:
    log["L5_" + c] = log.groupby("Team")[c].transform(lambda s: s.shift(1).rolling(5, min_periods=5).sum())

feat = log[["MatchID", "Team", "L5_Pts", "L5_GF", "L5_GA"]]
m = m.merge(feat.add_prefix("H_"), left_on=["MatchID", "HomeTeam"], right_on=["H_MatchID", "H_Team"])
m = m.merge(feat.add_prefix("A_"), left_on=["MatchID", "AwayTeam"], right_on=["A_MatchID", "A_Team"])
m = m.drop(columns=["H_MatchID", "H_Team", "A_MatchID", "A_Team"])

m.to_csv("data/processed/features_v1.csv", index=False)
print(m[["Date", "HomeTeam", "AwayTeam", "H_L5_Pts", "A_L5_Pts"]].tail())
print("Rows with NaN form (each team's first 5 games):", m.H_L5_Pts.isna().sum())

# Leakage test: recompute form by hand for 200 random matches
check = m.dropna(subset=["H_L5_Pts", "A_L5_Pts"]).sample(200, random_state=1)
bad = 0
for _, r in check.iterrows():
    home_past = log[(log.Team == r.HomeTeam) & (log.Date < r.Date)].tail(5)
    away_past = log[(log.Team == r.AwayTeam) & (log.Date < r.Date)].tail(5)
    if home_past.Pts.sum() != r.H_L5_Pts or away_past.Pts.sum() != r.A_L5_Pts:
        bad += 1
print("Leakage test, mismatches out of 200:", bad)

# ---------- Home/away split form ----------
# Cleaner way: rebuild home/away logs separately, same idea as Step 3 but split
h_log = pd.DataFrame({"MatchID": m.MatchID, "Date": m.Date, "Team": m.HomeTeam, "Pts": (m.FTHG > m.FTAG)*3 + (m.FTHG == m.FTAG)*1})
a_log = pd.DataFrame({"MatchID": m.MatchID, "Date": m.Date, "Team": m.AwayTeam, "Pts": (m.FTAG > m.FTHG)*3 + (m.FTHG == m.FTAG)*1})

h_log = h_log.sort_values(["Team", "Date"]).reset_index(drop=True)
a_log = a_log.sort_values(["Team", "Date"]).reset_index(drop=True)
h_log["HomeForm"] = h_log.groupby("Team")["Pts"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
a_log["AwayForm"] = a_log.groupby("Team")["Pts"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())

m = m.merge(h_log[["MatchID", "Team", "HomeForm"]], left_on=["MatchID", "HomeTeam"], right_on=["MatchID", "Team"]).drop(columns="Team")
m = m.merge(a_log[["MatchID", "Team", "AwayForm"]], left_on=["MatchID", "AwayTeam"], right_on=["MatchID", "Team"]).drop(columns="Team")

# ---------- Head-to-head ----------
# Average points the home team earned in their last 3 meetings with THIS opponent
def h2h_points(row, matches_df):
    past = matches_df[
        (((matches_df.HomeTeam == row.HomeTeam) & (matches_df.AwayTeam == row.AwayTeam)) |
         ((matches_df.HomeTeam == row.AwayTeam) & (matches_df.AwayTeam == row.HomeTeam)))
        & (matches_df.Date < row.Date)
    ].tail(3)
    if len(past) == 0:
        return None
    pts = 0
    for _, p in past.iterrows():
        if p.HomeTeam == row.HomeTeam:
            pts += 3 if p.FTHG > p.FTAG else (1 if p.FTHG == p.FTAG else 0)
        else:
            pts += 3 if p.FTAG > p.FTHG else (1 if p.FTHG == p.FTAG else 0)
    return pts / len(past)

m["H2H_HomePts"] = m.apply(lambda r: h2h_points(r, m), axis=1)

# ---------- Rolling shots / shots on target / corners ----------
h_stats = pd.DataFrame({"MatchID": m.MatchID, "Date": m.Date, "Team": m.HomeTeam, "S": m.HS, "ST": m.HST, "C": m.HC})
a_stats = pd.DataFrame({"MatchID": m.MatchID, "Date": m.Date, "Team": m.AwayTeam, "S": m.AS, "ST": m.AST, "C": m.AC})
stat_log = pd.concat([h_stats, a_stats]).sort_values(["Team", "Date"]).reset_index(drop=True)

for col in ["S", "ST", "C"]:
    stat_log["L5_" + col] = stat_log.groupby("Team")[col].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())

stat_feat = stat_log[["MatchID", "Team", "L5_S", "L5_ST", "L5_C"]]
m = m.merge(stat_feat.add_prefix("H_"), left_on=["MatchID", "HomeTeam"], right_on=["H_MatchID", "H_Team"]).drop(columns=["H_MatchID", "H_Team"])
m = m.merge(stat_feat.add_prefix("A_"), left_on=["MatchID", "AwayTeam"], right_on=["A_MatchID", "A_Team"]).drop(columns=["A_MatchID", "A_Team"])

m.to_csv("data/processed/features_v2.csv", index=False)
print(m.shape)
print(m[["Date","HomeTeam","AwayTeam","HomeForm","AwayForm","H2H_HomePts","H_L5_S","A_L5_S"]].tail())

# ---------- Elo ratings ----------
elo = {team: 1500 for team in pd.concat([m.HomeTeam, m.AwayTeam]).unique()}
K = 32
HOME_ADV = 75  # points added to the home team's rating, only for the expected-score calc

home_elo_before, away_elo_before = [], []

for _, r in m.sort_values("Date").iterrows():
    h_rating = elo[r.HomeTeam]
    a_rating = elo[r.AwayTeam]

    home_elo_before.append(h_rating)
    away_elo_before.append(a_rating)

    expected_home = 1 / (1 + 10 ** (-(h_rating + HOME_ADV - a_rating) / 400))

    if r.FTR == "H":
        actual_home = 1.0
    elif r.FTR == "D":
        actual_home = 0.5
    else:
        actual_home = 0.0

    elo[r.HomeTeam] = h_rating + K * (actual_home - expected_home)
    elo[r.AwayTeam] = a_rating + K * ((1 - actual_home) - (1 - expected_home))

# m was already sorted by Date, so this lines up correctly
m["H_Elo"] = home_elo_before
m["A_Elo"] = away_elo_before
m["EloDiff"] = m["H_Elo"] - m["A_Elo"]

m.to_csv("data/processed/features_v3.csv", index=False)
print(m[["Date", "HomeTeam", "AwayTeam", "H_Elo", "A_Elo", "EloDiff"]].tail(10))

# sanity check: current top 10 teams by rating
current = pd.Series(elo).sort_values(ascending=False)
print("\nCurrent Elo ratings, top 10:")
print(current.head(10))