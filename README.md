# Premier League Match Predictor

An interactive football predictor built around one question:

**How accurately can a data-driven system predict a Premier League match using only information available before kick-off?**

Pick any matchweek, tap **▶ Predict** on a match, and watch it play out: the two club badges slide in, a mini dot-match runs, the model's pick knocks the other badge off the screen, and then the numbers appear one by one (win / draw / loss probabilities, expected goals, likely scorelines, each player's chance to score, and the reasons behind the pick).

![alt text](<Screenshot 2026-10-22 182119.png>)
![alt text](<Screenshot 2026-10-22 182124.png>)
![alt text](<Screenshot 2026-10-22 182143.png>)
![alt text](<Screenshot 2026-10-22 182136.png>)
![alt text](<Screenshot 2026-10-22 182155.png>)


**Live app:** (http://localhost:8501)

> **How this was built.** 
I started as a beginner, learned pandas and scikit-learn first. then tried making I wrote the code myself while sometimes using external help to fix  issues which werent noticable at first glance. caught and fixed a data-leakage problem, and tested the predictor season by season. Then i let it run for few matchweeks and recorded the data.
---

## Results

Football is noisy, so a single season proves very little (accuracy for the same model swings between **47.7% and 57.1%** from one season to the next). I therefore test **season by season, always predicting a season using only earlier seasons** (a "walk-forward" test). That gives **6,828 predictions across 19 seasons (2007-08 to 2025-26)**.

| Method | Accuracy | Log loss | 95% margin |
|---|---|---|---|
| Always pick the usual result (home win) | 45.2% | 1.066 | ±1.2 |
| **This form model** (no odds) | **52.1%** | 0.997 | ±1.2 |
| Bookmaker odds (Bet365, margin removed) | 55.0% | 0.956 | ±1.2 |
| Form model + odds together | 54.6% | 0.961 | ±1.2 |

(Lower log loss is better. It scores the probabilities, not just the top pick.)

**What this shows**

1. **The model beats guessing by about 7 points** over 19 seasons, well outside the margin of error.
2. **Bookmakers are better**, at about 55%, and adding my form features on top of their odds did not help (the odds already contain that information). So the app uses the odds whenever they are published, and says so on screen ("Bookmaker-informed" vs "Form model only").
3. **Nobody gets much above 55%.** A tool claiming 70%+ across all matches is almost always leaking future information.
4. **Confidence matters.** The more confident a pick, the more often it is right (bookmaker picks, 6,828 matches):

| Favourite's probability | Matches | Pick was right |
|---|---|---|
| under 40% | 1,104 | 38% |
| 40-50% | 2,091 | 45% |
| 50-60% | 1,471 | 56% |
| 60-70% | 1,171 | 68% |
| 70%+ | 991 | 78% |

Every prediction in the app shows a confidence tag with this information. The form model alone behaves the same way (37% → 74% across the same bands).

### Earlier model experiments (single season, 2025-26, 374 matches)

Before the walk-forward test I compared models on one held-out season. These numbers are noisy (about ±5 points), but the pattern held:

| Model | Accuracy |
|---|---|
| Baseline (always home) | 42.5% |
| Last-5 form only | 47.4% |
| + home/away form, head-to-head, shots | 48.9% |
| + Elo rating | 47.9% (slightly worse) |
| Random Forest / XGBoost (untuned) | 47.3% / 45.2% |

- **More information helped, up to a point.**
- **More model complexity did not.** A plain logistic regression beat Random Forest and XGBoost on about 3,700 training matches (the tree models were not tuned).
- **Elo carries real signal on its own (44.1%) but did not add to the form features.**

### Limitations (read these)

- **It never predicts a draw as the most likely result** (0 of 102 draws in the 2025-26 test). It does give draws a probability (about 22-27%).
- **No lineups, tactics or individual injuries beyond the squad feed.** Expected goals (xG) data and rest days were not used.
- **The player model is simple** (scoring rate × minutes × availability, shared out of the team's expected goals). The position starting-rates are rough assumptions, and it has not been checked against past matches.
- **Odds are only available for upcoming fixtures a few days ahead.** Other matches use the form model alone.
- **Small samples early in a season.** A club needs 5 matches in the data before it can be predicted.

---

## How it works

**Data.** 24 complete seasons of results with shots and Bet365 odds (2002-03 onwards, football-data.co.uk), the official Fantasy Premier League feed (all 20 squads with goals, xG, minutes and injury news, plus the full fixture list), and club badges (TheSportsDB).

**Avoiding data leakage.** Every training feature uses only matches played before the match it describes: rolling statistics are shifted by one match, and all train/test splits are by season. A test recomputes the features by hand for 200 random matches (0 mismatches). For a match that has not been played, each club's form is taken from its latest 5 matches *including* the most recent one (`src/live_features.py`); a second test shows this reproduces the training features exactly (5,884 values across 400 past matches, 0 mismatches).

**Features.** Last-5 points, goals for and against; last-5 average shots, shots on target and corners; home team's form at home and away team's form away; head-to-head record.

**Goals and scorelines.** A Poisson model turns each club's attack and defence (shrunk toward the league average so five matches can't mislead it) into expected goals and a grid of scoreline probabilities. It is then fitted to the headline win/draw/loss probabilities, so the page never contradicts itself.

**Prediction record.** Every prediction is saved *before* kick-off. The first prediction for a fixture is the one that counts, a prediction only counts if it was saved before kick-off, and results fill in automatically from the final scores. The record is used to *check* the model, never to train it.

---

## Problems I found and fixed

Most of these were found by testing odd cases and questioning numbers that looked wrong.

| Problem | How it showed up | Fix |
|---|---|---|
| Inconsistent date formats across seasons | 760 unparsed dates | Fallback parsing |
| Matches silently dropped when reading old files | Some seasons had 335 matches instead of 380 | A reader that keeps every row |
| **Live predictions ignored each club's most recent match** | Arsenal showed 15/15 recent points, but their latest result (a loss) was missing | `live_features.py`, plus the equivalence test above |
| Small-sample noise | One side given 3.12 expected goals | Shrinkage toward the league average |
| Stale data for relegated clubs | "Recent form" from May 2024 | Warning when a club's last match is over 400 days old |
| Elo model failed to converge | Optimiser warning (Elo spans ±300, other features single digits) | Standardised features |
| Identical player percentages | Two strikers with different seasons both showed 12% | Rate × minutes × availability |
| Wrong club badges | An unrelated "Leeds Force" crest; for Nottingham Forest only a netball team | Football-only filter, labelled placeholder |
| Free squad API returned 10 players per club | Incomplete squads | Official Fantasy Premier League feed |
| Page contradicted itself | Higher expected goals for the team the model said would lose | Goals fitted to the headline probabilities |
| Record counted unresolved predictions as wrong | "1 resolved, 0% correct" with no result entered | Fixed; first prediction counts; saved-before-kick-off rule |
| Newly promoted club could not be predicted | No prediction for a club with under 5 matches | Clear message instead of a crash |

---

## Run it yourself

```bash
python -m venv venv
venv\Scripts\Activate.ps1                    # Windows PowerShell
pip install -r requirements.txt

python src/download_history.py               # older seasons (once)
python src/refresh_all.py                    # latest results, features, squads, fixtures
python src/model_6_walkforward.py            # the season-by-season accuracy test
python -m streamlit run app.py
```

On the public website, set the environment variable `PL_PUBLIC=1` (Streamlit Cloud: App settings → Secrets → `PL_PUBLIC = "1"`). Visitors can explore, but nothing is written to the official record.

```
app.py                       the Streamlit app
src/
  live_features.py           form going into a match that hasn't been played
  predict_match.py           win/draw/loss model + goals model
  market.py                  bookmaker odds, confidence, consistent goals
  player_model.py            who is likely to score
  tracker.py                 prediction record (auto-resolves from final scores)
  arena.py, arena_template.html   the animated match arena
  fixtures_view.py           matchweek browser helpers
  load_data.py, build_features.py, update_data.py, refresh_all.py   data pipeline
  fetch_players.py, fetch_fixtures.py, fetch_badges.py              squads, fixtures, badges
  model_0 ... model_6        the experiments, including the walk-forward test
data/raw, data/processed     season files, features, squads, fixtures, prediction record
static/                      club badges
```

## Data sources and credits

- Match results, shots and odds: [football-data.co.uk](https://www.football-data.co.uk)
- Squads, injuries and fixtures: the official Fantasy Premier League data feed
- Club badges: TheSportsDB. Badges are trademarks of their clubs, shown for illustration only.
- This is a student learning project. **It is not betting advice.**

## Still to do

- Check the player model against past matches; tune Random Forest and XGBoost
- Add rest days / fixture congestion and test whether they help
- Simulate the rest of the season (title, top-4 and relegation probabilities) from the same goals model
