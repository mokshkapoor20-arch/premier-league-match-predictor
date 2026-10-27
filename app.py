import json
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
from arena import render_arena
from fixtures_view import badge_uri, group_by_week, initials, kickoff_parts, load_fixtures, next_week
from live import get_fixtures, live_games, live_info, next_kickoff, score_text
from market import apply_market, market_probs
from predict_match import NotEnoughHistory, predict_match
from tracker import auto_resolve, breakdown, compute_stats, load_log, log_prediction, record_result

st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="centered")

# On the public website set PL_PUBLIC=1: visitors can explore, but nothing is saved to your official record.
PUBLIC = os.environ.get("PL_PUBLIC", "") == "1"

TEAMS = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
         'Cardiff', 'Chelsea', 'Coventry', 'Crystal Palace', 'Everton', 'Fulham',
         'Huddersfield', 'Hull', 'Ipswich', 'Leeds', 'Leicester', 'Liverpool', 'Luton',
         'Man City', 'Man United', 'Middlesbrough', 'Newcastle', 'Norwich',
         "Nott'm Forest", 'QPR', 'Sheffield United', 'Southampton', 'Stoke',
         'Sunderland', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham',
         'Wolves']

st.markdown("""<style>
.weektitle{text-align:center;line-height:1.2}
.weektitle .wk{font-size:1.55rem;font-weight:900;letter-spacing:.08em}
.weektitle .sub{color:#9aa3b6;font-size:.85rem}
.daybar{margin:1.2rem 0 .5rem;font-weight:800;letter-spacing:.14em;font-size:.78rem;color:#9aa3b6;
        text-transform:uppercase;border-bottom:1px solid #252c3e;padding-bottom:.35rem}
.crest{text-align:center}
.crest img{width:54px;height:54px;object-fit:contain}
.crest .ini{width:54px;height:54px;border-radius:50%;background:#445066;color:#fff;font-weight:800;
            display:inline-flex;align-items:center;justify-content:center}
.tname{text-align:center;font-weight:700;font-size:.9rem;margin-top:.15rem}
.mid{text-align:center;line-height:1.3}
.mid .vs{font-weight:900;letter-spacing:.16em;color:#8b93a7}
.mid .score{font-weight:900;font-size:1.6rem}
.mid .sub{color:#9aa3b6;font-size:.76rem}
.odds{text-align:center;color:#4ade80;font-size:.72rem;font-weight:700;letter-spacing:.06em;margin-top:.1rem}
.foot{color:#7d869a;font-size:.78rem;line-height:1.6}
.mid .score.live{color:#ff5a67}
.livepill{display:inline-block;color:#fff;background:#e5303f;border-radius:999px;padding:.05rem .55rem;
          font-size:.72rem;font-weight:800;letter-spacing:.06em;animation:pl-pulse 1.6s ease-in-out infinite}
@keyframes pl-pulse{0%,100%{opacity:1}50%{opacity:.55}}
.livebar{display:flex;justify-content:space-between;align-items:center;gap:.6rem;padding:.45rem .8rem;margin:.25rem 0;
         border:1px solid #5a2a31;background:#2a1519;border-radius:10px;font-weight:700;font-size:.92rem}
.livebar .sc{font-weight:900;letter-spacing:.04em}
.closed{text-align:center;color:#9aa3b6;font-size:.74rem;margin-top:.1rem}
.stamp{text-align:center;color:#7d869a;font-size:.74rem;margin:.9rem 0 0}
</style>""", unsafe_allow_html=True)

st.title("⚽ Premier League Match Predictor")
st.caption("Pick a match, tap ▶ Predict, and watch it play out. Built only from information available before kick-off.")

with st.expander("How this works, and how accurate it is"):
    st.markdown("""
**Data.** 24 seasons of Premier League results (football-data.co.uk), the official Fantasy Premier League feed
(squads, injuries, fixtures), and Bet365 pre-match odds when they are published.

**Model.** A logistic regression turns each team's last 5 matches (points, goals, shots), home and away form and
head-to-head record into win / draw / loss probabilities. A Poisson goals model gives expected goals and likely
scorelines. Player chances come from each player's scoring rate, minutes and availability.

**How accurate?** Tested season by season, always predicting a season using only earlier seasons
(6,828 matches): always picking the usual result is right 45% of the time, this form model 52%, and the bookmakers
55%. Football is noisy: nobody gets much above 55%, and a pick is far more reliable when its confidence is high.

**Honest limits.** It never predicts a draw as the most likely result. It does not know about injuries to
individual players beyond the squad feed, lineups, or tactics. This is a learning project, not betting advice.
""")
if PUBLIC:
    st.caption("Public demo: predictions made here are not saved. The author's official record is further down.")

if not PUBLIC:
    newly_resolved = auto_resolve()
    if newly_resolved:
        st.toast(f"{newly_resolved} prediction(s) marked right or wrong from the final scores")


def crest_html(team):
    uri = badge_uri(team)
    inner = f'<img src="{uri}" alt="{team}">' if uri else f'<span class="ini">{initials(team)}</span>'
    return f'<div class="crest">{inner}</div><div class="tname">{team}</div>'


# ---------- the prediction pop-ups ----------
def open_prediction(home, away):
    @st.dialog(f"{home} vs {away}", width="large")
    def _show():
        try:
            with st.spinner("Calculating..."):
                result = apply_market(predict_match(home, away), home, away)
                logged = None
                if not PUBLIC:
                    logged = log_prediction(home, away, result["p_home"], result["p_draw"], result["p_away"],
                                            source=result["source"], confidence=result["confidence"]["label"])
        except NotEnoughHistory as e:
            st.error(f"Can't predict this one yet. {e} It will be available after a few more matchweeks.")
            return
        for w in result.get("warnings", []):
            st.warning(w)
        render_arena(home, away, result)
        if PUBLIC:
            st.caption("Public demo: this prediction was not saved.")
        else:
            st.caption("📌 Saved to your prediction record. It is marked right or wrong automatically after the match."
                       if logged else
                       "📌 Already in your prediction record. The first prediction for a fixture is the one that counts.")

        st.subheader("Recent form")
        h, a = result["form"]["home"], result["form"]["away"]
        names = ["Points", "Goals for", "Goals against", "Shots (avg)", "On target (avg)"]
        hv = [h["Pts"], h["GF"], h["GA"], h["S"], h["ST"]]
        av = [a["Pts"], a["GF"], a["GA"], a["S"], a["ST"]]
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(8, 3.6))
            fig.patch.set_facecolor("#141926")
            ax.set_facecolor("#141926")
            x = range(len(names))
            ax.bar([i - 0.18 for i in x], hv, 0.36, label=home, color="#4c8dff")
            ax.bar([i + 0.18 for i in x], av, 0.36, label=away, color="#ff5a67")
            ax.set_xticks(list(x))
            ax.set_xticklabels(names)
            ax.set_title("Last 5 matches, side by side", color="#eef0f5")
            ax.legend(frameon=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            st.pyplot(fig)
    _show()


def open_week(upcoming, week):
    @st.dialog(f"Matchweek {week}: all predictions", width="large")
    def _show():
        rows = []
        with st.spinner("Predicting the whole matchweek..."):
            for g in upcoming:
                name = f"{g['home']} vs {g['away']}"
                try:
                    r = apply_market(predict_match(g["home"], g["away"]), g["home"], g["away"])
                except NotEnoughHistory:
                    rows.append({"Match": name, "Pick": "not enough history yet", "Chance": "", "Confidence": "",
                                 "Based on": "", "Record": ""})
                    continue
                probs = {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]}
                best = max(probs, key=probs.get)
                saved = None if PUBLIC else log_prediction(g["home"], g["away"], r["p_home"], r["p_draw"], r["p_away"],
                                                          source=r["source"], confidence=r["confidence"]["label"])
                rows.append({
                    "Match": name,
                    "Pick": {"H": g["home"], "D": "Draw", "A": g["away"]}[best],
                    "Chance": f"{probs[best]*100:.0f}%",
                    "Confidence": r["confidence"]["label"],
                    "Based on": "Bookmaker odds" if r["source"] == "market" else "Form model",
                    "Record": "not saved (demo)" if PUBLIC else ("saved" if saved else "already saved"),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Picks are saved to your prediction record and marked right or wrong automatically after the matches."
                   if not PUBLIC else "Public demo: nothing here is saved.")
    _show()


# ---------- choosing a match ----------
def _until(ko):
    mins = int((ko - datetime.now(timezone.utc)).total_seconds() // 60)
    if mins < 60:
        return f"{max(mins, 1)} min"
    if mins < 60 * 48:
        return f"{mins // 60}h {mins % 60:02d}m"
    return f"{mins // 1440} days"


@st.fragment(run_every=30)
def match_browser():
    """The matchweek browser. It re-runs by itself every 30 seconds so scores and live minutes stay current."""
    fixtures, source = get_fixtures()
    weeks = group_by_week(fixtures)
    if not weeks:
        st.caption("Fixture list not available yet. Check your internet connection, or run  python src\\refresh_all.py")
        return
    if not PUBLIC and source == "live":
        n = auto_resolve()
        if n:
            st.toast(f"{n} prediction(s) marked right or wrong from the final scores")

    now = datetime.now(timezone.utc)
    live_now = live_games(fixtures, now)
    if live_now:
        st.markdown("<div class='daybar'>● Live now</div>", unsafe_allow_html=True)
        for g in live_now:
            info = live_info(g, now)
            st.markdown(f"<div class='livebar'><span>{g['home']}</span>"
                        f"<span class='sc'>{score_text(g)}</span><span>{g['away']}</span>"
                        f"<span class='livepill'>{info['label']}</span></div>", unsafe_allow_html=True)
        st.write("")
    else:
        nk = next_kickoff(fixtures, now)
        if nk:
            from live import kickoff_dt
            st.caption(f"No match live right now. Next kick-off: {nk['home']} vs {nk['away']} in {_until(kickoff_dt(nk))}.")

    week_keys = list(weeks)
    default_week = next_week(weeks)
    if st.session_state.get("week") not in weeks:
        st.session_state["week"] = default_week

    def go_week(delta):
        i = week_keys.index(st.session_state["week"]) + delta
        st.session_state["week"] = week_keys[max(0, min(len(week_keys) - 1, i))]

    week = st.session_state["week"]
    games = weeks[week]
    infos = {id(g): live_info(g, now) for g in games}
    upcoming = [g for g in games if infos[id(g)]["state"] == "upcoming"]
    n_live = sum(1 for g in games if infos[id(g)]["state"] == "live")

    nav_l, nav_c, nav_r = st.columns([1, 4, 1], vertical_alignment="center")
    nav_l.button("◀", key="prev_week", on_click=go_week, args=(-1,), disabled=week == week_keys[0], width="stretch")
    status = ("live now" if n_live else "next up" if week == default_week
              else ("played" if all(g["finished"] for g in games) else "upcoming"))
    nav_c.markdown(f"<div class='weektitle'><div class='wk'>MATCHWEEK {week}</div>"
                   f"<div class='sub'>{status} · {len(games)} matches</div></div>", unsafe_allow_html=True)
    nav_r.button("▶", key="next_week", on_click=go_week, args=(1,), disabled=week == week_keys[-1], width="stretch")
    with st.expander("Jump to a matchweek"):
        st.selectbox("Matchweek", week_keys, key="week",
                     format_func=lambda w: f"Matchweek {w}" + ("  ·  next up" if w == default_week else ""))

    if upcoming and st.button(f"⚡ Predict all {len(upcoming)} upcoming match{'es' if len(upcoming) != 1 else ''} in Matchweek {week}", width="stretch"):
        st.session_state["open_week"] = (week, upcoming)
        st.rerun()                                   # full re-run, so the pop-up is not wiped by the 30s refresh
    if all(g["finished"] for g in games):
        st.info("Every match in this matchweek has been played. Use ▶ to move to the next one.")

    by_day = {}
    for g in games:
        by_day.setdefault(kickoff_parts(g["kickoff"])["day"], []).append(g)

    for day, day_games in by_day.items():
        st.markdown(f"<div class='daybar'>{day}</div>", unsafe_allow_html=True)
        for i in range(0, len(day_games), 2):
            grid = st.columns(2)
            for col, g in zip(grid, day_games[i:i + 2]):
                info = infos[id(g)]
                with col, st.container(border=True):
                    left, mid, right = st.columns([1, 1, 1], vertical_alignment="center")
                    left.markdown(crest_html(g["home"]), unsafe_allow_html=True)
                    right.markdown(crest_html(g["away"]), unsafe_allow_html=True)
                    if info["state"] == "ft":
                        score = score_text(g) if g.get("hs") is not None else "–"
                        mid.markdown(f"<div class='mid'><div class='score'>{score}</div><div class='sub'>Full time</div></div>",
                                     unsafe_allow_html=True)
                    elif info["state"] == "live":
                        mid.markdown(f"<div class='mid'><div class='score live'>{score_text(g)}</div>"
                                     f"<span class='livepill'>● {info['label']}</span></div>", unsafe_allow_html=True)
                        st.markdown("<div class='closed'>Kicked off: predictions are closed</div>", unsafe_allow_html=True)
                    else:
                        p = kickoff_parts(g["kickoff"])
                        mid.markdown(f"<div class='mid'><div class='vs'>VS</div><div class='sub'>{p['uk']} UK</div>"
                                     f"<div class='sub'>{p['ist']} IST</div></div>", unsafe_allow_html=True)
                        if market_probs(g["home"], g["away"]):
                            st.markdown("<div class='odds'>● BOOKMAKER ODDS IN</div>", unsafe_allow_html=True)
                        if st.button("▶ Predict", key=f"go_{g['home']}_{g['away']}", width="stretch"):
                            st.session_state["open_pred"] = (g["home"], g["away"])
                            st.rerun()

    stamp = datetime.now().strftime("%H:%M:%S")
    if source == "live":
        st.markdown(f"<div class='stamp'>Live feed · updated {stamp} · refreshes every 30 seconds"
                    f" · minutes marked ~ are estimated</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stamp'>⚠ Could not reach the live feed at {stamp}. Showing the last saved data; retrying.</div>",
                    unsafe_allow_html=True)


mode = st.radio("Choose a match from", ["Matchweek fixtures", "Any two teams"], horizontal=True)

if mode == "Matchweek fixtures":
    match_browser()
    if "open_pred" in st.session_state:
        open_prediction(*st.session_state.pop("open_pred"))
    elif "open_week" in st.session_state:
        wk, todo = st.session_state.pop("open_week")
        open_week(todo, wk)
else:
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home team", TEAMS, index=TEAMS.index("Chelsea"))
    with col2:
        away_options = [t for t in TEAMS if t != home_team]
        away_team = st.selectbox("Away team", away_options,
                                 index=away_options.index("Arsenal") if "Arsenal" in away_options else 0)
    if st.button(f"▶ Predict {home_team} vs {away_team}", type="primary", width="stretch"):
        open_prediction(home_team, away_team)


# ---------- prediction record ----------
def pretty_record(log):
    if log.empty:
        return log
    pick = lambda code, h, a: {"H": h, "D": "Draw", "A": a}.get(code, "")
    pct = lambda col: (pd.to_numeric(log[col]) * 100).round().astype(int).astype(str) + "%"
    return pd.DataFrame({
        "Saved": log["logged_at"],
        "Match": log["home_team"] + " vs " + log["away_team"],
        "Home": pct("p_home"), "Draw": pct("p_draw"), "Away": pct("p_away"),
        "Pick": [pick(c, h, a) for c, h, a in zip(log["predicted_outcome"], log["home_team"], log["away_team"])],
        "Result": [pick(c, h, a) if c else "…" for c, h, a in zip(log["actual_outcome"], log["home_team"], log["away_team"])],
        "Right?": log["correct"].map({"True": "✓", "False": "✗"}).fillna("…").replace("", "…"),
        "Based on": log["source"].map({"market": "Odds", "model": "Form model"}).fillna(""),
        "Confidence": log["confidence"],
    })


st.divider()
st.header("📊 Prediction Record")
st.caption("Every prediction is saved before kick-off, and results fill in automatically once the final score is in. "
           "The record is used to CHECK the model, not to train it, which keeps the test honest.")

stats = compute_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions made", stats["total"] + stats["pending"])
c2.metric("Resolved", stats["total"])
c3.metric("Correct", stats["correct"])
c4.metric("Accuracy", f"{stats['accuracy']*100:.1f}%" if stats["accuracy"] is not None else "n/a")

if stats["total"] > 0:
    st.markdown("**How the picks did, by confidence**")
    st.dataframe(breakdown("confidence"), hide_index=True, width="stretch")
    st.caption("These numbers only mean something after about 50 matches. Early on, a few results can swing them a lot.")

log = load_log()
pending = log[log["actual_outcome"] == ""]
if len(pending) > 0 and not PUBLIC:
    with st.expander("Enter a real result yourself (only needed if one didn't fill in on its own)"):
        pending_labels = [f"#{row.id}: {row.home_team} vs {row.away_team} (predicted {row.predicted_outcome})"
                          for _, row in pending.iterrows()]
        selected_label = st.selectbox("Which prediction do you want to resolve?", pending_labels)
        selected_id = int(selected_label.split("#")[1].split(":")[0])
        actual = st.radio("What actually happened?", ["H", "D", "A"], horizontal=True,
                          format_func=lambda x: {"H": "Home win", "D": "Draw", "A": "Away win"}[x])
        if st.button("Save result"):
            record_result(selected_id, actual)
            st.success("Recorded! Refresh to see updated stats.")

if log.empty:
    st.caption("No predictions yet. Predict a matchweek before kick-off to start the record.")
else:
    st.dataframe(pretty_record(log), hide_index=True, width="stretch")

# ---------- does it actually work? ----------
st.divider()
st.header("📈 Does it actually work?")

summary_path = "data/processed/walkforward_summary.json"
if os.path.exists(summary_path):
    with open(summary_path) as f:
        wf = json.load(f)
    mods = wf["models"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Always pick the usual result", f"{mods['baseline']['acc']*100:.1f}%")
    c2.metric("This form model", f"{mods['form']['acc']*100:.1f}%",
              f"{(mods['form']['acc']-mods['baseline']['acc'])*100:+.1f} pts vs baseline")
    c3.metric("Bookmaker odds", f"{mods['bookmaker']['acc']*100:.1f}%",
              f"{(mods['bookmaker']['acc']-mods['baseline']['acc'])*100:+.1f} pts vs baseline")
    st.caption(f"Walk-forward test: {wf['matches']} matches from {wf['first']} to {wf['last']}. Each season was "
               "predicted using only earlier seasons, so nothing from the future leaks in.")

    seasons_path = "data/processed/walkforward_seasons.json"
    if os.path.exists(seasons_path):
        with open(seasons_path) as f:
            ps = pd.DataFrame(json.load(f)["seasons"]).set_index("season")
        ps = (ps[["baseline", "form", "bookmaker"]] * 100).round(1)
        ps.columns = ["Always pick the usual result", "This form model", "Bookmaker odds"]
        st.markdown("**Accuracy by season (%)**")
        st.line_chart(ps)
        st.caption("A single season swings by several points either way, which is why one season proves very little.")

    table_path = "data/processed/confidence_table.json"
    if os.path.exists(table_path):
        with open(table_path) as f:
            bands = json.load(f)["market"]
        labels = ["Under 40%", "40-50%", "50-60%", "60-70%", "70%+"]
        right = pd.DataFrame({"Picks that were right (%)": [round(b["hit"] * 100) for b in bands]},
                             index=labels[:len(bands)])
        st.markdown("**How often a pick is right, by how confident it was**")
        st.bar_chart(right)
        st.caption("The more confident the favourite, the more often it wins. That is why every pick shows a confidence tag.")
else:
    st.caption("Run  python src\\model_6_walkforward.py  to generate the accuracy results.")

calibration_path = "data/processed/calibration_home.png"
if os.path.exists(calibration_path):
    with st.expander("Technical: calibration plot"):
        st.image(calibration_path, caption="Predicted vs actual home-win probability (closer to the dashed line is better)")

st.divider()
refreshed = ""
try:
    with open("data/processed/fixtures_all.json", encoding="utf-8") as f:
        refreshed = f" Data last refreshed: {json.load(f)['fetched_at'].replace('T', ' ')}."
except Exception:
    pass
st.markdown(f"<div class='foot'>Data: football-data.co.uk (results and odds), the official Fantasy Premier League feed "
            "(squads and fixtures), TheSportsDB (club badges; badges are trademarks of their clubs and are shown for "
            f"illustration). A student learning project. Not betting advice.{refreshed}</div>", unsafe_allow_html=True)
