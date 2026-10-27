"""
live.py
-------
Real-time fixtures and live scores from the official Fantasy Premier League feed.

get_fixtures()  -> (fixtures, source)   fresh data, cached for a short time, falls back to the saved file
live_info(g)    -> {state, label, estimated}   "live" / "ft" / "upcoming" plus a text like "33'" or "HT"
next_kickoff()  -> the soonest match that has not started

The feed can lag real life by a minute or two. If it has not yet marked a match as started, or does not
give the minute, the minute is estimated from the kick-off time and shown with a "~".
"""
import json
import os
import time
from datetime import datetime, timezone

import requests

from club_names import TEAM_FIX

BASE = "https://fantasy.premierleague.com/api/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PLMatchPredictorSchoolProject/1.0; personal learning project)"}
FIXTURES_PATH = "data/processed/fixtures_all.json"

_cache = {"t": 0.0, "fixtures": None}


def _get(path):
    r = requests.get(BASE + path, headers=HEADERS, timeout=12)
    r.raise_for_status()
    return r.json()


def fetch_feed():
    """Download teams and fixtures and return a clean list of fixture dicts. Raises on any failure."""
    teams = _get("bootstrap-static/")["teams"]
    raw = _get("fixtures/")
    names = {t["id"]: TEAM_FIX.get(t["name"], t["name"]) for t in teams}
    out = []
    for f in raw:
        home, away = names.get(f.get("team_h")), names.get(f.get("team_a"))
        if not home or not away:
            continue
        out.append({
            "event": f.get("event"),
            "kickoff": f.get("kickoff_time"),
            "home": home,
            "away": away,
            "started": bool(f.get("started")),
            "finished": bool(f.get("finished") or f.get("finished_provisional")),
            "minutes": f.get("minutes") or 0,
            "hs": f.get("team_h_score"),
            "as": f.get("team_a_score"),
        })
    if not out:
        raise ValueError("the feed returned no fixtures")
    return out


def _write_file(fixtures):
    """Save to disk without ever leaving a half-written file (write a temp file, then swap it in)."""
    os.makedirs(os.path.dirname(FIXTURES_PATH), exist_ok=True)
    tmp = FIXTURES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"fetched_at": datetime.now().isoformat(timespec="seconds"), "fixtures": fixtures},
                  fh, ensure_ascii=False, indent=1)
    os.replace(tmp, FIXTURES_PATH)


def _read_file():
    try:
        with open(FIXTURES_PATH, encoding="utf-8") as fh:
            return json.load(fh)["fixtures"]
    except Exception:
        return None


def get_fixtures(persist=True, ttl=None):
    """
    Returns (fixtures, source). source is:
      'live'   fetched from the feed within the last `ttl` seconds (25s during a match, 90s otherwise)
      'saved'  the feed could not be reached, so this is the last saved file
      'none'   nothing available (fixtures is [])
    """
    now = time.time()
    if ttl is None:      # check often while a match is on, rarely otherwise
        ttl = 25 if _cache["fixtures"] and live_games(_cache["fixtures"]) else 90
    if _cache["fixtures"] is not None and now - _cache["t"] < ttl:
        return _cache["fixtures"], "live"
    try:
        fixtures = fetch_feed()
    except Exception:
        saved = _cache["fixtures"] or _read_file()
        return (saved, "saved") if saved else ([], "none")
    _cache["t"], _cache["fixtures"] = now, fixtures
    if persist:
        try:
            _write_file(fixtures)           # so the prediction record can mark finished matches
        except OSError:
            pass
    return fixtures, "live"


def kickoff_dt(g):
    """Kick-off as an aware UTC datetime, or None."""
    k = g.get("kickoff")
    if not k:
        return None
    try:
        return datetime.fromisoformat(k.replace("Z", "+00:00"))
    except ValueError:
        return None


def _estimate_minute(elapsed_min):
    """Match clock from minutes since kick-off, allowing a 15 minute half-time. Returns (text, minute)."""
    if elapsed_min < 0:
        return "", 0
    if elapsed_min <= 45:
        return f"{max(1, int(elapsed_min) + 1)}'", int(elapsed_min) + 1
    if elapsed_min <= 47:
        return "45+'", 45
    if elapsed_min <= 62:
        return "HT", 45
    m = 45 + int(elapsed_min - 62) + 1
    if m <= 90:
        return f"{m}'", m
    return "90+'", 90


def live_info(g, now=None):
    """
    {'state': 'live'|'ft'|'upcoming', 'label': text, 'estimated': bool}
    A match counts as live when the feed says it started and has not finished, or when the feed has not
    caught up yet but kick-off was less than 2 hours 15 minutes ago.
    """
    now = now or datetime.now(timezone.utc)
    if g.get("finished"):
        return {"state": "ft", "label": "Full time", "estimated": False}
    ko = kickoff_dt(g)
    elapsed = (now - ko).total_seconds() / 60 if ko else None

    if g.get("started"):
        feed_min = g.get("minutes") or 0
        if feed_min > 0:
            if feed_min >= 90:
                label = f"{int(feed_min)}'" if feed_min > 90 else "90'"
            else:
                label = f"{int(feed_min)}'"
            # the feed's minute freezes between updates, and does not know about half-time
            if elapsed is not None and 47 < elapsed <= 62 and 45 <= feed_min <= 46:
                label = "HT"
            return {"state": "live", "label": label, "estimated": False}
        if elapsed is not None:
            text, _ = _estimate_minute(elapsed)
            return {"state": "live", "label": "~" + text if text else "Live", "estimated": True}
        return {"state": "live", "label": "Live", "estimated": True}

    if elapsed is not None and 0 <= elapsed < 135:
        text, _ = _estimate_minute(elapsed)
        return {"state": "live", "label": "~" + text, "estimated": True}
    return {"state": "upcoming", "label": "", "estimated": False}


def score_text(g):
    """'2 – 1'; a match that is live but has no score yet from the feed shows 0 – 0."""
    hs = g.get("hs") if g.get("hs") is not None else 0
    aws = g.get("as") if g.get("as") is not None else 0
    return f"{hs} – {aws}"


def live_games(fixtures, now=None):
    return [g for g in fixtures if live_info(g, now)["state"] == "live"]


def next_kickoff(fixtures, now=None):
    """The soonest match that has not started, or None."""
    now = now or datetime.now(timezone.utc)
    future = [(kickoff_dt(g), g) for g in fixtures
              if not g.get("finished") and kickoff_dt(g) and kickoff_dt(g) > now]
    return min(future, key=lambda x: x[0])[1] if future else None
