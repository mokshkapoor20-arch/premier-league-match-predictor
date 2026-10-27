"""Helpers for the matchweek browser in the app."""
import base64
import io
import json
import os
from datetime import datetime, timedelta
from functools import lru_cache

FIXTURES_PATH = "data/processed/fixtures_all.json"


def group_by_week(fixtures):
    """{matchweek: [fixture, ...]} sorted by kick-off, from a flat list of fixtures."""
    weeks = {}
    for g in fixtures or []:
        if g.get("event") is not None:
            weeks.setdefault(int(g["event"]), []).append(g)
    for w in weeks:
        weeks[w].sort(key=lambda g: g.get("kickoff") or "9999")
    return dict(sorted(weeks.items()))


def load_fixtures():
    """{matchweek: [fixture, ...]} from the saved file, or {} if the file isn't there yet."""
    if not os.path.exists(FIXTURES_PATH):
        return {}
    try:
        with open(FIXTURES_PATH, encoding="utf-8") as f:
            data = json.load(f)["fixtures"]
    except Exception:
        return {}
    return group_by_week(data)


def next_week(weeks):
    """The first matchweek that still has a match to play."""
    for w, games in weeks.items():
        if any(not g.get("finished") for g in games):
            return w
    return max(weeks) if weeks else None


def _last_sunday(year, month):
    d = datetime(year, month, 31)               # March and October both have 31 days
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d


def uk_time(utc):
    """UK clock time from a UTC time (British Summer Time: last Sunday of March to last Sunday of October)."""
    start = _last_sunday(utc.year, 3).replace(hour=1)
    end = _last_sunday(utc.year, 10).replace(hour=1)
    return utc + timedelta(hours=1) if start <= utc < end else utc


def kickoff_parts(iso):
    """{'day': 'Sat 10 Oct', 'uk': '12:30 PM', 'ist': '5:00 PM'} for a kick-off time."""
    tbc = {"day": "Date TBC", "uk": "TBC", "ist": "TBC"}
    if not iso:
        return tbc
    try:
        utc = datetime.fromisoformat(iso.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return tbc
    uk = uk_time(utc)
    ist = utc + timedelta(hours=5, minutes=30)

    def clock(t):
        return t.strftime("%I:%M %p").lstrip("0")

    return {"day": uk.strftime("%a %d %b"), "uk": clock(uk),
            "ist": clock(ist) + (" +1 day" if ist.date() > uk.date() else "")}


def kickoff_text(iso):
    p = kickoff_parts(iso)
    if p["day"] == "Date TBC":
        return "Date TBC"
    return f"{p['day']} · {p['uk']} UK ({p['ist']} IST)"


def initials(team):
    words = team.replace("'", "").split()
    return ("".join(w[0] for w in words[:3]) if len(words) > 1 else team[:3]).upper()


@lru_cache(maxsize=64)
def badge_uri(team, size=96):
    """A small badge image as a data URI for the fixture cards (None if no badge file)."""
    path = os.path.join("static", team.replace(" ", "_").replace("'", "") + ".png")
    if not os.path.exists(path):
        return None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None
