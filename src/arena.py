"""
arena.py
--------
Builds the animated "match arena" (badges slide in, mini dot-match, winner
bashes the loser, stats pop in) and shows it inside the Streamlit app.

The animation itself lives in arena_template.html. This file's job is to
fill that template with real data: badge images, club colours, and the
numbers from predict_match().
"""
import base64
import io
import json
import os
from collections import Counter

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "static")
TEMPLATE = os.path.join(HERE, "arena_template.html")

FALLBACK_RGB = (68, 80, 102)


def _badge_file(team):
    # Same naming rule as the app: spaces -> "_", apostrophes removed
    name = team.replace(" ", "_").replace("'", "") + ".png"
    path = os.path.join(STATIC, name)
    return path if os.path.exists(path) else None


def _badge_uri(team):
    """The badge as a base64 'data URI', so it can sit inside the HTML."""
    path = _badge_file(team)
    if not path:
        return None
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((360, 360))                 # keep the page light
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _dominant_rgb(team):
    """Most common 'colourful' colour in the badge = the club colour."""
    path = _badge_file(team)
    if not path:
        return FALLBACK_RGB
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((80, 80))
        data = img.tobytes()
        counts = Counter()
        for i in range(0, len(data), 4):
            r, g, b, a = data[i], data[i + 1], data[i + 2], data[i + 3]
            if a < 140:
                continue                           # transparent
            if r > 225 and g > 225 and b > 225:
                continue                           # white
            saturation = max(r, g, b) - min(r, g, b)
            bucket = (r // 32, g // 32, b // 32)
            counts[bucket] += 3 if saturation > 50 else 1   # prefer colourful pixels
        if not counts:
            return FALLBACK_RGB
        qr, qg, qb = counts.most_common(1)[0][0]
        return (qr * 32 + 16, qg * 32 + 16, qb * 32 + 16)
    except Exception:
        return FALLBACK_RGB


def _readable(rgb, min_lum=0.36):
    """Lighten very dark colours so they show up on the dark stage."""
    r, g, b = rgb
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    if lum >= min_lum:
        return rgb
    t = (min_lum - lum) / (1 - lum)
    return tuple(int(c + (255 - c) * t) for c in rgb)


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def _initials(team):
    words = team.replace("'", "").split()
    return ("".join(w[0] for w in words[:3]) if len(words) > 1 else team[:3]).upper()


def _team_payload(team, rgb):
    return {"name": team, "short": _initials(team), "badge": _badge_uri(team), "c1": _hex(rgb)}


def _players(rows):
    """[{name, position, prob_score}] -> [[name, position, percent]] (top 6)."""
    return [[p["name"], p["position"], int(round(float(p["prob_score"]) * 100))]
            for p in (rows or [])[:6]]


def build_arena_html(home_team, away_team, result=None):
    """Return (html, height). With result=None you get the badges-only preview."""
    home_rgb = _readable(_dominant_rgb(home_team))
    away_rgb = _readable(_dominant_rgb(away_team))

    # Two similar colours (e.g. two red clubs) would be hard to tell apart,
    # so give the away side a pale colour instead.
    if sum((a - b) ** 2 for a, b in zip(home_rgb, away_rgb)) ** 0.5 < 90:
        away_rgb = (230, 233, 242)

    data = {
        "mode": "predict" if result else "preview",
        "home": _team_payload(home_team, home_rgb),
        "away": _team_payload(away_team, away_rgb),
    }

    height = 560            # preview: just the stage (the page shrinks this to fit)
    if result:
        data.update({
            "p": {"H": float(result["p_home"]), "D": float(result["p_draw"]), "A": float(result["p_away"])},
            "xg": [round(float(result["exp_home_goals"]), 2), round(float(result["exp_away_goals"]), 2)],
            "scores": [[int(h), int(a), round(float(p) * 100, 1)] for h, a, p in result["top_scorelines"]],
            "players": {"H": _players(result.get("home_players")), "A": _players(result.get("away_players"))},
            "factors": list(result.get("key_factors", [])),
            "meta": {"source": result.get("source", "model"), "confidence": result.get("confidence")},
        })
        has_players = bool(data["players"]["H"] or data["players"]["A"])
        height = 1450 if has_players else 1000      # safe maximum; the page shrinks it to fit

    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read().replace("__ARENA_DATA__", payload)
    return html, height


def render_arena(home_team, away_team, result=None):
    import streamlit as st
    html, height = build_arena_html(home_team, away_team, result)
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)            # newer Streamlit
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)   # older Streamlit
