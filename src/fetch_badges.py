import os
import time
import requests

TEAMS = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
         'Cardiff', 'Chelsea', 'Coventry', 'Crystal Palace', 'Everton', 'Fulham',
         'Huddersfield', 'Hull', 'Ipswich', 'Leeds', 'Leicester', 'Liverpool', 'Luton',
         'Man City', 'Man United', 'Middlesbrough', 'Newcastle', 'Norwich',
         "Nott'm Forest", 'QPR', 'Sheffield United', 'Southampton', 'Stoke',
         'Sunderland', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham',
         'Wolves']
ONLY = ["Nott'm Forest"]   # set to [] to refetch everyone
for team in TEAMS:
    if ONLY and team not in ONLY:
        continue
    picked = None

# Fuller names first; if those find nothing we fall back to the short name
FULL_NAMES = {
    "Man City": "Manchester City", "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest", "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion", "Leeds": "Leeds United",
    "Newcastle": "Newcastle United", "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers", "West Ham": "West Ham United",
    "Brighton": "Brighton and Hove Albion", "Leicester": "Leicester City",
    "Norwich": "Norwich City", "Hull": "Hull City", "Stoke": "Stoke City",
    "Swansea": "Swansea City", "Cardiff": "Cardiff City", "Coventry": "Coventry City",
    "Ipswich": "Ipswich Town", "Luton": "Luton Town", "Huddersfield": "Huddersfield Town",
}

BAD_WORDS = ("women", "ladies", "u21", "u23", "u18", "u19", "reserve", "academy")

def pick_team(results):
    # Only football clubs, no women's/youth sides, prefer England/Wales
    soccer = [t for t in results
              if t.get("strSport") == "Soccer"
              and not any(w in (t.get("strTeam") or "").lower() for w in BAD_WORDS)]
    preferred = [t for t in soccer if t.get("strCountry") in ("England", "Wales")]
    pool = preferred or soccer
    return pool[0] if pool else None

os.makedirs("static", exist_ok=True)
problems = []

for team in TEAMS:
    picked = None
    for name in [FULL_NAMES.get(team, team), team]:
        try:
            data = requests.get(
                f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={name}",
                timeout=10).json()
        except Exception:
            data = {}
        time.sleep(1)
        picked = pick_team(data.get("teams") or [])
        if picked and picked.get("strBadge"):
            break
        picked = None

    if not picked:
        print(f"CHECK  {team:<18} no football club found, kept the old badge")
        problems.append(team)
        continue

    try:
        img = requests.get(picked["strBadge"], timeout=10).content
        path = "static/" + team.replace(" ", "_").replace("'", "") + ".png"
        with open(path, "wb") as f:
            f.write(img)
        print(f"OK     {team:<18} -> {picked['strTeam']}  [{picked.get('strLeague')}]")
    except Exception as e:
        print(f"CHECK  {team:<18} download failed: {e}")
        problems.append(team)

print("\nNeed a manual look:", problems if problems else "none")
