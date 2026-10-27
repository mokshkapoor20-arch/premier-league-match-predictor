import json

RAW_SQUADS = {
    "Chelsea": [
        ("Cole Palmer", "AM/RW", 2), ("Morgan Rogers", "AM/LW", 3), ("Geovany Quenda", "RW/RWB", 0),
        ("Estevão Willian", "RW/AM", 0), ("Pedro Neto", "RW/LW", 1), ("João Pedro", "ST/AM", 3),
        ("Jamie Gittens", "LW", 0), ("Liam Delap", "ST", 0), ("Nicolas Jackson", "ST", 0),
        ("Emmanuel Emegha", "ST", 0), ("Danny Welbeck", "ST", 0), ("Enzo Fernández", "CM", 0),
        ("Moisés Caicedo", "DM/CM", 0), ("Reece James", "RB/RWB/CM", 0), ("Roméo Lavia", "DM/CM", 1),
    ],
    "Arsenal": [
        ("Bukayo Saka", "RW", 3), ("Gabriel Martinelli", "LW", 0), ("Gabriel Jesus", "ST/RW", 0),
        ("Eberechi Eze", "AM/LW", 0), ("Viktor Gyökeres", "ST", 0), ("Christos Tzolis", "LW/RW", 0),
        ("Noni Madueke", "RW/LW", 0), ("Kai Havertz", "ST/AM", 2), ("Fabio Vieira", "AM/CM", 0),
        ("Reiss Nelson", "LW/RW", 0), ("Martin Ødegaard", "AM/CM", 2), ("Mikel Merino", "CM", 0),
        ("Declan Rice", "CM/DM", 0), ("Bruno Guimarães", "CM/DM", 1),
    ],
}

POSITION_WEIGHTS = {
    "ST": 1.0, "RW": 0.85, "LW": 0.85, "AM": 0.75, "CM": 0.45, "DM": 0.25,
    "RB": 0.15, "LB": 0.15, "RWB": 0.15, "LWB": 0.15, "CB": 0.1, "GK": 0.0,
}

def compute_position_weight(position_string):
    positions = position_string.replace(" ", "").split("/")
    weights = [POSITION_WEIGHTS.get(p, 0.1) for p in positions]
    return max(weights) if weights else 0.1

def compute_final_weight(position_weight, goals, max_goals_in_squad):
    # 40% position baseline (so a 0-goal striker doesn't vanish),
    # 60% actual scoring form this season, normalized against the
    # squad's current top scorer
    form_component = goals / max_goals_in_squad if max_goals_in_squad > 0 else 0
    return 0.4 * position_weight + 0.6 * form_component

squads = {}
for team, players in RAW_SQUADS.items():
    max_goals = max(g for _, _, g in players) if players else 1
    squads[team] = []
    for name, pos, goals in players:
        pos_weight = compute_position_weight(pos)
        final_weight = compute_final_weight(pos_weight, goals, max_goals)
        squads[team].append({
            "name": name, "position": pos, "goals": goals,
            "attack_weight": round(final_weight, 3),
        })

with open("data/processed/squads_manual.json", "w") as f:
    json.dump(squads, f, indent=2)

print("Saved squads_manual.json")
for team, players in squads.items():
    print(f"\n{team}:")
    for p in sorted(players, key=lambda x: -x["attack_weight"]):
        print(f"  {p['name']:<22} {p['position']:<12} goals={p['goals']}  weight={p['attack_weight']}")