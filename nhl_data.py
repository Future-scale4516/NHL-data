"""
nhl_data.py
Pulls team-level and goalie-level data from the free NHL API (api-web.nhle.com).
No API key needed. This mirrors the role pybaseball/MLB Stats API played in the MLB app.

NOTE: NHL's public API field names shift occasionally. The first time you run this
against live data, call debug_sample_standings() / debug_sample_goalie() and eyeball
the JSON — adjust the field lookups below if anything's renamed.
"""

import requests
from datetime import date

NHL_API_BASE = "https://api-web.nhle.com/v1"


def get_standings(as_of_date: str = None) -> list[dict]:
    """
    Returns one dict per team with season-to-date goals for/against and games played.
    as_of_date: 'YYYY-MM-DD', defaults to today.
    """
    d = as_of_date or date.today().isoformat()
    url = f"{NHL_API_BASE}/standings/{d}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    teams = []
    for row in data.get("standings", []):
        teams.append({
            "team_abbrev": row["teamAbbrev"]["default"],
            "team_name": row["teamName"]["default"],
            "games_played": row["gamesPlayed"],
            "goals_for": row["goalFor"],
            "goals_against": row["goalAgainst"],
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "ot_losses": row.get("otLosses"),
        })
    return teams


def compute_team_strengths(teams: list[dict]) -> dict:
    """
    Converts raw standings into attack/defense strength multipliers vs league average,
    same idea as the MLB app's batter/pitcher rate-stat normalization.
    """
    total_goals = sum(t["goals_for"] for t in teams)
    total_games = sum(t["games_played"] for t in teams)
    league_avg_goals_per_team_per_game = total_goals / total_games  # goals scored per team per game

    strengths = {}
    for t in teams:
        gp = t["games_played"]
        if gp == 0:
            continue
        gf_per_game = t["goals_for"] / gp
        ga_per_game = t["goals_against"] / gp
        strengths[t["team_abbrev"]] = {
            "attack": gf_per_game / league_avg_goals_per_team_per_game,
            "defense": ga_per_game / league_avg_goals_per_team_per_game,
            "games_played": gp,
        }
    strengths["_league_avg"] = league_avg_goals_per_team_per_game
    return strengths


def get_starting_goalie_stats(team_abbrev: str, season: str, goalie_name: str = None) -> dict:
    """
    Returns a goalie's season save percentage and games played, plus the team's
    overall goalie save pct for comparison (used as the baseline in the goalie adjustment).

    season format: '20262027'
    If goalie_name is None, returns the goalie with the most games played this season
    (a reasonable stand-in until you wire up confirmed-starter data).
    """
    url = f"{NHL_API_BASE}/club-stats/{team_abbrev}/{season}/2"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    goalies = data.get("goalies", [])
    if not goalies:
        return {}

    if goalie_name:
        matches = [g for g in goalies if g["name"]["default"].lower() == goalie_name.lower()]
        goalie = matches[0] if matches else max(goalies, key=lambda g: g.get("gamesPlayed", 0))
    else:
        goalie = max(goalies, key=lambda g: g.get("gamesPlayed", 0))

    team_shots_against = sum(g.get("shotsAgainst", 0) for g in goalies)
    team_saves = sum(g.get("saves", 0) for g in goalies)
    team_sv_pct = team_saves / team_shots_against if team_shots_against else None

    return {
        "name": goalie["name"]["default"],
        "games_played": goalie.get("gamesPlayed"),
        "save_pct": goalie.get("savePctg"),
        "team_save_pct": team_sv_pct,
    }


def debug_sample_standings():
    """Run once to confirm field names haven't changed before trusting compute_team_strengths."""
    teams = get_standings()
    print(teams[0] if teams else "No data returned")


def debug_sample_goalie(team_abbrev: str, season: str):
    url = f"{NHL_API_BASE}/club-stats/{team_abbrev}/{season}/2"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    print(resp.json().get("goalies", [])[:1])
