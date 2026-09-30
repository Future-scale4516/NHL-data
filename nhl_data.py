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


PRIOR_SEASON_END = "2026-04-16"   # last regular-season day; confirm with debug_sample_standings
PRIOR_WEIGHT_GAMES = 20           # last season's rates count as this many games of evidence
PRIOR_REGRESSION = 0.25           # pull last season's rates 25% back toward league average


def get_prior_standings() -> list[dict]:
    """Last season's final standings, used as the early-season prior. Empty list on failure."""
    try:
        return get_standings(PRIOR_SEASON_END)
    except Exception:
        return []


def compute_team_strengths(teams: list[dict], prior_teams: list[dict] = None) -> dict:
    """
    Attack/defense multipliers vs league average, blended with last season's rates:
    blended = (gp * current + K * prior) / (gp + K)
    With 0-1 games played the prior dominates; by ~40 games current form carries most weight.
    """
    def per_game(rows):
        gp = sum(r["games_played"] for r in rows)
        return (sum(r["goals_for"] for r in rows) / gp) if gp else None

    cur_league = per_game(teams)
    prior_league = per_game(prior_teams) if prior_teams else None
    league_avg = prior_league or cur_league or 3.0   # NHL ~3.0 goals per team per game
    total_gp = sum(t["games_played"] for t in teams)
    if cur_league and total_gp >= 200:               # enough current data, use it
        league_avg = cur_league

    prior = {}
    for t in (prior_teams or []):
        gp = t["games_played"]
        if gp:
            a = (t["goals_for"] / gp) / league_avg
            d = (t["goals_against"] / gp) / league_avg
            prior[t["team_abbrev"]] = (
                1 + (a - 1) * (1 - PRIOR_REGRESSION),
                1 + (d - 1) * (1 - PRIOR_REGRESSION),
            )

    k = PRIOR_WEIGHT_GAMES
    strengths = {}
    for t in teams:
        gp = t["games_played"]
        p_att, p_def = prior.get(t["team_abbrev"], (1.0, 1.0))
        cur_att = (t["goals_for"] / gp) / league_avg if gp else p_att
        cur_def = (t["goals_against"] / gp) / league_avg if gp else p_def
        strengths[t["team_abbrev"]] = {
            "attack": (gp * cur_att + k * p_att) / (gp + k),
            "defense": (gp * cur_def + k * p_def) / (gp + k),
            "games_played": gp,
        }
    strengths["_league_avg"] = league_avg
    return strengths


def get_games(day: str) -> list[dict]:
    """Games scheduled on a given date (YYYY-MM-DD), as listed by the NHL schedule endpoint."""
    resp = requests.get(f"{NHL_API_BASE}/schedule/{day}", timeout=10)
    resp.raise_for_status()
    games = []
    for wk in resp.json().get("gameWeek", []):
        if wk.get("date") != day:
            continue
        for g in wk.get("games", []):
            games.append({
                "id": g["id"],
                "start_utc": g.get("startTimeUTC"),
                "home": g["homeTeam"]["abbrev"],
                "away": g["awayTeam"]["abbrev"],
                "state": g.get("gameState"),
            })
    return games


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
