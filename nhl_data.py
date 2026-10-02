"""
nhl_data.py
Pulls team-level and goalie-level data from the free NHL API (api-web.nhle.com).
No API key needed. This mirrors the role pybaseball/MLB Stats API played in the MLB app.

NOTE: NHL's public API field names shift occasionally. The first time you run this
against live data, call debug_sample_standings() / debug_sample_goalie() and eyeball
the JSON — adjust the field lookups below if anything's renamed.
"""

import time
import requests
from functools import lru_cache
from datetime import date, timedelta

NHL_API_BASE = "https://api-web.nhle.com/v1"


def _get_json(url: str, timeout: int = 15, tries: int = 4):
    """GET with exponential backoff -- long backtests make hundreds of calls and hit rate limits."""
    last = None
    for i in range(tries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests (rate limited)")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last = e
            time.sleep(0.5 * 2 ** i)
    raise last


def get_standings(as_of_date: str = None, tries: int = 4) -> list[dict]:
    """
    Returns one dict per team with season-to-date goals for/against and games played.
    as_of_date: 'YYYY-MM-DD', defaults to today.
    """
    d = as_of_date or date.today().isoformat()
    data = _get_json(f"{NHL_API_BASE}/standings/{d}", tries=tries)

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


PRIOR_WEIGHT_GAMES = 70           # last season's rates count as this many games of evidence. Fitted on the full
                                  # 2025-26 season (1,192 games): at 20 the team spread was overconfident (slope 0.70,
                                  # 95% CI 0.44-0.95); 70 brings it to ~0.93. Sweep rows were within noise of each other.
PRIOR_REGRESSION = 0.5            # pull last season's rates 50% back toward average (beat 0.25 at every K in the sweep)
STRENGTH_SHRINK = 1.0             # 1.0 = off. An extra flat shrink of attack/defense deviations. A 0.5 fit on
                                  # Oct-Dec looked right but FAILED out-of-sample (late season it was already
                                  # well calibrated), so prior weight (PRIOR_WEIGHT_GAMES) does this job instead.


@lru_cache(maxsize=8)
def prior_season_end_date(season_start_year: int) -> str:
    """
    Date of the final regular-season standings for the season BEFORE the one starting in
    `season_start_year`. Read from the NHL API's own season list, because the standings
    endpoint only answers sensibly for dates inside a season (a guessed off-season date
    silently returns the wrong table).
    """
    try:
        resp = requests.get(f"{NHL_API_BASE}/standings-season", timeout=10)
        resp.raise_for_status()
        for s in resp.json().get("seasons", []):
            if str(s.get("id", ""))[:4] == str(season_start_year - 1) and s.get("standingsEnd"):
                return s["standingsEnd"]
    except Exception:
        pass
    return f"{season_start_year}-04-14"   # fallback: safely inside the regular season


def prior_is_valid(teams: list[dict]) -> bool:
    """A real end-of-season table: every team at ~82 games. Anything else is the wrong table."""
    return bool(teams) and sum(t["games_played"] for t in teams) / len(teams) >= 70


def get_prior_standings(today: date = None) -> list[dict]:
    """Last season's final standings, used as the early-season prior. [] if unavailable/invalid."""
    d = today or date.today()
    y = d.year if d.month >= 9 else d.year - 1      # start year of the current season
    try:
        teams = get_standings(prior_season_end_date(y))
    except Exception:
        return []
    return teams if prior_is_valid(teams) else []


def compute_team_strengths(teams: list[dict], prior_teams: list[dict] = None,
                           k: float = None, regression: float = None, shrink: float = None) -> dict:
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

    # Each season's rates are divided by THAT season's league average, so attack x defense
    # averages to 1. (Mixing seasons inflates both factors whenever league scoring shifts.)
    cur_norm = cur_league if (cur_league and total_gp >= 200) else league_avg
    prior_norm = prior_league or league_avg

    k = PRIOR_WEIGHT_GAMES if k is None else k                      # overrides let the Backtest page sweep these
    regression = PRIOR_REGRESSION if regression is None else regression
    shrink = STRENGTH_SHRINK if shrink is None else shrink

    prior = {}
    for t in (prior_teams or []):
        gp = t["games_played"]
        if gp:
            a = (t["goals_for"] / gp) / prior_norm
            d = (t["goals_against"] / gp) / prior_norm
            prior[t["team_abbrev"]] = (
                1 + (a - 1) * (1 - regression),
                1 + (d - 1) * (1 - regression),
            )

    strengths = {}
    for t in teams:
        gp = t["games_played"]
        p_att, p_def = prior.get(t["team_abbrev"], (1.0, 1.0))
        cur_att = (t["goals_for"] / gp) / cur_norm if gp else p_att
        cur_def = (t["goals_against"] / gp) / cur_norm if gp else p_def
        strengths[t["team_abbrev"]] = {
            "attack": 1 + shrink * ((gp * cur_att + k * p_att) / (gp + k) - 1),
            "defense": 1 + shrink * ((gp * cur_def + k * p_def) / (gp + k) - 1),
            "games_played": gp,
        }
    strengths["_league_avg"] = league_avg
    return strengths


def get_games(day: str) -> list[dict]:
    """Games scheduled on a given date (YYYY-MM-DD), as listed by the NHL schedule endpoint."""
    payload = _get_json(f"{NHL_API_BASE}/schedule/{day}")
    games = []
    for wk in payload.get("gameWeek", []):
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


def get_completed_games_range(start: date, end: date) -> list[dict]:
    """
    Final REGULAR-SEASON games between start and end (inclusive) with scores.
    Uses the schedule endpoint (returns a week at a time), so a season is ~30 calls.
    last_period is REG / OT / SO -- needed because the shootout winner's final score
    includes one extra goal that totals and goal-rate comparisons must exclude.
    """
    games, seen, d = [], set(), start
    while d <= end:
        payload = _get_json(f"{NHL_API_BASE}/schedule/{d.isoformat()}")
        max_day = d
        for wk in payload.get("gameWeek", []):
            try:
                wd = date.fromisoformat(wk.get("date"))
            except (TypeError, ValueError):
                continue
            max_day = max(max_day, wd)
            if wd < start or wd > end:
                continue
            for g in wk.get("games", []):
                if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
                    continue
                h, a = g.get("homeTeam", {}), g.get("awayTeam", {})
                if h.get("score") is None or a.get("score") is None or g["id"] in seen:
                    continue
                seen.add(g["id"])
                games.append({
                    "id": g["id"], "date": wd.isoformat(), "start_utc": g.get("startTimeUTC"),
                    "home": h["abbrev"], "away": a["abbrev"],
                    "home_score": h["score"], "away_score": a["score"],
                    "last_period": (g.get("gameOutcome") or {}).get("lastPeriodType", "REG"),
                })
        d = max(max_day + timedelta(days=1), d + timedelta(days=1))
    return games


def debug_sample_completed(day: str):
    """Run once: confirm score / gameOutcome fields match what get_completed_games_range expects."""
    d = date.fromisoformat(day)
    print(get_completed_games_range(d, d)[:2])


def get_boxscore(game_id: int, tries: int = 4) -> dict:
    """Full boxscore for one game (player stats per team, incl. goalies)."""
    return _get_json(f"{NHL_API_BASE}/gamecenter/{game_id}/boxscore", tries=tries)


def debug_sample_boxscore(game_id: int):
    """Run once: confirm the goalie fields (starter, toi, shotsAgainst, saves) match nhl_goalies.parse_goalies."""
    box = get_boxscore(game_id)
    print("top-level keys:", list(box.keys()))
    pbs = box.get("playerByGameStats", {})
    print("playerByGameStats keys:", list(pbs.keys()))
    print("goalies sample:", (pbs.get("homeTeam", {}).get("goalies") or [])[:2])
