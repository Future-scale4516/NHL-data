"""
odds.py
The Odds API client for NHL: fetch, match to NHL-API games, and pick best prices.
Kept free of Streamlit so it can move into shared/ later. Caching is done in app.py.
"""

import unicodedata
from collections import Counter
from datetime import datetime

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "icehockey_nhl"
REGIONS = "uk"                    # add ",us" for more books (doubles credit cost)
MARKETS = "h2h,spreads,totals"    # 3 markets x 1 region = 3 credits per fetch

ABBREV_TO_NAME = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken", "SJS": "San Jose Sharks",
    "STL": "St Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}


def _norm(name: str) -> str:
    """Lowercase, strip accents and dots so 'Montréal' / 'St. Louis' match reliably."""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return n.lower().replace(".", "").strip()


def _parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fetch_odds(api_key: str) -> list[dict]:
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
        params={"apiKey": api_key, "regions": REGIONS, "markets": MARKETS, "oddsFormat": "decimal"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def match_event(events: list[dict], home_abbrev: str, away_abbrev: str, start_utc: str):
    """Find the odds event for a game: same two teams, closest start time (handles rematches)."""
    home_n = _norm(ABBREV_TO_NAME.get(home_abbrev, ""))
    away_n = _norm(ABBREV_TO_NAME.get(away_abbrev, ""))
    cands = [e for e in events if _norm(e["home_team"]) == home_n and _norm(e["away_team"]) == away_n]
    if not cands:
        return None
    if not start_utc:
        return cands[0]
    t0 = _parse_time(start_utc)
    return min(cands, key=lambda e: abs((_parse_time(e["commence_time"]) - t0).total_seconds()))


def _best(current, price, book):
    return (price, book) if current is None or price > current[0] else current


def best_prices(event: dict) -> dict:
    """
    Best available decimal price per outcome across bookmakers.
    Returns {"h2h": {"home","away"}, "spreads": {"home_-1.5", "away_+1.5", "away_-1.5", "home_+1.5"},
             "totals": {"line", "over", "under"} or None}; each price is (price, bookmaker) or None.
    """
    home, away = event["home_team"], event["away_team"]
    h2h = {"home": None, "away": None}
    spreads = {"home_-1.5": None, "away_+1.5": None, "away_-1.5": None, "home_+1.5": None}
    total_rows = []  # (point, side, price, book)

    for bk in event.get("bookmakers", []):
        book = bk["title"]
        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                # The model's ML includes OT/SO. Skip 3-way (home/draw/away) regulation markets,
                # and any pair implying under 97% (a real 2-way market always carries a margin).
                if any(o["name"].lower() == "draw" for o in mkt["outcomes"]):
                    continue
                if len(mkt["outcomes"]) == 2 and sum(1 / o["price"] for o in mkt["outcomes"]) < 0.97:
                    continue
            for o in mkt["outcomes"]:
                if mkt["key"] == "h2h":
                    side = "home" if o["name"] == home else "away" if o["name"] == away else None
                    if side:
                        h2h[side] = _best(h2h[side], o["price"], book)
                elif mkt["key"] == "spreads":
                    side = "home" if o["name"] == home else "away" if o["name"] == away else None
                    pt = o.get("point")
                    if side and pt in (-1.5, 1.5):
                        key = f"{side}_{pt:+g}"
                        if key in spreads:
                            spreads[key] = _best(spreads[key], o["price"], book)
                elif mkt["key"] == "totals":
                    total_rows.append((o["point"], o["name"].lower(), o["price"], book))

    totals = None
    if total_rows:
        # Main line = the point most books quote on the Over side
        line = Counter(pt for pt, side, _, _ in total_rows if side == "over").most_common(1)
        if line:
            line = line[0][0]
            over = under = None
            for pt, side, price, book in total_rows:
                if pt == line and side == "over":
                    over = _best(over, price, book)
                elif pt == line and side == "under":
                    under = _best(under, price, book)
            totals = {"line": line, "over": over, "under": under}

    return {"h2h": h2h, "spreads": spreads, "totals": totals}
