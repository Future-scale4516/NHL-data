"""
odds.py
The Odds API client for NHL: fetch, match to NHL-API games, and pick best prices.
Kept free of Streamlit so it can move into shared/ later. Caching is done in app.py.
"""

import unicodedata
from collections import Counter
from datetime import datetime

import pandas as pd
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


LAST_QUOTA = {}   # filled from the response headers of the most recent live odds call


def fetch_odds(api_key: str) -> list[dict]:
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
        params={"apiKey": api_key, "regions": REGIONS, "markets": MARKETS, "oddsFormat": "decimal"},
        timeout=15,
    )
    resp.raise_for_status()
    LAST_QUOTA.update({k: resp.headers.get(h) for k, h in (("used", "x-requests-used"),
                                                           ("remaining", "x-requests-remaining"),
                                                           ("last", "x-requests-last"))})
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


COST_LOG = []   # one entry per real (uncached) historical call; lets the UI report credits spent


def credits_per_snapshot(markets: str = MARKETS) -> int:
    """Historical odds cost 10 credits per market per region."""
    return 10 * len([m for m in markets.split(",") if m]) * len([r for r in REGIONS.split(",") if r])


def fetch_historical_odds(api_key: str, iso_ts: str, markets: str = MARKETS):
    """
    Odds snapshot as of iso_ts (YYYY-MM-DDTHH:MM:SSZ). Historical calls bill 10 credits per region per market.
    Retries rate limits / server errors with backoff. A 401 (out of credits or bad key) is raised at once with the
    API's own message. Returns (events, meta).
    """
    import time
    last = None
    for i in range(4):
        resp = requests.get(
            f"{ODDS_API_BASE}/historical/sports/{SPORT_KEY}/odds",
            params={"apiKey": api_key, "regions": REGIONS, "markets": markets, "oddsFormat": "decimal", "date": iso_ts},
            timeout=20,
        )
        if resp.status_code == 401:
            raise RuntimeError(f"401 from the Odds API (out of credits or invalid key): {resp.text[:200]}")
        if resp.status_code in (429, 500, 502, 503, 504):
            last = f"{resp.status_code} {resp.text[:120]}"
            time.sleep(1.5 * 2 ** i)
            continue
        resp.raise_for_status()
        meta = {"last": int(resp.headers.get("x-requests-last", 0) or 0),
                "remaining": resp.headers.get("x-requests-remaining")}
        return resp.json().get("data", []), meta
    raise RuntimeError(f"Odds API kept failing after retries: {last}")


def fair_probs(event: dict, line: float = None) -> dict:
    """
    Consensus no-vig probability per selection, averaged over bookmakers that quote the FULL two-way
    market. Returns {key: (probability, number_of_books)} with the same keys selection_rows uses:
    h2h:home/away, spreads:home_-1.5 etc., totals:over/under (only at `line`).
    Spreads are only paired as (home -1.5 with away +1.5) or (away -1.5 with home +1.5).
    """
    home, away = event["home_team"], event["away_team"]
    acc = {}

    def add(k_a, k_b, price_a, price_b):
        inv_a, inv_b = 1 / price_a, 1 / price_b
        tot = inv_a + inv_b
        if tot < 0.97:                      # a real two-way market always carries a margin
            return
        acc.setdefault(k_a, []).append(inv_a / tot)
        acc.setdefault(k_b, []).append(inv_b / tot)

    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            outs = mkt["outcomes"]
            if mkt["key"] == "h2h":
                if len(outs) != 2 or any(o["name"].lower() == "draw" for o in outs):
                    continue                # 3-way regulation market: not comparable
                px = {o["name"]: o["price"] for o in outs}
                if home in px and away in px:
                    add("h2h:home", "h2h:away", px[home], px[away])
            elif mkt["key"] == "spreads":
                px = {}
                for o in outs:
                    side = "home" if o["name"] == home else "away" if o["name"] == away else None
                    if side and o.get("point") in (-1.5, 1.5):
                        px[(side, o["point"])] = o["price"]
                for hs, hp, as_, ap in (("home", -1.5, "away", 1.5), ("away", -1.5, "home", 1.5)):
                    if (hs, hp) in px and (as_, ap) in px:
                        add(f"spreads:{hs}_{hp:+g}", f"spreads:{as_}_{ap:+g}", px[(hs, hp)], px[(as_, ap)])
            elif mkt["key"] == "totals" and line is not None:
                over = [o["price"] for o in outs if o["name"].lower() == "over" and o.get("point") == line]
                under = [o["price"] for o in outs if o["name"].lower() == "under" and o.get("point") == line]
                if over and under:
                    add("totals:over", "totals:under", over[0], under[0])
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}


PROP_MARKET_SOG = "player_shots_on_goal"


def fetch_event_props(api_key: str, event_id: str, market: str = PROP_MARKET_SOG, regions: str = "us"):
    """
    Player props for ONE event (props are only served per event). The Odds API carries NHL player props from US
    bookmakers only, so the default region is 'us' (the UK region returns nothing for this market). Costs 1 credit per
    market per region. Returns (event json, meta). An event with no prop prices comes back with empty bookmakers.
    """
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds",
        params={"apiKey": api_key, "regions": regions, "markets": market, "oddsFormat": "decimal"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json(), {"last": int(resp.headers.get("x-requests-last", 0) or 0),
                         "remaining": resp.headers.get("x-requests-remaining")}


def props_table(event: dict, market: str = PROP_MARKET_SOG) -> pd.DataFrame:
    """Flat rows: player, line, side, price, book."""
    rows = []
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != market:
                continue
            for o in mkt["outcomes"]:
                if o.get("description") and o.get("point") is not None:
                    rows.append({"player": o["description"], "line": float(o["point"]), "side": o["name"].lower(),
                                 "price": o["price"], "book": bk["title"]})
    return pd.DataFrame(rows, columns=["player", "line", "side", "price", "book"])
