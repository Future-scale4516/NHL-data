"""
nhl_bets.py
Suggested bets: turn lit selections into a small, capped, flat-staked list, and settle them. No Streamlit.

Guard rails by design (the model has NOT shown an edge on moneyline or totals):
  - only selections that pass the lights are eligible (green by default; amber optional at a reduced stake)
  - at most one selection per game (moneyline, puck line and total on one game are one correlated bet)
  - a hard cap on the number of picks, flat stakes in units, never more than a small % of bankroll per pick
  - doubles are off by default: they multiply model error and the margin together
"""

from itertools import combinations

import pandas as pd

from nhl_lights import GREEN, AMBER, ORDER
from nhl_backtest import outcome

# Stake multipliers by market. Puck line is the only market whose backtest Brier beat the base rate (and its
# interval still touches zero); moneyline and totals matched it; shots on goal is unvalidated.
DEFAULT_TRUST = {"Puck line": 1.0, "Moneyline": 0.5, "Total": 0.5, "Shots on goal": 0.25}
AMBER_STAKE_FACTOR = 0.5
FINAL_STATES = ("OFF", "FINAL")


def edge_ref(row) -> float:
    e = row.get("Edge vs market (pp)")
    return e if pd.notna(e) else row["Edge (pp)"]


def pick_bets(sel: pd.DataFrame, bankroll: float, unit_pct: float = 0.5, allow_amber: bool = False,
              max_picks: int = 5, trust: dict = None, markets: list = None) -> pd.DataFrame:
    """Returns one row per suggested single, with Units and Stake. Empty frame if nothing qualifies."""
    trust = trust or DEFAULT_TRUST
    if sel is None or sel.empty:
        return pd.DataFrame()
    lights = [GREEN] + ([AMBER] if allow_amber else [])
    d = sel[sel["Light"].isin(lights)].copy()
    if markets is not None:
        d = d[d["Market"].isin(markets)]
    if d.empty:
        return pd.DataFrame()
    d["Edge ref"] = d.apply(edge_ref, axis=1)
    d["Trust"] = d["Market"].map(trust).fillna(0.25)
    d["Units"] = d["Trust"] * d["Light"].map({GREEN: 1.0, AMBER: AMBER_STAKE_FACTOR})
    d["_rank"] = d["Light"].map(ORDER)                                      # explicit order: never sort emoji as text
    d = d.sort_values(["_rank", "Edge ref"], ascending=[True, False])      # green before amber, then by edge
    d = d.drop_duplicates("Game ID", keep="first").head(max_picks).drop(columns="_rank")   # one per game, capped
    unit = bankroll * unit_pct / 100
    d["Stake"] = (d["Units"] * unit).round(2)
    return d.reset_index(drop=True)


def build_doubles(picks: pd.DataFrame, max_doubles: int = 4, stake_factor: float = 0.5) -> pd.DataFrame:
    """Pairs of GREEN singles from different games. Combined odds are the product; the stake is reduced."""
    g = picks[picks["Light"] == GREEN]
    rows = []
    for (_, a), (_, b) in combinations(g.iterrows(), 2):
        rows.append({"Leg 1": f"{a['Selection']} ({a['Game']})", "Leg 2": f"{b['Selection']} ({b['Game']})",
                     "Combined odds": round(a["Odds"] * b["Odds"], 2),
                     "Model prob %": round(a["Model %"] * b["Model %"] / 100, 1),
                     "Units": round(min(a["Units"], b["Units"]) * stake_factor, 3),
                     "Stake": round(min(a["Stake"], b["Stake"]) * stake_factor, 2),
                     "_keys": ((a["Game ID"], a["Key"], a["Line"]), (b["Game ID"], b["Key"], b["Line"])),
                     "_odds": (a["Odds"], b["Odds"]), "_edge": min(edge_ref(a), edge_ref(b))})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_edge", ascending=False).head(max_doubles).reset_index(drop=True)


def settle(key: str, line, game: dict) -> str:
    """
    'won' / 'lost' / 'push' for a finished game, 'live' while in play, 'pending' before the start.
    game: {'state','home_score','away_score','last_period'}. Same rules the backtest uses: the puck line settles
    on the FINAL score (a shootout win is by 1), totals exclude the shootout-deciding goal.
    """
    state = game.get("state")
    if state not in FINAL_STATES:
        return "live" if state in ("LIVE", "CRIT") else "pending"
    o = outcome({"home_score": game["home_score"], "away_score": game["away_score"],
                 "last_period": game.get("last_period", "REG")})
    margin, total = o["margin"], o["total_goals"]
    market, _, sel = key.partition(":")
    if market == "h2h":
        return "won" if (margin > 0) == (sel == "home") else "lost"
    if market == "spreads":
        side, pt = sel.split("_")
        pt = float(pt)
        cover = (margin if side == "home" else -margin) + pt
        return "won" if cover > 0 else "lost"
    if market == "totals":
        if total == line:
            return "push"
        return "won" if (total > line) == (sel == "over") else "lost"
    return "pending"


def pnl(result: str, odds: float, stake: float) -> float:
    return {"won": stake * (odds - 1), "lost": -stake}.get(result, 0.0)
