"""
nhl_live.py
Today's slate with the model, best prices, market consensus and traffic lights. No Streamlit imports,
so the app, Suggested Bets and Props pages all build selections the same way.
"""

import pandas as pd

from nhl_data import get_standings, get_prior_standings, compute_team_strengths, get_games
from model import run_game_model
from odds import match_event, best_prices, fair_probs
from nhl_lights import assign_light


def load_slate(day, events: list[dict]):
    """Returns (rows, warnings). One row per scheduled game that has strengths for both teams."""
    warnings = []
    teams = get_standings()
    prior = get_prior_standings()
    if not prior:
        warnings.append("Last season's final standings couldn't be loaded, so team strengths are using this "
                        "season's few games only. Treat every number below as unreliable.")
    strengths = compute_team_strengths(teams, prior)
    rows = []
    for g in get_games(day.isoformat()):
        if g["home"] not in strengths or g["away"] not in strengths:
            continue
        event = match_event(events, g["home"], g["away"], g["start_utc"])
        prices = best_prices(event) if event else None
        line = prices["totals"]["line"] if prices and prices["totals"] else None
        g["result"] = run_game_model(g["home"], g["away"], strengths, total_line=line)
        g["prices"] = prices
        g["fair"] = fair_probs(event, line=line) if event else {}
        g["event_id"] = event.get("id") if event else None
        g["min_gp"] = min(strengths[g["home"]]["games_played"], strengths[g["away"]]["games_played"])
        rows.append(g)
    return rows, warnings


def _tuples(g):
    """(market, label, line, model prob, key, best offer) for every selection the model prices."""
    p = g["prices"]
    if not p:
        return []
    r, h, a = g["result"], g["home"], g["away"]
    ml, pl, tot = r["moneyline"], r["puck_line"], r["totals"]
    out = [
        ("Moneyline", f"{h} ML", "", ml["home_win_prob"], "h2h:home", p["h2h"]["home"]),
        ("Moneyline", f"{a} ML", "", ml["away_win_prob"], "h2h:away", p["h2h"]["away"]),
        ("Puck line", f"{h} -1.5", -1.5, pl["home_-1.5"], "spreads:home_-1.5", p["spreads"]["home_-1.5"]),
        ("Puck line", f"{a} +1.5", 1.5, pl["away_+1.5"], "spreads:away_+1.5", p["spreads"]["away_+1.5"]),
        ("Puck line", f"{a} -1.5", -1.5, pl["away_-1.5"], "spreads:away_-1.5", p["spreads"]["away_-1.5"]),
        ("Puck line", f"{h} +1.5", 1.5, pl["home_+1.5"], "spreads:home_+1.5", p["spreads"]["home_+1.5"]),
    ]
    if tot and p["totals"]:
        line = p["totals"]["line"]
        decided = tot["over"] + tot["under"]            # condition on no push
        out += [("Total", f"Over {line:g}", line, tot["over"] / decided, "totals:over", p["totals"]["over"]),
                ("Total", f"Under {line:g}", line, tot["under"] / decided, "totals:under", p["totals"]["under"])]
    return out


def selections(rows: list[dict]) -> pd.DataFrame:
    """One row per priced selection, with consensus edge and a traffic light."""
    recs = []
    for g in rows:
        for market, label, line, mp, key, offer in _tuples(g):
            if not offer:
                continue
            price, book = offer
            cons = g["fair"].get(key)                     # (no-vig prob, number of books) or None
            edge_best = (mp - 1 / price) * 100
            edge_cons = (mp - cons[0]) * 100 if cons and cons[1] >= 2 else None   # a 1-book "consensus" isn't one
            light, why = assign_light(market, mp, edge_cons if edge_cons is not None else edge_best,
                                      cons[1] if cons else None, g.get("min_gp"))
            recs.append({
                "Game ID": g["id"], "Game": f"{g['away']} @ {g['home']}", "Start": g["start_utc"],
                "Home": g["home"], "Away": g["away"], "Market": market, "Selection": label, "Line": line,
                "Key": key, "Model %": round(mp * 100, 1), "Odds": price, "Book": book,
                "Implied %": round(100 / price, 1), "Edge (pp)": round(edge_best, 1),
                "Market %": round(cons[0] * 100, 1) if cons else None, "Books": cons[1] if cons else 0,
                "Edge vs market (pp)": round(edge_cons, 1) if edge_cons is not None else None,
                "EV %": round((mp * price - 1) * 100, 1), "Min GP": g.get("min_gp"),
                "Light": light, "Why": why})
    return pd.DataFrame(recs)
