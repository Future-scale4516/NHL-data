"""
app.py
NHL model vs market, one tab per market. Slate + odds are fetched on button press and stored
in session_state, so changing the sort dropdown or switching tabs never wipes the display.
"""

from datetime import date, datetime
import streamlit as st
from nhl_data import get_standings, get_prior_standings, compute_team_strengths, get_games
from model import run_game_model
from odds import fetch_odds, match_event, best_prices

EDGE_FLAG_PP = 8.0   # edges above this are more likely a model/data problem than real value

st.set_page_config(page_title="NHL Model", layout="centered")
st.title("NHL Game Model")


def get_api_key():
    for name in ("ODDS_API_KEY", "odds_api_key", "THE_ODDS_API_KEY", "API_KEY"):
        if name in st.secrets:
            return st.secrets[name]
    return None


@st.cache_data(ttl=900, show_spinner=False)   # 15 min, same as the MLB app
def cached_odds(api_key: str):
    return fetch_odds(api_key)


def edge_pp(model_p, offer):
    return (model_p - 1 / offer[0]) * 100 if offer else None


def market_line(label, model_p, offer):
    price, book = offer
    implied = 1 / price
    edge = (model_p - implied) * 100
    ev = (model_p * price - 1) * 100
    row = f"{label}: model {model_p:.0%} | {price:.2f} ({book}) = {implied:.0%} | edge {edge:+.1f}pp | EV {ev:+.1f}%"
    if edge > EDGE_FLAG_PP:
        return f"**{row}** ⚠"
    return f"**{row}**" if edge > 0 else row


def build_rows(g, market):
    """(label, model_prob, best_offer) rows for one game in one market; [] if not priced."""
    p = g["prices"]
    if not p:
        return []
    r, h, a = g["result"], g["home"], g["away"]
    ml, pl, tot = r["moneyline"], r["puck_line"], r["totals"]
    if market == "Moneyline":
        return [
            (f"{h} ML", ml["home_win_prob"], p["h2h"]["home"]),
            (f"{a} ML", ml["away_win_prob"], p["h2h"]["away"]),
        ]
    if market == "Puck line":
        return [
            (f"{h} -1.5", pl["home_-1.5"], p["spreads"]["home_-1.5"]),
            (f"{a} +1.5", pl["away_+1.5"], p["spreads"]["away_+1.5"]),
            (f"{a} -1.5", pl["away_-1.5"], p["spreads"]["away_-1.5"]),
            (f"{h} +1.5", pl["home_+1.5"], p["spreads"]["home_+1.5"]),
        ]
    if market == "Totals" and tot and p["totals"]:
        line = p["totals"]["line"]
        decided = tot["over"] + tot["under"]          # condition on no push
        return [
            (f"Over {line:g}", tot["over"] / decided, p["totals"]["over"]),
            (f"Under {line:g}", tot["under"] / decided, p["totals"]["under"]),
        ]
    return []


def kickoff(g):
    if not g["start_utc"]:
        return ""
    return datetime.fromisoformat(g["start_utc"].replace("Z", "+00:00")).strftime("%H:%M UTC")


def render_market(market, slate):
    sort_by = st.selectbox("Sort by", ["Best edge", "Model %", "Kickoff"], key=f"sort_{market}")
    entries, unpriced = [], []
    for g in slate:
        rows = [row for row in build_rows(g, market) if row[2]]   # only rows with a price
        (entries if rows else unpriced).append((g, rows))

    def best_edge(entry):
        return max(edge_pp(m, o) for _, m, o in entry[1])

    def best_model(entry):
        return max(m for _, m, _ in entry[1])      # the model's most confident side in this market

    if sort_by == "Best edge":
        entries.sort(key=best_edge, reverse=True)
    elif sort_by == "Model %":
        entries.sort(key=best_model, reverse=True)
    else:
        entries.sort(key=lambda e: e[0]["start_utc"] or "")

    for g, rows in entries:
        r = g["result"]
        with st.container(border=True):
            st.markdown(f"**{g['away']} @ {g['home']}**  ·  {kickoff(g)}")
            if market == "Totals":
                tot = r["totals"]
                cap = f"model total xG {r['home_exp_goals'] + r['away_exp_goals']:.2f}"
                if tot["push"] > 0.001:
                    cap += f"  |  push {tot['push']:.0%} (excluded from Over/Under %)"
                st.caption(cap)
            else:
                st.caption(f"xG {g['home']} {r['home_exp_goals']} - {r['away_exp_goals']} {g['away']}")
            st.caption("  \n".join(market_line(*row) for row in rows))
    for g, _ in unpriced:
        st.caption(f"{g['away']} @ {g['home']}: no {market.lower()} prices posted yet")
    if entries:
        st.caption(f"⚠ = edge above {EDGE_FLAG_PP:g}pp. Treat as suspect (model or data), not as value.")


day = st.date_input("Date", value=date.today())

if st.button("Load games"):
    api_key = get_api_key()
    if not api_key:
        st.error("No Odds API key found in secrets (tried ODDS_API_KEY, odds_api_key, THE_ODDS_API_KEY, API_KEY).")
        st.stop()
    with st.spinner("Loading slate and odds..."):
        try:
            teams = get_standings()
            prior = get_prior_standings()
            if not prior:
                st.warning("Last season's final standings couldn't be loaded, so team strengths are using "
                           "this season's few games only. Treat every number below as unreliable.")
            strengths = compute_team_strengths(teams, prior)
            games = get_games(day.isoformat())
            try:
                events = cached_odds(api_key)
            except Exception as e:
                events = []
                st.warning(f"Odds unavailable, showing model only: {e}")
            rows = []
            for g in games:
                if g["home"] not in strengths or g["away"] not in strengths:
                    continue
                event = match_event(events, g["home"], g["away"], g["start_utc"])
                prices = best_prices(event) if event else None
                line = prices["totals"]["line"] if prices and prices["totals"] else None
                g["result"] = run_game_model(g["home"], g["away"], strengths, total_line=line)
                g["prices"] = prices
                rows.append(g)
            st.session_state["nhl_slate"] = rows
        except Exception as e:
            st.error(f"Couldn't load slate: {e}")

slate = st.session_state.get("nhl_slate")
if slate is not None:
    if not slate:
        st.info("No games found for that date.")
    else:
        tabs = st.tabs(["Moneyline", "Puck line", "Totals"])
        for tab, market in zip(tabs, ["Moneyline", "Puck line", "Totals"]):
            with tab:
                render_market(market, slate)
