"""
app.py
NHL slate view: model probabilities per game next to the best market price and edge.
Slate + odds are fetched on button press, stored in session_state, and rendered outside
the button so changing widgets afterwards doesn't wipe the display.
"""

from datetime import date, datetime
import streamlit as st
from nhl_data import get_standings, get_prior_standings, compute_team_strengths, get_games
from model import run_game_model
from odds import fetch_odds, match_event, best_prices

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


def market_line(label, model_p, offer):
    """One display row: model prob vs best price, implied prob, edge (pp) and EV."""
    if not offer:
        return f"{label}: model {model_p:.0%} | no price"
    price, book = offer
    implied = 1 / price
    edge = (model_p - implied) * 100
    ev = (model_p * price - 1) * 100
    row = f"{label}: model {model_p:.0%} | {price:.2f} ({book}) = {implied:.0%} | edge {edge:+.1f}pp | EV {ev:+.1f}%"
    return f"**{row}**" if edge > 0 else row


day = st.date_input("Date", value=date.today())

if st.button("Load games"):
    api_key = get_api_key()
    if not api_key:
        st.error("No Odds API key found in secrets (tried ODDS_API_KEY, odds_api_key, THE_ODDS_API_KEY, API_KEY).")
        st.stop()
    with st.spinner("Loading slate and odds..."):
        try:
            teams = get_standings()
            strengths = compute_team_strengths(teams, get_prior_standings())
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
    for g in slate:
        r, p = g["result"], g["prices"]
        h, a = g["home"], g["away"]
        ml, pl, tot = r["moneyline"], r["puck_line"], r["totals"]
        when = ""
        if g["start_utc"]:
            when = datetime.fromisoformat(g["start_utc"].replace("Z", "+00:00")).strftime("%H:%M UTC")
        with st.container(border=True):
            st.markdown(f"**{a} @ {h}**  ·  {when}")
            st.caption(f"xG {h} {r['home_exp_goals']} - {r['away_exp_goals']} {a}")
            if not p:
                st.caption(
                    f"No odds posted yet  |  ML {h} {ml['home_win_prob']:.0%} / {a} {ml['away_win_prob']:.0%}  |  "
                    f"PL {h} -1.5 {pl['home_-1.5']:.0%} / {a} +1.5 {pl['away_+1.5']:.0%}"
                )
                continue
            rows = [
                market_line(f"ML {h}", ml["home_win_prob"], p["h2h"]["home"]),
                market_line(f"ML {a}", ml["away_win_prob"], p["h2h"]["away"]),
                market_line(f"{h} -1.5", pl["home_-1.5"], p["spreads"]["home_-1.5"]),
                market_line(f"{a} +1.5", pl["away_+1.5"], p["spreads"]["away_+1.5"]),
                market_line(f"{a} -1.5", pl["away_-1.5"], p["spreads"]["away_-1.5"]),
                market_line(f"{h} +1.5", pl["home_+1.5"], p["spreads"]["home_+1.5"]),
            ]
            if tot and p["totals"]:
                line = p["totals"]["line"]
                decided = tot["over"] + tot["under"]          # condition on no push
                rows.append(market_line(f"Over {line:g}", tot["over"] / decided, p["totals"]["over"]))
                rows.append(market_line(f"Under {line:g}", tot["under"] / decided, p["totals"]["under"]))
                if tot["push"] > 0.001:
                    rows.append(f"(push on {line:g}: {tot['push']:.0%}, excluded from the Over/Under %)")
            st.caption("  \n".join(rows))
