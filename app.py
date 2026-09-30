"""
app.py
NHL slate view: pick a date, load that day's games, see model output per game.
Fetch results live in session_state and render outside the button, so changing
any widget afterwards doesn't wipe the display.
"""

from datetime import date, datetime
import streamlit as st
from nhl_data import get_standings, get_prior_standings, compute_team_strengths, get_games
from model import run_game_model

st.set_page_config(page_title="NHL Model", layout="centered")
st.title("NHL Game Model")

day = st.date_input("Date", value=date.today())
total_line = st.number_input("Total line (O/U)", value=6.0, step=0.5)

if st.button("Load games"):
    with st.spinner("Loading slate..."):
        try:
            teams = get_standings()
            strengths = compute_team_strengths(teams, get_prior_standings())
            games = get_games(day.isoformat())
            rows = []
            for g in games:
                if g["home"] in strengths and g["away"] in strengths:
                    g["result"] = run_game_model(g["home"], g["away"], strengths, total_line=total_line)
                    rows.append(g)
            st.session_state["nhl_slate"] = rows
        except Exception as e:
            st.error(f"Couldn't load slate: {e}")

slate = st.session_state.get("nhl_slate")
if slate is not None:
    if not slate:
        st.info("No games found for that date.")
    for g in slate:
        r = g["result"]
        ml, pl, tot = r["moneyline"], r["puck_line"], r["totals"]
        when = ""
        if g["start_utc"]:
            when = datetime.fromisoformat(g["start_utc"].replace("Z", "+00:00")).strftime("%H:%M UTC")
        with st.container(border=True):
            st.markdown(f"**{g['away']} @ {g['home']}**  ·  {when}")
            st.caption(
                f"xG {g['home']} {r['home_exp_goals']} - {r['away_exp_goals']} {g['away']}  |  "
                f"ML {g['home']} {ml['home_win_prob']:.0%} / {g['away']} {ml['away_win_prob']:.0%}  |  "
                f"PL {g['home']} -1.5 {pl['home_-1.5']:.0%} / {g['away']} +1.5 {pl['away_+1.5']:.0%}  |  "
                f"O{total_line} {tot['over']:.0%} / U {tot['under']:.0%}"
            )
