"""
app.py
Bare-bones Streamlit page for the NHL game model scaffold — the NHL equivalent of the
MLB app's original Game-by-Game view, before Suggested Bets / traffic lights / etc.
Get this working end-to-end against live standings first, then layer the rest on top.
"""

import streamlit as st
from nhl.data import get_standings, compute_team_strengths, get_starting_goalie_stats
from nhl.model import run_game_model, GoalieAdjustment

st.set_page_config(page_title="NHL Model", layout="centered")
st.title("NHL Game Model")

CURRENT_SEASON = "20262027"

with st.spinner("Loading standings..."):
    try:
        teams = get_standings()
        strengths = compute_team_strengths(teams)
    except Exception as e:
        st.error(f"Couldn't load standings: {e}")
        st.stop()

team_abbrevs = sorted([k for k in strengths.keys() if not k.startswith("_")])

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", team_abbrevs)
with col2:
    away_team = st.selectbox("Away team", team_abbrevs, index=1)

total_line = st.number_input("Total line (O/U)", value=6.0, step=0.5)

use_goalie_adj = st.checkbox("Apply goalie adjustment (leave off until confirmed-starter data is wired up)")

home_goalie_adj = away_goalie_adj = None
if use_goalie_adj:
    with st.spinner("Loading goalie stats..."):
        try:
            home_g = get_starting_goalie_stats(home_team, CURRENT_SEASON)
            away_g = get_starting_goalie_stats(away_team, CURRENT_SEASON)
            if home_g:
                home_goalie_adj = GoalieAdjustment(home_g["save_pct"], home_g["team_save_pct"])
                st.caption(f"Home starter: {home_g['name']} ({home_g['save_pct']})")
            if away_g:
                away_goalie_adj = GoalieAdjustment(away_g["save_pct"], away_g["team_save_pct"])
                st.caption(f"Away starter: {away_g['name']} ({away_g['save_pct']})")
        except Exception as e:
            st.warning(f"Goalie stats unavailable, continuing without adjustment: {e}")

if st.button("Run model"):
    result = run_game_model(
        home_team, away_team, strengths,
        total_line=total_line,
        home_goalie_adj=home_goalie_adj,
        away_goalie_adj=away_goalie_adj,
    )

    st.subheader("Expected goals")
    st.write(f"{home_team}: {result['home_exp_goals']}  |  {away_team}: {result['away_exp_goals']}")

    st.subheader("Moneyline")
    ml = result["moneyline"]
    st.write(f"{home_team} win: {ml['home_win_prob']:.1%}  |  {away_team} win: {ml['away_win_prob']:.1%}")
    st.caption(f"(regulation tie probability before OT/SO split: {ml['reg_tie_prob']:.1%})")

    st.subheader("Puck line (-1.5 / +1.5)")
    pl_raw = result["puck_line_raw"]
    pl = result["puck_line"]
    st.write(f"Raw (no empty-net adj): {home_team} -1.5: {pl_raw['home_-1.5_raw']:.1%}")
    st.write(f"Corrected: {home_team} -1.5: {pl['home_-1.5']:.1%}  |  {away_team} +1.5: {pl['away_+1.5']:.1%}")

    st.subheader(f"Total ({total_line})")
    tot = result["totals"]
    st.write(f"Over: {tot['over']:.1%}  |  Under: {tot['under']:.1%}")
