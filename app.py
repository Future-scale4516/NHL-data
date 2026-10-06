import streamlit as st
from nhl_ui import setup_page, sidebar_date

setup_page("NHL Model")
sel_date = sidebar_date()

st.title("🏒 NHL Model")
st.caption("A goals model for NHL game bets and shots-on-goal props, checked against the market with traffic lights "
           "that default to 'no bet'. Pick a date in the sidebar (it carries across pages), then:")
st.markdown("""
- **🎯 Game Bets** — Money Line, Puck Line and Totals vs the UK books, a 'Most Likely' tab and an accumulator builder
- **🎰 Player Props** — shots on goal, including a checker for bet365's 'X or more' ladders
- **📊 Backtest** — calibration, settings sweep and the goalie test on last season
- **📋 Results** — how past picks actually turned out, with priced-up reconstructions
- **📋 Suggested Bets** — a capped, flat-staked list from the lights, with live status
- **📈 CLV** — whether the market moves toward the model's picks
""")
st.info("**Where the evidence stands:** last season the model matched the base rate on moneyline and totals, and the "
        "closing-line test hasn't shown an edge. Treat everything as paper-trading until that changes. The lights are "
        "strict on purpose, and early in the season nothing rates better than 🟡.")
