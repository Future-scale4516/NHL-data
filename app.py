"""
app.py
NHL model vs market, one tab per market, with traffic lights. Slate + odds are fetched on button press
and stored in session_state, so changing a dropdown or switching tabs never wipes the display.
"""

from datetime import date, datetime
import pandas as pd
import streamlit as st
from nhl_live import load_slate, selections
from nhl_lights import ORDER, LIGHT_CONFIG, EARLY_SEASON_GP, GREEN, AMBER, RED, NONE
from common import get_api_key, cached_odds

st.set_page_config(page_title="NHL Model", layout="centered")
st.title("NHL Game Model")


def row_text(r):
    market = f"market {r['Market %']:.0f}% ({int(r['Books'])} bk)" if pd.notna(r["Market %"]) else "market n/a"
    edge = r["Edge vs market (pp)"] if pd.notna(r["Edge vs market (pp)"]) else r["Edge (pp)"]   # best-price edge when <2 books
    txt = (f"{r['Light']} {r['Selection']}: model {r['Model %']:.0f}% | {r['Odds']:.2f} ({r['Book']}) | "
           f"{market} | edge {edge:+.1f}pp | EV {r['EV %']:+.1f}%")
    if r["Light"] == RED:
        txt += f"  \n&nbsp;&nbsp;&nbsp;↳ {r['Why']}"
    return f"**{txt}**" if r["Light"] == GREEN else txt


def kickoff(start):
    return datetime.fromisoformat(start.replace("Z", "+00:00")).strftime("%H:%M UTC") if start else ""


def render_market(market, label, df, slate):
    c1, c2 = st.columns(2)
    sort_by = c1.selectbox("Sort by", ["Best light", "Best edge", "Model %", "Kickoff"], key=f"sort_{market}")
    show = c2.selectbox("Show", ["All rows", "Lights only (🟢🟡🔴)", "🟢 only"], key=f"show_{market}")
    d = df[df["Market"] == market]
    if show.startswith("Lights"):
        d = d[d["Light"] != NONE]
    elif show.startswith("🟢"):
        d = d[d["Light"] == GREEN]
    by_game = {gid: grp for gid, grp in d.groupby("Game ID", sort=False)}
    games = [g for g in slate if g["id"] in by_game]

    def edge_of(grp):
        return grp["Edge vs market (pp)"].fillna(grp["Edge (pp)"]).max()

    if sort_by == "Best light":
        games.sort(key=lambda g: (min(ORDER[x] for x in by_game[g["id"]]["Light"]), -edge_of(by_game[g["id"]])))
    elif sort_by == "Best edge":
        games.sort(key=lambda g: -edge_of(by_game[g["id"]]))
    elif sort_by == "Model %":
        games.sort(key=lambda g: -by_game[g["id"]]["Model %"].max())
    else:
        games.sort(key=lambda g: g["start_utc"] or "")

    for g in games:
        grp, r = by_game[g["id"]], g["result"]
        with st.container(border=True):
            st.markdown(f"**{g['away']} @ {g['home']}**  ·  {kickoff(g['start_utc'])}")
            if market == "Total":
                tot = r["totals"]
                cap = f"model total xG {r['home_exp_goals'] + r['away_exp_goals']:.2f}"
                if tot and tot["push"] > 0.001:
                    cap += f"  |  push {tot['push']:.0%} (excluded from Over/Under %)"
                st.caption(cap)
            else:
                st.caption(f"xG {g['home']} {r['home_exp_goals']} - {r['away_exp_goals']} {g['away']}")
            st.caption("  \n".join(row_text(row) for _, row in grp.iterrows()))
    shown = set(by_game)
    for g in slate:
        if g["id"] not in shown and not df[(df["Game ID"] == g["id"]) & (df["Market"] == market)].shape[0]:
            st.caption(f"{g['away']} @ {g['home']}: no {label.lower()} prices posted yet")


with st.expander("What do the lights mean?"):
    st.markdown(
        f"{GREEN} **passes every check** (not a prediction that it wins). {AMBER} **marginal**: edge above the "
        f"usual range, no multi-book market to check against, or early season. {RED} **suspect**: edge so big it's "
        f"more likely a model blind spot, or the model is more extreme than it was calibrated on. {NONE} **no edge**.\n\n"
        f"Rules (from the 2025-26 backtest): edge counts only over the **market consensus**; moneyline needs "
        f"{LIGHT_CONFIG['Moneyline']['edge_min']:g}pp (green up to {LIGHT_CONFIG['Moneyline']['edge_green_max']:g}, "
        f"red from {LIGHT_CONFIG['Moneyline']['edge_red']:g}), totals {LIGHT_CONFIG['Total']['edge_min']:g}pp "
        f"(red from {LIGHT_CONFIG['Total']['edge_red']:g}). Before **{EARLY_SEASON_GP} games** per team nothing is "
        "better than amber: last season the model only beat the base rate in the second half.")

day = st.date_input("Date", value=date.today())

if st.button("Load games"):
    api_key = get_api_key()
    if not api_key:
        st.error("No Odds API key found in secrets (tried ODDS_API_KEY, odds_api_key, THE_ODDS_API_KEY, API_KEY).")
        st.stop()
    with st.spinner("Loading slate and odds..."):
        try:
            try:
                events = cached_odds(api_key)
            except Exception as e:
                events = []
                st.warning(f"Odds unavailable, showing model only: {e}")
            rows, warnings = load_slate(day, events)
            for w in warnings:
                st.warning(w)
            st.session_state["nhl_slate"] = rows
            st.session_state["nhl_sel"] = selections(rows)
        except Exception as e:
            st.error(f"Couldn't load slate: {e}")

slate = st.session_state.get("nhl_slate")
sel = st.session_state.get("nhl_sel")
if slate is not None:
    if not slate:
        st.info("No games found for that date.")
    else:
        if sel is not None and not sel.empty:
            n = sel["Light"].value_counts()
            st.caption(f"{n.get(GREEN, 0)} 🟢  ·  {n.get(AMBER, 0)} 🟡  ·  {n.get(RED, 0)} 🔴  ·  "
                       f"{n.get(NONE, 0)} ⚪   across {sel['Game ID'].nunique()} priced games")
            gp = sel["Min GP"].dropna()
            if len(gp) and gp.min() < EARLY_SEASON_GP:
                st.info(f"Early season: the least-played team in a game has {int(gp.min())} to {int(gp.max())} games, "
                        f"so nothing can be better than 🟡 until {EARLY_SEASON_GP}.")
        base = sel if sel is not None else pd.DataFrame(columns=["Market", "Light", "Game ID"])
        tabs = st.tabs(["Moneyline", "Puck line", "Totals"])
        for tab, (market, label) in zip(tabs, [("Moneyline", "Moneyline"), ("Puck line", "Puck line"), ("Total", "Totals")]):
            with tab:
                render_market(market, label, base, slate)
