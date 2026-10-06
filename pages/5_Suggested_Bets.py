import pandas as pd
import streamlit as st
from datetime import date, datetime
from common import get_api_key, cached_odds
from nhl_live import load_slate, selections
from nhl_bets import pick_bets, build_doubles, settle, pnl, DEFAULT_TRUST, AMBER_STAKE_FACTOR
from nhl_lights import GREEN, AMBER, RED, NONE
from nhl_data import get_scoreboard
from nhl_ui import nhl_today, uk_time

st.set_page_config(page_title="NHL Model - Suggested Bets", layout="centered")
st.title("🎯 Suggested Bets")
st.warning("**Paper-trade first.** On last season the model matched the base rate on moneyline and totals, and the "
           "closing-line test hasn't shown an edge yet. This page applies guard rails (lights only, one pick per "
           "game, a hard cap, small flat stakes) but it is a tracker, not advice. Only stake what you can afford to "
           "lose, and only after the CLV evidence says the model earns it.")

c1, c2, c3 = st.columns(3)
day = c1.date_input("Date (NHL game date)", value=nhl_today(), key="sb_day")
bankroll = c2.number_input("Bankroll (£)", value=1000.0, min_value=0.0, step=100.0)
unit_pct = c3.slider("Unit = % of bankroll", 0.1, 1.0, 0.5, 0.1,
                     help="1 unit is this % of the bankroll. Stakes are multiples of a unit.")
d1, d2 = st.columns(2)
max_picks = d1.slider("Max picks", 1, 8, 5)
which = d2.radio("Lights allowed", ["🟢 only (recommended)", "🟢 + 🟡 (amber at half stake)"], horizontal=True)
with st.expander("Stake weight by market"):
    st.caption("Multiplies the unit. Puck line is the only market whose backtest beat the base rate (barely); "
               "moneyline and totals matched it; shots on goal is unvalidated.")
    trust = {m: st.slider(m, 0.0, 2.0, v, 0.25, key=f"trust_{m}") for m, v in DEFAULT_TRUST.items()}
doubles = st.checkbox("Also suggest doubles (green pairs, half stake)", value=False,
                      help="Off by default: a double multiplies the model's error and the bookmaker's margin.")

if st.button("Build suggested bets"):
    api_key = get_api_key()
    if not api_key:
        st.error("No Odds API key found in secrets.")
        st.stop()
    with st.spinner("Loading slate and odds..."):
        try:
            try:
                events = cached_odds(api_key)
            except Exception as exc:
                events, _ = [], st.warning(f"Odds unavailable: {exc}")
            rows, warnings = load_slate(day, events)
            st.session_state["sb"] = {"sel": selections(rows), "day": str(day), "warnings": warnings, "games": len(rows)}
            st.session_state.pop("sb_status", None)
        except Exception as exc:
            st.error(f"Couldn't build the slate: {exc}")

sb = st.session_state.get("sb")
if sb and sb["day"] == str(day):
    sel = sb["sel"]
    for w in sb["warnings"]:
        st.warning(w)
    if sel.empty:
        st.info(f"{sb['games']} game(s) on the slate but no priced selections yet.")
    else:
        n = sel["Light"].value_counts()
        st.caption(f"{n.get(GREEN, 0)} 🟢 · {n.get(AMBER, 0)} 🟡 · {n.get(RED, 0)} 🔴 · {n.get(NONE, 0)} ⚪ "
                   f"across {sel['Game ID'].nunique()} priced games")
        picks = pick_bets(sel, bankroll, unit_pct, allow_amber=which.startswith("🟢 +"), max_picks=max_picks, trust=trust)
        if picks.empty:
            st.info("**No bets today.** Nothing passes the checks, which is the normal outcome far more often than not.")
            blocked = sel[sel["Light"].isin([AMBER, RED])]["Why"].str.split("; ").explode().value_counts().head(4)
            if len(blocked):
                st.caption("What blocked the near-misses: " + " · ".join(f"{k} ({v})" for k, v in blocked.items()))
        else:
            show = picks.assign(Kickoff=picks["Start"].map(uk_time))
            cols = ["Light", "Game", "Kickoff", "Selection", "Odds", "Book", "Model %", "Market %", "Units", "Stake"]
            st.dataframe(show[cols], width="stretch", hide_index=True,
                         column_config={"Stake": st.column_config.NumberColumn("Stake (£)", format="£%.2f")})
            total = picks["Stake"].sum()
            st.caption(f"{len(picks)} pick(s) · total staked £{total:.2f} ({total / bankroll * 100:.1f}% of bankroll)" if bankroll
                       else f"{len(picks)} pick(s)")
            if doubles:
                dbl = build_doubles(picks)
                st.markdown("#### Doubles")
                if dbl.empty:
                    st.caption("Needs two green singles from different games.")
                else:
                    st.dataframe(dbl[["Leg 1", "Leg 2", "Combined odds", "Model prob %", "Units", "Stake"]],
                                 width="stretch", hide_index=True)

            st.markdown("#### Live status")
            if st.button("Check live status"):
                try:
                    board = {g["id"]: g for g in get_scoreboard(str(day))}
                    out = []
                    for _, p in picks.iterrows():
                        g = board.get(p["Game ID"], {"state": "pending"})
                        res = settle(p["Key"], p["Line"], g)
                        score = (f"{g['away']} {g['away_score']} - {g['home_score']} {g['home']}"
                                 if g.get("home_score") is not None else "")
                        out.append({"Selection": f"{p['Selection']} ({p['Game']})", "Status": {
                            "won": "✅ won", "lost": "❌ lost", "push": "➖ push", "live": "🔴 live",
                            "pending": "⏳ not started"}[res], "Score": score,
                            "P/L (£)": round(pnl(res, p["Odds"], p["Stake"]), 2)})
                    st.session_state["sb_status"] = pd.DataFrame(out)
                except Exception as exc:
                    st.error(f"Couldn't fetch scores: {exc}")
            status = st.session_state.get("sb_status")
            if status is not None:
                st.dataframe(status, width="stretch", hide_index=True)
                st.caption(f"Settled P/L so far: £{status['P/L (£)'].sum():+.2f}")
            st.download_button("Download picks CSV (your paper-trading log)",
                               data=picks.assign(Date=sb["day"]).drop(columns=["Why"]).to_csv(index=False).encode("utf-8"),
                               file_name=f"nhl_suggested_{sb['day']}.csv", mime="text/csv")
else:
    st.info("Pick a date and click **Build suggested bets**.")
