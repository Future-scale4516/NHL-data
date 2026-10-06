import pandas as pd
import streamlit as st
from nhl_ui import setup_page, sidebar_date, render_pick_card, sort_picker, uk_time
from common import get_api_key, cached_odds
from nhl_live import load_slate, selections, likely_table
from nhl_lights import GREEN, AMBER, RED, NONE, LIGHT_CONFIG, EARLY_SEASON_GP
import odds as odds_mod

setup_page("NHL Model — Game Bets")
sel_date = sidebar_date()

st.markdown("## 🎯 Game Bets — Money Line · Puck Line · Totals")
st.caption("Estimates each team's goals with a Poisson model (team attack and defence, home ice), then compares it "
           "with the consensus of the UK books to surface edges. Positive edge = the model rates the bet better than "
           "the market price. Always confirm the live price at your book before staking: lines move.")

if st.button("Analyse game bets (UK odds)"):
    api_key = get_api_key()
    if not api_key:
        st.error("No Odds API key found in secrets (tried ODDS_API_KEY, odds_api_key, THE_ODDS_API_KEY, API_KEY).")
        st.stop()
    with st.spinner("Fetching schedule, team stats and UK odds..."):
        try:
            events = cached_odds(api_key)
        except Exception as exc:
            events = []
            st.warning(f"Odds unavailable, showing model only: {exc}")
        try:
            rows, warnings = load_slate(sel_date, events)
            # Stored here and rendered from the persistent block below: touching a Sort dropdown reruns the page
            # with this button False again, so nothing nested under it would survive.
            st.session_state["nhl_gb"] = {"rows": rows, "warnings": warnings, "sel": selections(rows),
                                          "likely": likely_table(rows), "date": str(sel_date)}
        except Exception as exc:
            st.session_state.pop("nhl_gb", None)
            st.error(f"Couldn't load the slate: {exc}")

gb = st.session_state.get("nhl_gb")
if gb:
    rows, sel, likely = gb["rows"], gb["sel"], gb["likely"]
    if gb["date"] != str(sel_date):
        st.info(f"Showing {gb['date']}. Click **Analyse game bets** to load {sel_date}.")
    q = odds_mod.LAST_QUOTA
    if q.get("remaining"):
        st.caption(f"Odds quota — used {q.get('used')}, remaining {q.get('remaining')}")
    for w in gb["warnings"]:
        st.warning(w)
    if not rows:
        st.info("No games found for that date.")
    else:
        has_odds = sel is not None and not sel.empty
        if has_odds:
            n = sel["Light"].value_counts()
            st.markdown(f"### {GREEN} {n.get(GREEN, 0)} green · {AMBER} {n.get(AMBER, 0)} amber · "
                        f"{RED} {n.get(RED, 0)} red")
            cm, ct = LIGHT_CONFIG["Moneyline"], LIGHT_CONFIG["Total"]
            st.caption(f"{GREEN} edge {cm['edge_min']:g}–{cm['edge_green_max']:g} pts over the market (totals "
                       f"{ct['edge_min']:g}–{ct['edge_green_max']:g}) = believable value · {AMBER} above that, only one "
                       f"book to check against, or early season = treat with caution · {RED} {cm['edge_red']:g}+ pts "
                       f"(totals {ct['edge_red']:g}+) = almost certainly something the model can't see (goalie, injury), "
                       f"or the model is more extreme than it was calibrated on · {NONE} no signal. Before "
                       f"{EARLY_SEASON_GP} games a team, nothing rates better than {AMBER}: last season the model only "
                       "beat the base rate in the second half.")
            gp = sel["Min GP"].dropna()
            if len(gp) and gp.min() < EARLY_SEASON_GP:
                st.info(f"Early season: the least-played team in a game has {int(gp.min())}–{int(gp.max())} games, so "
                        f"nothing can be better than {AMBER} until {EARLY_SEASON_GP}.")
        else:
            st.info("No UK odds posted for these games yet. The **Most Likely** tab still works (it ignores odds).")

        def show_market(tab, market_name):
            with tab:
                if not has_odds:
                    st.write("No odds available for this market today.")
                    return
                sub = sel[sel["Market"] == market_name].copy()
                if sub.empty:
                    st.write("No odds available for this market today.")
                    return
                sub["_edge"] = sub["Edge vs market (pp)"].fillna(sub["Edge (pp)"])
                hide = st.checkbox(f"Hide {NONE} no-signal selections", value=True, key=f"hide_{market_name}")
                if hide:
                    sub = sub[sub["Light"] != NONE]
                if sub.empty:
                    st.write(f"Nothing in this market clears the {NONE} no-signal line today.")
                    return
                sub = sort_picker(sub, [("Start time", "Start", True), ("Edge (high to low)", "_edge", False),
                                        ("Model % (high to low)", "Model %", False),
                                        ("Odds (high to low)", "Odds", False)], key=f"sort_{market_name}")
                for _, r in sub.iterrows():
                    fair = f"{r['Market %']:.1f}%" + (f" ({int(r['Books'])} bk)" if r["Books"] else "") \
                        if pd.notna(r["Market %"]) else "n/a"
                    reason = r["Reason"] + (f" · {r['Why']}" if r["Light"] in (AMBER, RED) else "")
                    render_pick_card(r["Light"], f"{r['Selection']} @ {r['Odds']:.2f}",
                                     f"{r['Game']} · {uk_time(r['Start'])}",
                                     [("Model %", f"{r['Model %']:.1f}%"), ("Fair %", fair),
                                      ("Edge", f"{r['_edge']:+.1f} pts"), ("EV %", f"{r['EV %']:+.1f}%")],
                                     reason=reason)

        def show_most_likely(tab):
            with tab:
                st.caption("Pure model confidence, ignoring the market entirely: 'what does the model predict', not "
                           "'where's the value'. Ignore the odds for this question. A high % is still not a "
                           "guarantee: last season the model's confidence only matched the base rate on moneyline.")
                mkts = st.multiselect("Markets to include:", ["Moneyline", "Puck line", "Total"],
                                      default=["Moneyline", "Puck line", "Total"], key="ml_market_pick")
                floor = st.slider("Only selections the model rates at least (%)", 50, 90, 60, 5, key="ml_floor")
                sub = likely[likely["Market"].isin(mkts) & (likely["Model %"] >= floor)].copy()
                if sub.empty:
                    st.write("No selections match the chosen markets and threshold.")
                    return
                sub = sort_picker(sub, [("Model % (high to low)", "Model %", False), ("Start time", "Start", True)],
                                  key="sort_most_likely_gb")
                for _, r in sub.iterrows():
                    render_pick_card(None, f"{r['Selection']} · {r['Market']}", f"{r['Game']} · {uk_time(r['Start'])}",
                                     [("Model %", f"{r['Model %']:.1f}%")])

        ml_tab, pl_tab, tot_tab, likely_tab = st.tabs(["💰 Money Line", "📏 Puck Line", "📊 Totals", "🎯 Most Likely"])
        show_market(ml_tab, "Moneyline")
        show_market(pl_tab, "Puck line")
        show_market(tot_tab, "Total")
        show_most_likely(likely_tab)
        st.caption("Model %: our probability · Fair %: the books' de-vigged consensus (and how many books) · Edge: model "
                   "minus fair · EV %: expected return per unit at the best price. Very large edges usually mean the "
                   "model is missing something about that game, not real value.")

        if has_odds:
            st.markdown("### 🎟️ Accumulator builder")
            st.caption(f"Builds a multi-fold from {GREEN} green selections. Tick the box to include {AMBER} amber ones "
                       "too (early in the season nothing is better than amber). Combined odds and the model's chance "
                       "of the whole bet landing are worked out for you.")
            amber_ok = st.checkbox(f"Include {AMBER} amber selections", value=False, key="acc_amber")
            pool = sel[sel["Light"].isin([GREEN] + ([AMBER] if amber_ok else []))].copy()
            if pool.empty:
                st.write("No qualifying selections on the analysed slate to build an accumulator from.")
            else:
                pool["_edge"] = pool["Edge vs market (pp)"].fillna(pool["Edge (pp)"])
                pool = pool.sort_values("_edge", ascending=False).reset_index(drop=True)
                pool["pick"] = pool.apply(lambda r: f"{r['Selection']}  @ {r['Odds']:.2f}  ·  {r['Market']}  ·  "
                                                    f"{r['Game']}  (edge {r['_edge']:.1f})", axis=1)
                chosen = st.multiselect("Choose your legs (each is one selection):", pool["pick"].tolist())
                stake = st.number_input("Stake (£)", min_value=0.0, value=10.0, step=1.0)
                if chosen:
                    legs = pool[pool["pick"].isin(chosen)]
                    odds = model = 1.0
                    for _, r in legs.iterrows():
                        odds *= float(r["Odds"])
                        model *= float(r["Model %"]) / 100.0
                    ret = stake * odds
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Legs", len(legs))
                    a2.metric("Combined odds", f"{odds:.2f}")
                    a3.metric(f"Return on £{stake:.0f}", f"£{ret:.2f}", f"+£{ret - stake:.2f}")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Model: chance it lands", f"{model * 100:.1f}%")
                    b2.metric("Market-implied chance", f"{100 / odds:.1f}%")
                    b3.metric("Combined EV", f"{(model * odds - 1) * 100:+.1f}%")
                    dupes = [g for g, c in legs["Game"].value_counts().items() if c > 1]
                    if dupes:
                        st.warning("⚠️ Multiple legs from the same game (" + ", ".join(dupes) + "). Those outcomes are "
                                   "correlated, so the combined chance above is optimistic, and a bookmaker's Bet "
                                   "Builder price for them will differ from these odds multiplied together.")
                    for _, r in legs.iterrows():
                        render_pick_card(r["Light"], f"{r['Selection']} @ {r['Odds']:.2f}", f"{r['Game']} · {r['Market']}",
                                         [("Model %", f"{r['Model %']:.1f}%"), ("Edge", f"{r['_edge']:+.1f} pts")],
                                         reason=r["Why"] if r["Light"] in (AMBER, RED) else None)
                    st.caption("The model chance assumes the legs are independent (true across different games). A multi "
                               "needs every leg to win, so it is high-variance even when each leg has an edge.")
