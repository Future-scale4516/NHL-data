import streamlit as st
from datetime import date, timedelta
from common import get_api_key
from nhl_data import get_completed_games_range
from nhl_backtest import build_free_results, build_priced_results, pick_games, snapshot_ts
from odds import fetch_historical_odds, COST_LOG

st.set_page_config(page_title="NHL Model - Results", layout="centered")
st.title("📋 NHL Results")
st.caption("How the model's predictions turned out for a past date: what it projected, what happened, "
           "and whether it landed. Model-only mode is free. Priced mode adds the real odds that were "
           "available before each game, for edge and P/L.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_games(day_iso):
    d = date.fromisoformat(day_iso)
    return get_completed_games_range(d, d)


@st.cache_data(ttl=86400, show_spinner=False)   # re-opening the same date is free
def cached_snapshot(api_key, ts):
    events, meta = fetch_historical_odds(api_key, ts)
    COST_LOG.append(meta)                        # only runs on a real (uncached) call
    return events


day = st.date_input("Date", value=date.today() - timedelta(days=1), key="res_date")
mode = st.radio("Results mode", ["Model reads only (free)", "Priced-up picks (uses historical odds credits)"],
                help="Model reads scores every selection the model priced against what happened. Priced-up "
                     "adds the real best odds from a snapshot taken before each game, for true edge and P/L.")
priced = mode.startswith("Priced")

if priced:
    max_g = st.slider("Games to reconstruct", 1, 15, 6)
    try:
        chosen = pick_games(cached_games(day.isoformat()), max_g)
        n_snap = len({snapshot_ts(g["start_utc"]) for g in chosen if g["start_utc"]})
        st.warning(f"**Credit cost:** each distinct start time is one snapshot, 30 credits (10 per market x "
                   f"3 markets). Reconstructing {len(chosen)} game(s) on this date needs **{n_snap} "
                   f"snapshot(s), about {30 * n_snap} credits**. Cached for a day, so re-opening the same "
                   "date is free.")
    except Exception as e:
        st.caption(f"Couldn't pre-count games for the cost estimate: {e}")
    st.caption("Odds come from a snapshot 60 minutes before each game's own start, so every pick is priced "
               "at what was genuinely available shortly before it began.")

if st.button("Load results"):
    with st.spinner("Rebuilding picks and checking them against results..."):
        try:
            cost = None
            if priced:
                api_key = get_api_key()
                if not api_key:
                    st.error("No Odds API key found in secrets.")
                    st.stop()
                COST_LOG.clear()
                rdf, note = build_priced_results(day, max_g, lambda ts: cached_snapshot(api_key, ts))
                cost = {"credits": sum(m["last"] for m in COST_LOG),
                        "remaining": COST_LOG[-1]["remaining"] if COST_LOG else None}
            else:
                rdf, note = build_free_results(day)
            st.session_state["res"] = {"df": rdf, "note": note, "day": str(day), "priced": priced, "cost": cost}
        except Exception as e:
            st.error(f"Couldn't load results: {e}")

r = st.session_state.get("res")
if r and r["day"] == str(day):
    rdf = r["df"]
    st.caption(r["note"])
    if rdf is None or rdf.empty:
        st.warning(r["note"] or "No results available.")
    else:
        if r["cost"] and r["cost"]["credits"]:
            st.caption(f"Odds API: this run used ~{r['cost']['credits']} credits, {r['cost']['remaining']} remaining.")
        floor = st.slider("Only show picks the model rated at least this likely (%)", 0, 90,
                          0 if r["priced"] else 50, 5,
                          help="The model prices every side of every market. Filtering to its more confident "
                               "calls shows how the picks you'd actually have considered performed.")
        min_edge = st.slider("Minimum edge (pp)", -10, 15, 3, 1,
                             help="Only count selections where the model beat the price by at least this much.") \
            if r["priced"] else None

        sub = rdf[(rdf["Model %"] >= floor) & (~rdf["Push"])]
        if min_edge is not None:
            sub = sub[sub["Edge (pp)"] >= min_edge]
        if sub.empty:
            st.warning("No picks at or above those filters on this date.")
        else:
            hits, total = int(sub["Hit"].sum()), len(sub)
            rate, avg_model = hits / total * 100, sub["Model %"].mean()
            if r["priced"]:
                pl = sub["P/L (1u)"].sum()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Landed", f"{hits}/{total}")
                c2.metric("Hit rate", f"{rate:.1f}%")
                c3.metric("P/L (1u stakes)", f"{pl:+.2f}u")
                c4.metric("ROI", f"{pl / total * 100:+.1f}%")
                st.caption("Flat 1-unit stake on every pick shown, settled at the best price available 60 "
                           "minutes before each game. One slate is a data point, not a verdict.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Landed", f"{hits}/{total}")
                c2.metric("Actual hit rate", f"{rate:.1f}%")
                c3.metric("Model predicted", f"{avg_model:.1f}%")
            if abs(rate - avg_model) <= 5:
                st.success(f"Model said {avg_model:.1f}%, reality was {rate:.1f}% - closely calibrated on this slate.")
            elif rate < avg_model:
                st.warning(f"Model said {avg_model:.1f}% but only {rate:.1f}% landed - overconfident on this slate.")
            else:
                st.info(f"Model said {avg_model:.1f}% and {rate:.1f}% landed - underconfident on this slate.")
            st.caption("One slate is a small sample. The Backtest page is where calibration is judged properly.")

            tabs = st.tabs(["📊 All", "Moneyline", "Puck line", "Total"])
            for tab, mk in zip(tabs, [None, "Moneyline", "Puck line", "Total"]):
                with tab:
                    t = sub if mk is None else sub[sub["Market"] == mk]
                    if t.empty:
                        st.write("No picks in this market at those filters.")
                        continue
                    h, n = int(t["Hit"].sum()), len(t)
                    hdr = f"**{h}/{n} landed ({h / n * 100:.1f}%)** · model predicted {t['Model %'].mean():.1f}%"
                    if "P/L (1u)" in t.columns:
                        hdr += f" · P/L {t['P/L (1u)'].sum():+.2f}u"
                    st.markdown(hdr)
                    disp = t.copy()
                    disp["Result"] = disp["Hit"].map({True: "✅", False: "❌"})
                    cols = ["Result", "Selection", "Market", "Model %", "Implied %", "Edge (pp)", "Odds", "Book",
                            "P/L (1u)", "Game", "Score"] if r["priced"] else \
                           ["Result", "Selection", "Market", "Model %", "Game", "Score"]
                    cols = [c for c in cols if c in disp.columns and not (mk and c == "Market")]
                    st.dataframe(disp[cols].sort_values("Model %", ascending=False).reset_index(drop=True),
                                 width="stretch", hide_index=True,
                                 column_config={"Result": st.column_config.TextColumn("", width="small")})
            st.download_button("Download results CSV", data=sub.to_csv(index=False).encode("utf-8"),
                               file_name=f"nhl_results_{day}.csv", mime="text/csv")
else:
    st.info("Pick a date and click **Load results**.")
