import streamlit as st
from datetime import date, timedelta
from common import get_api_key
from nhl_data import get_completed_games_range
from nhl_clv import build_clv_frame, clv_summary, plan_snapshots, CREDITS_PER_SNAPSHOT
from odds import fetch_historical_odds, COST_LOG

st.set_page_config(page_title="NHL Model - CLV", layout="centered")
st.title("📈 Closing Line Value")
st.caption("The best test of whether the model finds real edges. For each game: take the best price available "
           "some time before the start (what you could have bet) and compare it with the de-vigged consensus at "
           "the close. If the picks the model flags beat the close more than ordinary selections do, the market "
           "moved toward the model. Uses historical odds credits.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_games(s_iso, e_iso):
    return get_completed_games_range(date.fromisoformat(s_iso), date.fromisoformat(e_iso))


@st.cache_data(ttl=86400, show_spinner=False)   # a snapshot already fetched is never paid for twice
def cached_snapshot(api_key, ts):
    events, meta = fetch_historical_odds(api_key, ts)
    COST_LOG.append(meta)                        # only runs on a real (uncached) call
    return events


c1, c2 = st.columns(2)
start = c1.date_input("From", value=date(2026, 3, 23), key="clv_start")
end = c2.date_input("To", value=date(2026, 3, 29), key="clv_end")
c3, c4 = st.columns(2)
pick_lead = c3.selectbox("Pick price taken (min before start)", [60, 120, 180, 360], index=2,
                         help="Earlier = more time for the market to move toward (or away from) the model, "
                              "and for goalie news to land.")
close_lead = c4.selectbox("Closing price taken (min before start)", [5, 10], index=0)
max_snap = st.slider("Max snapshots to fetch", 6, 80, 30, 2,
                     help=f"Each snapshot costs {CREDITS_PER_SNAPSHOT} credits. Games are taken in date order "
                          "until the cap is reached.")

try:
    games = cached_games(str(start), str(end))
    covered, n_all, n_used = plan_snapshots(games, pick_lead, close_lead, max_snap)
    st.warning(f"**Credit cost:** {len(games)} completed games in this range need **{n_all} distinct snapshots "
               f"(about {CREDITS_PER_SNAPSHOT * n_all} credits)** to cover fully. With your cap this run covers "
               f"{len(covered)} games using {n_used} snapshots, **about {CREDITS_PER_SNAPSHOT * n_used} credits** "
               "(less if some are already cached).")
except Exception as e:
    st.caption(f"Couldn't pre-count games for the cost estimate: {e}")

if pick_lead <= close_lead + 15:
    st.error("The pick time must be well before the close.")
elif st.button("Build CLV report"):
    api_key = get_api_key()
    if not api_key:
        st.error("No Odds API key found in secrets.")
        st.stop()
    bar = st.progress(0.0, text="Fetching standings and odds snapshots...")
    try:
        COST_LOG.clear()
        df, notes = build_clv_frame(start, end, lambda ts: cached_snapshot(api_key, ts), pick_lead, close_lead,
                                    max_snap, progress=lambda f: bar.progress(min(f, 1.0)))
        cost = {"credits": sum(m["last"] for m in COST_LOG),
                "remaining": COST_LOG[-1]["remaining"] if COST_LOG else None}
        st.session_state["nhl_clv"] = {"df": df, "notes": notes, "cost": cost}
    except Exception as e:
        st.error(f"CLV report failed: {e}")
    bar.empty()

r = st.session_state.get("nhl_clv")
if r:
    df = r["df"]
    for n in r["notes"]:
        (st.warning if n.startswith("⚠") else st.caption)(n)
    if r["cost"]["credits"]:
        st.caption(f"Odds API: this run used ~{r['cost']['credits']} credits, {r['cost']['remaining']} remaining.")
    if df.empty:
        st.warning("No usable picks. Check that the date range is a past regular-season stretch with odds coverage.")
    else:
        f1, f2 = st.columns(2)
        min_edge = f1.slider("Flag a pick when the model beats the CONSENSUS price by at least (pp)", 0, 10, 3, 1,
                             help="Measured against the average de-vigged price at pick time, not the best price: "
                                  "flagging on the best price over-selects one book's outlier, which reverts by the "
                                  "close and looks like skill.")
        floor = f2.slider("...and the model rates it at least (%)", 0, 70, 0, 5)
        s = clv_summary(df, min_edge, floor)
        mv, cl = s["move"], s["clv"]
        ok = s["n_flag"] > 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Flagged picks", s["n_flag"])
        m2.metric("Market move, flagged", f"{mv['flagged']:+.2f}pp" if ok else "n/a",
                  help="Average shift in the consensus probability between the pick and the close, in the pick's "
                       "favour. Positive = the market moved toward the pick.")
        m3.metric("Market move, all selections", f"{mv['all']:+.2f}pp", help="Baseline: about zero by construction.")
        m4, m5, m6 = st.columns(3)
        m4.metric("Flagged minus baseline", f"{mv['diff']:+.2f}pp" if ok else "n/a",
                  delta=(f"95% CI {mv['ci'][0]:+.2f} to {mv['ci'][1]:+.2f}" if ok and mv["ci"] else None),
                  delta_color="off", help="The headline number: did the market move toward the picks the model flagged?")
        m5.metric("CLV at best price, flagged minus baseline", f"{cl['diff']:+.2f}pp" if ok else "n/a",
                  delta=(f"95% CI {cl['ci'][0]:+.2f} to {cl['ci'][1]:+.2f}" if ok and cl["ci"] else None),
                  delta_color="off", help="What you'd actually have realised at the best available price. Supporting "
                  "evidence only: in testing it carries a small positive bias (~0.2pp) from price noise.")
        m6.metric("Flagged picks the market moved toward", f"{mv['beat']:.0f}%" if ok else "n/a")

        if s["n_flag"] < 30 or not mv["ci"]:
            st.info(f"Only {s['n_flag']} flagged picks: too few to conclude anything. Widen the date range, raise "
                    "the snapshot cap, or lower the edge threshold.")
        elif mv["ci"][0] > 0 and cl["diff"] > 0:
            st.success("The market moved toward the picks the model flagged, by more than it did for ordinary "
                       "selections (interval excludes zero), and the best available price beat the close. "
                       "Encouraging. Repeat it on a different stretch of games before trusting it.")
        elif mv["ci"][0] > 0:
            st.info("The market moved toward flagged picks, but the best price you could get didn't beat the "
                    "close. The signal may be real while the timing or price isn't capturable. Try a different "
                    "pick time.")
        elif mv["ci"][1] < 0:
            st.error("The market moved AWAY from flagged picks relative to the baseline: the model's edges look "
                     "like noise or a blind spot (goalies are the likely one).")
        else:
            st.info("No detectable difference: the interval spans zero. That's what 'no proven edge yet' looks "
                    "like, not proof there is none. More games would narrow it.")

        st.markdown("#### By market")
        st.dataframe(s["by_market"].round(2), width="stretch", hide_index=True)
        st.markdown("#### By model edge over the consensus at pick time")
        st.caption("If the model has real information, higher-edge picks should show a larger market move. A flat "
                   "or backwards pattern means the edge isn't real.")
        st.dataframe(s["by_edge"].round(2), width="stretch", hide_index=True)

        with st.expander("All priced selections"):
            cols = ["Date", "Game", "Selection", "Model %", "Pick fair %", "Edge vs consensus (pp)", "Odds", "Book",
                    "Close fair %", "Market move (pp)", "CLV (pp)", "Hit", "P/L (1u)"]
            st.dataframe(df[cols].sort_values("Edge vs consensus (pp)", ascending=False),
                         width="stretch", hide_index=True)
        st.download_button("Download CLV CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"nhl_clv_{start}_{end}.csv", mime="text/csv")
