import pandas as pd
import streamlit as st
from datetime import date
from common import get_api_key
from nhl_data import get_completed_games_range
from nhl_clv import build_clv_frame, clv_summary, plan_snapshots, slot_filter, pool_frames
from odds import fetch_historical_odds, credits_per_snapshot, COST_LOG
from nhl_lights import add_lights, lights_summary

st.set_page_config(page_title="NHL Model - CLV", layout="centered")
st.title("📈 Closing Line Value")
st.caption("The best test of whether the model finds real edges. For each game: take the best price available "
           "some time before the start (what you could have bet) and compare it with the de-vigged consensus at "
           "the close. If the picks the model flags beat the close more than ordinary selections do, the market "
           "moved toward the model. Uses historical odds credits.")
st.info("**Run it a week at a time.** Each run is cached for a day, but the cache disappears if the app restarts, so "
        "download each run's CSV and pool them at the bottom. Nothing you've already paid for is ever re-fetched or "
        "lost that way. Check your remaining credits on the Odds API dashboard first: MLB and NFL draw from the "
        "same pool.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_games(s_iso, e_iso):
    return get_completed_games_range(date.fromisoformat(s_iso), date.fromisoformat(e_iso))


@st.cache_data(ttl=86400, show_spinner=False)   # a snapshot already fetched is never paid for twice
def cached_snapshot(api_key, ts, markets):
    events, meta = fetch_historical_odds(api_key, ts, markets)
    COST_LOG.append(meta)                        # only runs on a real (uncached) call
    return events


c1, c2 = st.columns(2)
start = c1.date_input("From", value=date(2026, 3, 2), key="clv_start",
                      help="The second half of the season is where the model showed skill, so test there.")
end = c2.date_input("To", value=date(2026, 3, 8), key="clv_end")
c3, c4 = st.columns(2)
pick_lead = c3.selectbox("Pick price taken (min before start)", [60, 120, 180, 360], index=2,
                         help="Earlier = more time for the market to move toward (or away from) the model, "
                              "and for goalie news to land.")
close_lead = c4.selectbox("Closing price taken (min before start)", [5, 10], index=0)
c5, c6 = st.columns(2)
puck = c5.checkbox("Include the puck line", value=False,
                   help="Adds a third market (+10 credits per snapshot). UK books rarely price the puck line "
                        "through this feed, so it mostly adds cost, not data.")
min_slot = c6.selectbox("Only start times shared by at least N games", [1, 2, 3], index=1,
                        help="Games that start together share the same two snapshots, so busy slots are much "
                             "cheaper per game than a lone late start.")
markets = "h2h,spreads,totals" if puck else "h2h,totals"
cps = credits_per_snapshot(markets)
cap = st.slider("Max credits to spend this run", 200, 3000, 1000, 100)
max_snap = max(2, cap // cps)

try:
    games = cached_games(str(start), str(end))
    eligible = slot_filter(games, pick_lead, min_slot)
    covered, n_all, n_used = plan_snapshots(games, pick_lead, close_lead, max_snap, min_slot)
    st.warning(f"**Credit cost:** {len(games)} completed games in this range; {len(eligible)} qualify after the "
               f"start-time filter and need **{n_all} snapshots x {cps} credits = about {cps * n_all} credits** to "
               f"cover fully. Your cap covers {len(covered)} games using {n_used} snapshots, **about "
               f"{cps * n_used} credits** (less if some are already cached).")
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
        df, notes = build_clv_frame(start, end, lambda ts: cached_snapshot(api_key, ts, markets), pick_lead, close_lead,
                                    max_snap, progress=lambda f: bar.progress(min(f, 1.0)), min_games_per_slot=min_slot)
        cost = {"credits": sum(m["last"] for m in COST_LOG),
                "remaining": COST_LOG[-1]["remaining"] if COST_LOG else None}
        st.session_state["nhl_clv"] = {"df": df, "notes": notes, "cost": cost, "start": str(start), "end": str(end)}
    except Exception as e:
        st.error(f"CLV report failed: {e}")
    bar.empty()

r = st.session_state.get("nhl_clv")
st.markdown("#### Pool earlier runs")
uploads = st.file_uploader("Add CSVs from earlier runs (each game and selection is counted once)", type="csv",
                           accept_multiple_files=True)
frames = ([r["df"]] if r else []) + [pd.read_csv(u) for u in uploads]
df = pool_frames(frames)

if r:
    for n in r["notes"]:
        (st.warning if n.startswith("⚠") else st.caption)(n)
    if r["cost"]["credits"]:
        st.caption(f"Odds API: this run used ~{r['cost']['credits']} credits, {r['cost']['remaining']} remaining.")

if df.empty:
    st.info("No results yet. Build a report above, or upload CSVs from earlier runs.")
else:
    st.caption(f"Analysing {len(df)} priced selections from {df['Game ID'].nunique()} games"
               + (f" ({len(frames)} runs pooled)" if len(frames) > 1 else "") + ".")
    if True:
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

        if ok and mv["ci"]:
            half = (mv["ci"][1] - mv["ci"][0]) / 2
            st.caption(f"With this sample the interval is about ±{half:.2f}pp, so a real average market move smaller "
                       "than that can't be told apart from zero.")
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

        st.markdown("#### By traffic light")
        st.caption("Do the lights earn their keep? Green should show a larger market move than amber or red, and "
                   "red should show none. Only meaningful once there are a few hundred picks.")
        lit = add_lights(df, edge_col="Edge vs consensus (pp)", books_col="Close books", gp_col="Min GP")
        st.dataframe(lights_summary(lit), width="stretch", hide_index=True)

        with st.expander("All priced selections"):
            cols = [c for c in ["Date", "Game", "Selection", "Model %", "Pick fair %", "Edge vs consensus (pp)", "Odds",
                                "Book", "Close fair %", "Market move (pp)", "CLV (pp)", "Hit", "P/L (1u)"] if c in df.columns]
            st.dataframe(df[cols].sort_values("Edge vs consensus (pp)", ascending=False),
                         width="stretch", hide_index=True)
        if r is not None and len(r["df"]):
            st.download_button("Download this run's CSV", data=r["df"].to_csv(index=False).encode("utf-8"),
                               file_name=f"nhl_clv_{r['start']}_{r['end']}.csv", mime="text/csv")
        if len(frames) > 1:
            st.download_button("Download pooled CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="nhl_clv_pooled.csv", mime="text/csv")
