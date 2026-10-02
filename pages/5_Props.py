import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date
from common import get_api_key, cached_odds
from nhl_data import get_completed_games_range
from nhl_live import load_slate
from nhl_goalies import fetch_boxscores
from nhl_backtest import calibration, season_start_year
from nhl_lights import GREEN, AMBER, RED, NONE, LIGHT_CONFIG
from nhl_props import (NB_R, sog_backtest, fetch_stats_table, player_table, team_factors, props_selections)
from odds import fetch_event_props, COST_LOG

st.set_page_config(page_title="NHL Model - Shots on goal", layout="centered")
st.title("🏒 Shots on goal")
st.caption("Per player: shots per minute (regressed toward his position's average) x recent ice time x how many shots "
           "the opponent allows, then a negative binomial for P(over a line). The calibration tab checks it against "
           "real box scores for free. It has NOT been compared with historical prices, so treat live edges as "
           "paper-trade only.")


@st.cache_data(ttl=21600, show_spinner=False)
def cached_stats(report, season_id):
    return fetch_stats_table(report, season_id)


@st.cache_data(ttl=900, show_spinner=False)
def cached_props(api_key, event_id):
    event, meta = fetch_event_props(api_key, event_id)
    COST_LOG.append(meta)
    return event


live_tab, cal_tab = st.tabs(["Live props", "Calibration backtest"])

with cal_tab:
    st.caption("Replays a whole season of box scores through the model, using only earlier games for every prediction. "
               "Fetches about 1,300 box scores (throttled; shared with the goalie test, so a re-run only retries "
               "failures). Naive = always predicting the base rate of that line for the player's position.")
    y0 = st.selectbox("Season", [2025, 2024], format_func=lambda y: f"{y}-{str(y + 1)[2:]}")
    if st.button("Run shots-on-goal backtest"):
        bar = st.progress(0.0, text="Fetching box scores...")
        try:
            games = get_completed_games_range(date(y0, 10, 1), date(y0 + 1, 4, 30))
            _, skl, failed = fetch_boxscores(games, progress=lambda f: bar.progress(min(f, 1.0)))
            if skl.empty:
                raise RuntimeError("No skater rows could be read from the box scores. "
                                   + (f"Example: {next(iter(failed.values()))[:260]}" if failed else ""))
            bar.progress(1.0, text="Scoring the model...")
            res = sog_backtest(skl)
            res["failed"] = len(failed)
            res["first_error"] = next(iter(failed.values()), None)
            res["games"] = len(games)
            st.session_state["sog_bt"] = res
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
        bar.empty()
    bt = st.session_state.get("sog_bt")
    if bt:
        st.caption(f"{bt['n_rows']:,} player-games from {bt['games'] - bt['failed']} of {bt['games']} games. "
                   f"Fitted dispersion r = {bt['r_on']:.1f} (set it under Live props).")
        if bt["failed"]:
            st.warning(f"⚠ {bt['failed']} box score(s) could not be fetched (e.g. {bt['first_error'][:140]}). Click "
                       "again: games already fetched are remembered.")
        s = bt["summary"]
        st.markdown("#### Calibration and skill (opponent factor on)")
        st.dataframe(s[s["variant"] == "opponent factor on"].drop(columns="variant"), width="stretch", hide_index=True)
        st.caption("'vs naive': Brier minus the always-predict-the-base-rate Brier, so negative = the model adds "
                   "information. 'model avg' should sit on 'base rate'. Halves show whether it holds all season.")
        st.markdown("#### Does the opponent shot-suppression factor help?")
        oc = bt["opp_compare"]
        st.dataframe(oc, width="stretch", hide_index=True)
        helps = int((oc["95% CI high"] < 0).sum())
        if helps >= 2:
            st.success(f"Yes: it improves the Brier score at {helps} of {len(oc)} lines (interval below zero). Keep "
                       "OPP_WEIGHT = 1.")
        else:
            st.info("No clear gain from the opponent factor. Consider setting OPP_WEIGHT to 0 in nhl_props.py.")
        f = bt["frame"].assign(y25=lambda d: d["sog"] > 2.5)
        c = calibration(f, "p25", "y25")
        if c and c["buckets"]:
            cdf = pd.DataFrame(c["buckets"], columns=["Predicted band", "Players", "Model avg %", "Actual %"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="#888"), name="Perfect"))
            fig.add_trace(go.Scatter(x=cdf["Model avg %"], y=cdf["Actual %"], mode="markers+lines", name="Model"))
            fig.update_layout(height=300, title="P(3+ shots): predicted vs actual", xaxis_title="Model %",
                              yaxis_title="Actual %", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

with live_tab:
    day = st.date_input("Date", value=date.today(), key="sog_day")
    r_disp = st.number_input("Dispersion r", value=float(bt["r_on"]) if bt else NB_R, min_value=2.0, max_value=200.0,
                             step=1.0, help="Lower = fatter tails. The calibration backtest fits it.")
    st.caption("Costs about 1 Odds API credit per game that has props (cached for 15 minutes). UK books may carry few "
               "or no shots-on-goal props: an empty result means none were offered, not that the model failed.")
    if st.button("Load props"):
        api_key = get_api_key()
        if not api_key:
            st.error("No Odds API key found in secrets.")
            st.stop()
        with st.spinner("Loading slate, player stats and props..."):
            try:
                try:
                    events = cached_odds(api_key)
                except Exception as exc:
                    events = []
                    st.warning(f"Odds unavailable: {exc}")
                rows, warns = load_slate(day, events)
                y = season_start_year(day)
                table = player_table(cached_stats("skater/summary", f"{y}{y + 1}"),
                                     cached_stats("skater/summary", f"{y - 1}{y}"))
                try:
                    tfac = team_factors(cached_stats("team/summary", f"{y}{y + 1}"), cached_stats("team/summary", f"{y - 1}{y}"))
                except Exception:
                    tfac = {}
                COST_LOG.clear()
                items, errs = [], []
                for g in rows:
                    if not g.get("event_id"):
                        continue
                    try:
                        items.append((g, cached_props(api_key, g["event_id"])))
                    except Exception as exc:
                        errs.append(f"{g['away']} @ {g['home']}: {exc}")
                sel, info = props_selections(items, table, tfac, r=r_disp)
                st.session_state["sog"] = {"sel": sel, "info": info, "day": str(day), "warns": warns, "errs": errs,
                                           "games": len(rows), "tfac": bool(tfac),
                                           "credits": sum(m["last"] for m in COST_LOG),
                                           "remaining": COST_LOG[-1]["remaining"] if COST_LOG else None}
            except Exception as exc:
                st.error(f"Couldn't load props: {exc}")
    sg = st.session_state.get("sog")
    if sg and sg["day"] == str(day):
        for w in sg["warns"]:
            st.warning(w)
        if sg["errs"]:
            st.warning("Some games' props failed to load: " + "; ".join(sg["errs"])[:300])
        if not sg["tfac"]:
            st.caption("Opponent shot-suppression data unavailable, so no opponent adjustment was applied.")
        if sg["credits"]:
            st.caption(f"Odds API: this run used ~{sg['credits']} credits, {sg['remaining']} remaining.")
        sel = sg["sel"]
        if sel.empty:
            st.info(f"No shots-on-goal prices found for {sg['games']} game(s). UK books often don't offer them through "
                    "this feed.")
        else:
            n = sel["Light"].value_counts()
            st.caption(f"{n.get(GREEN, 0)} 🟢 · {n.get(AMBER, 0)} 🟡 · {n.get(RED, 0)} 🔴 · {n.get(NONE, 0)} ⚪ across "
                       f"{sel['Game ID'].nunique()} game(s); {sg['info']['props']} player lines")
            if sg["info"]["unmatched"]:
                st.caption(f"Couldn't match {len(sg['info']['unmatched'])} name(s) to the stats table: "
                           + ", ".join(sg["info"]["unmatched"][:8]))
            cfg = LIGHT_CONFIG["Shots on goal"]
            lights = st.multiselect("Lights", [GREEN, AMBER, RED, NONE], default=[GREEN, AMBER, RED])
            st.caption(f"Provisional rules: edge from {cfg['edge_min']:g}pp, green up to {cfg['edge_green_max']:g}, red from "
                       f"{cfg['edge_red']:g}, and a multi-book consensus is required for green. Not yet validated "
                       "against historical prices.")
            show = sel[sel["Light"].isin(lights)].copy()
            order = {GREEN: 0, AMBER: 1, RED: 2, NONE: 3}
            show["_o"] = show["Light"].map(order)
            show["_e"] = show["Edge vs market (pp)"].fillna(show["Edge (pp)"])
            show = show.sort_values(["_o", "_e"], ascending=[True, False])
            cols = ["Light", "Selection", "Game", "Model mean", "Model %", "Odds", "Book", "Market %", "Books",
                    "Edge vs market (pp)", "EV %"]
            st.dataframe(show[cols], width="stretch", hide_index=True)
            st.download_button("Download props CSV", data=sel.to_csv(index=False).encode("utf-8"),
                               file_name=f"nhl_sog_{sg['day']}.csv", mime="text/csv")
