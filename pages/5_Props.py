import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date
from common import get_api_key, cached_odds
from nhl_data import get_completed_games_range, get_games, get_boxscore
from nhl_live import load_slate
from nhl_goalies import fetch_boxscores
from nhl_backtest import calibration, season_start_year
from nhl_lights import GREEN, AMBER, RED, NONE, LIGHT_CONFIG
from nhl_props import (NB_R, sog_backtest, fetch_stats_table, player_table, team_factors, props_selections,
                       parse_ladder_text, ladder_table, settle_ladder, ladder_summary)
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


@st.cache_data(ttl=600, show_spinner=False)
def cached_games(day_iso):
    return get_games(day_iso)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_box(game_id):
    return get_boxscore(game_id)


def load_tables(day):
    """Skater table (this + last season) and opponent shot-suppression factors."""
    y = season_start_year(day)
    table = player_table(cached_stats("skater/summary", f"{y}{y + 1}"), cached_stats("skater/summary", f"{y - 1}{y}"))
    try:
        tfac = team_factors(cached_stats("team/summary", f"{y}{y + 1}"), cached_stats("team/summary", f"{y - 1}{y}"))
    except Exception:
        tfac = {}
    return table, tfac


live_tab, ladder_tab, cal_tab = st.tabs(["Live props", "bet365 ladder", "Calibration backtest"])

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

with ladder_tab:
    st.caption("bet365 isn't in the odds feed, so type its 'X or more shots' ladder in here. Each rung is priced with the "
               "model and compared with bet365's price. Log what you check, then settle it against real box scores "
               "later: that is the only way to find out whether the model beats bet365's actual prices.")
    ldate = st.date_input("Date", value=date.today(), key="lad_day")
    try:
        slate_games = cached_games(str(ldate))
    except Exception as exc:
        slate_games = []
        st.warning(f"Couldn't load the schedule: {exc}")
    if not slate_games:
        st.info("No games found for that date.")
    else:
        labels = {f"{g['away']} @ {g['home']}": g for g in slate_games}
        pick = st.selectbox("Game", list(labels))
        text = st.text_area("Paste or type the ladder (one player per line: name, then the prices for 1+, 2+, 3+ ...)",
                            height=170, key="lad_text",
                            placeholder="Andrei Svechnikov 1.04 1.25 1.79 3.05 5.50\nOwen Tippett 1.066 1.40 2.30 4.30 8.50")
        ladder_r = st.number_input("Dispersion r", value=float(bt["r_on"]) if bt else NB_R, min_value=2.0,
                                   max_value=200.0, step=1.0, key="lad_r")
        if st.button("Price the ladder"):
            entries, bad = parse_ladder_text(text)
            if not entries:
                st.error("No ladder lines found. Format: Player Name 1.04 1.25 1.79 3.05 5.50")
            else:
                try:
                    table, tfac = load_tables(ldate)
                    ldf, linfo = ladder_table(entries, labels[pick], table, tfac, r=ladder_r)
                    st.session_state["lad"] = {"df": ldf, "info": linfo, "bad": bad, "day": str(ldate), "game": pick}
                except Exception as exc:
                    st.error(f"Couldn't price the ladder: {exc}")
        lad = st.session_state.get("lad")
        if lad and lad["day"] == str(ldate) and lad["game"] == pick:
            if lad["bad"]:
                st.caption("Skipped lines I couldn't read: " + " | ".join(lad["bad"][:4]))
            if lad["info"]["unmatched"]:
                st.warning("Couldn't match to a player in this game: " + ", ".join(lad["info"]["unmatched"]))
            ldf = lad["df"]
            if not ldf.empty:
                st.markdown("#### Free bet: the best rungs")
                st.caption("A free bet returns winnings only, so its value is p x (odds - 1) per £1. That favours longer odds "
                           "even at fair prices, so the best free-bet rungs are usually the 3+ to 5+ ones. The model's "
                           "probability is what separates one long shot from another, and it has not been shown to "
                           "beat bet365. Avoid any 🔴. A value near or above 1.0 means the model is claiming bet365's price "
                           "is wrong by a lot, which is far more likely to be the model's error.")
                top = ldf[ldf["Light"] != "🔴"].sort_values("EV free bet", ascending=False).head(5)
                st.dataframe(top[["Light", "Player", "Rung", "Odds", "Implied %", "Model %", "EV free bet"]],
                             width="stretch", hide_index=True)
                st.markdown("#### Every rung")
                st.dataframe(ldf[["Light", "Player", "Rung", "Odds", "Implied %", "Model %", "Edge (pp)", "EV cash %",
                                  "EV free bet", "Model mean", "Ladder mean (~)", "Season shots/gp"]],
                             width="stretch", hide_index=True)
                st.caption("'Ladder mean' is the shots per game bet365's own prices imply (it reads a little high because "
                           "the prices include their margin). If it's far from 'Model mean' the model is probably missing "
                           "a lineup, line or injury change, and the rung goes 🔴. Check the starting lineup first.")
                st.download_button("Download this check (your paper-trading log)", ldf.assign(Date=lad["day"])
                                   .to_csv(index=False).encode("utf-8"), file_name=f"nhl_ladder_{lad['day']}.csv",
                                   mime="text/csv")
    st.markdown("#### Settle an earlier log")
    up = st.file_uploader("Upload a log you downloaded above", type="csv", key="lad_up")
    if up is not None and st.button("Settle with box scores"):
        try:
            settled = settle_ladder(pd.read_csv(up), cached_box)
            st.session_state["lad_settled"] = settled
        except Exception as exc:
            st.error(f"Couldn't settle the log: {exc}")
    settled = st.session_state.get("lad_settled")
    if settled is not None:
        summ = ladder_summary(settled)
        if summ.empty:
            st.info("Nothing to settle yet: those games haven't finished or no rungs were readable.")
        else:
            st.dataframe(summ, width="stretch", hide_index=True)
            st.caption("'Hit %' vs 'Model said %' and 'Price implied %': if the model is better than bet365's prices, hit "
                       "rate should track what the model said, not what the price implied. It takes a couple of hundred "
                       "rungs to tell. Void = the player didn't play; pending = game unfinished.")
            st.dataframe(settled[["Player", "Rung", "Odds", "Model %", "Shots", "Result", "P/L cash (1u)",
                                  "P/L free bet (1u)"]], width="stretch", hide_index=True)

