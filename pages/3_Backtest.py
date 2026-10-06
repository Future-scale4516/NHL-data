import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date
from nhl_backtest import build_game_frame, calibration, tuning_diagnostics, sweep_settings, MARKETS
from nhl_goalies import run_goalie_test, fetch_boxscores
from nhl_data import get_completed_games_range
from nhl_props import sog_backtest

st.set_page_config(page_title="NHL Model - Backtest", layout="centered")
st.title("📊 NHL Model Backtest")
st.caption("Replays completed games through the model and checks its probabilities against what "
           "actually happened. Well-calibrated means: when it says 60%, that happens about 60% of "
           "the time. Free (NHL API only), no odds or credits used. Unlike the MLB backtest there is "
           "no lookahead: every game is modelled from the standings as of the day before it was played.")


def render_calibration(df, name, p_col, y_col):
    c = calibration(df, p_col, y_col)
    if not c:
        return
    st.markdown(f"#### {name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brier score", c["brier"], delta=f"{c['brier'] - c['brier_naive']:+.4f} vs naive",
              delta_color="inverse",
              help="Lower is better. 'Naive' = always predicting the base rate. The model must beat it "
                   "(negative delta) to be adding any information.")
    m2.metric("Accuracy", f"{c['acc']}%")
    m3.metric("Base rate", f"{c['base_rate']}%")
    m4.metric("Model avg", f"{c['mean_p']}%", help="If this sits away from the base rate, the model is biased.")
    if not c["buckets"]:
        st.caption("Too few games per probability band to draw a calibration curve.")
        return
    cdf = pd.DataFrame(c["buckets"], columns=["Predicted band", "Games", "Model avg %", "Actual %"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="#888"), name="Perfect"))
    fig.add_trace(go.Scatter(x=cdf["Model avg %"], y=cdf["Actual %"], mode="markers+lines",
                             marker=dict(size=9), name="Model"))
    fig.update_layout(height=300, xaxis_title="Model predicted %", yaxis_title="Actual %",
                      margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cdf, width="stretch", hide_index=True)


c1, c2 = st.columns(2)
start = c1.date_input("From", value=date(2025, 10, 20), key="bt_start",
                      help="Last season: the new season has too few completed games to test yet.")
end = c2.date_input("To", value=date(2026, 1, 15), key="bt_end")
st.caption("Longer windows give steadier numbers (a full season is ~1,300 games) but take longer to "
           "fetch. Start a couple of weeks into the season, or expect early dates to be skipped.")

if st.button("Run backtest"):
    if end < start:
        st.error("'To' must be on or after 'From'.")
    else:
        bar = st.progress(0.0, text="Fetching results and point-in-time standings...")
        try:
            df, notes, ctx = build_game_frame(start, end, progress=lambda f: bar.progress(min(f, 1.0)))
            st.session_state["nhl_bt"] = (df, notes, str(start), str(end), ctx)
            st.session_state.pop("nhl_sweep", None)
            st.session_state.pop("nhl_goalie", None)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
        bar.empty()

res = st.session_state.get("nhl_bt")
if res:
    df, notes, s, e, ctx = res
    if df.empty:
        st.warning("No completed games found in that window.")
    else:
        st.caption(f"Window {s} to {e}.")
        for n in notes:
            (st.warning if n.startswith("⚠") else st.caption)(n)
        if len(df) < 300:
            st.info(f"Only {len(df)} games - calibration bands will be noisy. Widen the window before "
                    "changing any model constant.")

        st.markdown("## 🔧 Tuning diagnostics")
        st.caption("Model vs reality on the quantities the model's constants control. Fit on one window, "
                   "then re-run on a different one (e.g. Oct-Dec, then Jan-Apr) to confirm the change "
                   "holds up rather than just fitting this sample.")
        st.dataframe(tuning_diagnostics(df), width="stretch", hide_index=True)

        st.markdown("## 🎛️ Tune team-strength settings")
        st.caption("Re-scores the same games under different settings for how much weight last season's ratings "
                   "get (PRIOR_WEIGHT_GAMES: higher = trust current form more slowly) and how far they're pulled "
                   "toward average (PRIOR_REGRESSION). No API calls. Values are Brier vs the always-predict-the-base-"
                   "rate forecast, so **negative = beats naive, lower = better**. Prefer a setting that is good in "
                   "BOTH halves of the window, not just the best average: one window alone has misled us before.")
        if st.button("Run settings sweep"):
            bar2 = st.progress(0.0, text="Re-scoring games under each setting...")
            st.session_state["nhl_sweep"] = sweep_settings(ctx, progress=lambda f: bar2.progress(min(f, 1.0)))
            bar2.empty()
        sw = st.session_state.get("nhl_sweep")
        if sw is not None and not sw.empty:
            st.dataframe(sw.round(4), width="stretch", hide_index=True)
            st.caption("To apply a setting, edit PRIOR_WEIGHT_GAMES / PRIOR_REGRESSION at the top of nhl_data.py. "
                       "Differences under ~0.001 are noise on a window this size.")

        st.markdown("## 🥅 Goalie value test")
        st.caption("Does knowing the starting goalie improve the model? Pulls every box score for the season (one "
                   "call per game, about 1,300; throttled, and cached so a re-run only retries failures), rates "
                   "each goalie by save % as of each date (earlier games only, regressed toward average), and "
                   "compares him with his own team's usual goalie, since that is already baked into the team's "
                   "goals-against rating. This is the BEST CASE: it uses the goalie who actually started, so if it "
                   "doesn't help here, a live starter feed won't either. Weight 0 = off.")
        if st.button("Run goalie test"):
            bar3 = st.progress(0.0, text="Fetching box scores...")
            try:
                st.session_state["nhl_goalie"] = run_goalie_test(ctx, progress=lambda f: bar3.progress(min(f, 1.0)))
            except Exception as exc:                      # not `e`: that name holds the window end date below
                st.error(f"Goalie test failed: {exc}")
            bar3.empty()
        gt = st.session_state.get("nhl_goalie")
        if gt:
            cv = gt["coverage"]
            st.caption(f"{cv['games_scored']} games scored; goalie data for {cv['games_with_goalie_data']}; "
                       f"{cv['backup_games']} had a backup in net on at least one side.")
            if cv["boxscores_failed"]:
                st.warning(f"⚠ {cv['boxscores_failed']} box score(s) could not be fetched (e.g. {cv['first_error'][:140]}). "
                           "Click Run goalie test again: games already fetched are remembered.")
            sm = gt["summary"]
            st.dataframe(sm.round(4), width="stretch", hide_index=True)
            st.caption("Brier vs naive (negative = beats it). 'Avg vs off' is the change in average Brier from "
                       "turning the adjustment on, with a bootstrap interval over games: an interval entirely "
                       "below zero means a real improvement.")
            on = sm[sm["goalie weight"] > 0].sort_values("Avg vs off")
            best = on.iloc[0]
            off = sm.iloc[0]
            halves_ok = best["1st half"] <= off["1st half"] and best["2nd half"] <= off["2nd half"]
            if best["95% CI high"] < 0 and halves_ok:
                st.success(f"Knowing the starter helps: weight {best['goalie weight']:g} improves average Brier by "
                           f"{-best['Avg vs off']:.4f}, the interval excludes zero and both halves of the season "
                           "improve. Worth building the live starter feed.")
            elif best["95% CI high"] < 0:
                st.info("The overall improvement is real, but it doesn't show in both halves of the season. "
                        "Promising rather than proven.")
            elif best["95% CI low"] > 0:
                st.error("The adjustment makes the model worse. Don't build a live starter feed on this evidence.")
            else:
                st.info("No clear improvement (the interval spans zero). Even with perfect knowledge of the "
                        "starter, it doesn't move the model much, so a live feed isn't worth building yet.")
            st.markdown("#### Where does it matter?")
            st.caption("If goalies matter, the improvement should sit in games where a backup started, and the "
                       "'both usual starters' rows should barely move.")
            st.dataframe(gt["subsets"].round(4), width="stretch", hide_index=True)
            st.download_button("Download goalie game log CSV", data=gt["log"].to_csv(index=False).encode("utf-8"),
                               file_name="nhl_goalie_log.csv", mime="text/csv")

        st.markdown("## Calibration by market")
        for name, (p_col, y_col) in MARKETS.items():
            render_calibration(df, name, p_col, y_col)
        st.caption("This validates the model's calibration, NOT whether you'd beat a bookmaker - that needs "
                   "prices (see the Results page, priced mode). Goalie adjustment is off in the backtest, "
                   "matching the live app.")
        st.download_button("Download backtest CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"nhl_backtest_{s}_{e}.csv", mime="text/csv")


st.divider()
st.markdown("## 🎯 Player Prop Backtest: shots on goal")
st.caption("Checks the shots-on-goal model against real box scores from a whole past season: every prediction uses only "
           "earlier games. Same idea as the game backtest, but for player props. Free (NHL API only; about 1,300 box "
           "scores, throttled, and shared with the goalie test so a re-run only retries failures). 'Naive' = always "
           "predicting the base rate of that line for the player's position. The fitted dispersion is picked up "
           "automatically by the Player Props page.")
sog_y0 = st.selectbox("Season", [2025, 2024], format_func=lambda y: f"{y}-{str(y + 1)[2:]}", key="sog_season")
if st.button("Run shots-on-goal backtest"):
    sog_bar = st.progress(0.0, text="Fetching box scores...")
    try:
        sog_games = get_completed_games_range(date(sog_y0, 10, 1), date(sog_y0 + 1, 4, 30))
        _, sog_skl, sog_failed = fetch_boxscores(sog_games, progress=lambda f: sog_bar.progress(min(f, 1.0)))
        if sog_skl.empty:
            raise RuntimeError("No skater rows could be read from the box scores. "
                               + (f"Example: {next(iter(sog_failed.values()))[:260]}" if sog_failed else ""))
        sog_bar.progress(1.0, text="Scoring the model...")
        sog_res = sog_backtest(sog_skl)
        sog_res["failed"] = len(sog_failed)
        sog_res["first_error"] = next(iter(sog_failed.values()), None)
        sog_res["games"] = len(sog_games)
        st.session_state["sog_bt"] = sog_res
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
    sog_bar.empty()

sog_bt = st.session_state.get("sog_bt")
if sog_bt:
    st.caption(f"{sog_bt['n_rows']:,} player-games from {sog_bt['games'] - sog_bt['failed']} of {sog_bt['games']} games. "
               f"Fitted dispersion r = {sog_bt['r_on']:.1f}.")
    if sog_bt["failed"]:
        st.warning(f"⚠ {sog_bt['failed']} box score(s) could not be fetched (e.g. {sog_bt['first_error'][:140]}). Click "
                   "again: games already fetched are remembered.")
    sog_s = sog_bt["summary"]
    st.markdown("#### Calibration and skill (opponent factor on)")
    st.dataframe(sog_s[sog_s["variant"] == "opponent factor on"].drop(columns="variant"), width="stretch", hide_index=True)
    st.caption("'vs naive': Brier minus the always-predict-the-base-rate Brier, so negative = the model adds "
               "information. 'model avg' should sit on 'base rate'. Halves show whether it holds all season.")
    st.markdown("#### Does the opponent shot-suppression factor help?")
    sog_oc = sog_bt["opp_compare"]
    st.dataframe(sog_oc, width="stretch", hide_index=True)
    sog_helps = int((sog_oc["95% CI high"] < 0).sum())
    if sog_helps >= 2:
        st.success(f"Yes: it improves the Brier score at {sog_helps} of {len(sog_oc)} lines (interval below zero). Keep "
                   "OPP_WEIGHT = 1.")
    else:
        st.info("No clear gain from the opponent factor. Consider setting OPP_WEIGHT to 0 in nhl_props.py.")
    sog_f = sog_bt["frame"].assign(y25=lambda d: d["sog"] > 2.5)
    sog_c = calibration(sog_f, "p25", "y25")
    if sog_c and sog_c["buckets"]:
        sog_cdf = pd.DataFrame(sog_c["buckets"], columns=["Predicted band", "Players", "Model avg %", "Actual %"])
        sog_fig = go.Figure()
        sog_fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="#888"), name="Perfect"))
        sog_fig.add_trace(go.Scatter(x=sog_cdf["Model avg %"], y=sog_cdf["Actual %"], mode="markers+lines", name="Model"))
        sog_fig.update_layout(height=300, title="P(3+ shots): predicted vs actual", xaxis_title="Model %",
                              yaxis_title="Actual %", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(sog_fig, width="stretch")
