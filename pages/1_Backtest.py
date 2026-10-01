import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date
from nhl_backtest import build_game_frame, calibration, tuning_diagnostics, sweep_settings, MARKETS

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

        st.markdown("## Calibration by market")
        for name, (p_col, y_col) in MARKETS.items():
            render_calibration(df, name, p_col, y_col)
        st.caption("This validates the model's calibration, NOT whether you'd beat a bookmaker - that needs "
                   "prices (see the Results page, priced mode). Goalie adjustment is off in the backtest, "
                   "matching the live app.")
        st.download_button("Download backtest CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"nhl_backtest_{s}_{e}.csv", mime="text/csv")
