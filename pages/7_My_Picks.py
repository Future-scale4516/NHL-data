import json
import pandas as pd
import streamlit as st
from nhl_ui import setup_page, sidebar_date, uk_time
import nhl_tracker as tr

setup_page("NHL Model - My Picks")
sidebar_date()
st.title("📒 My Picks")
st.caption("Every pick you saved from Game Bets, Suggested Bets or Player Props, settled against real results. "
           "This is the evidence page: it says whether the lights and the model earn their keep, on picks you "
           "actually made.")

token = tr.has_token()
if not token:
    st.warning("**No GitHub token in secrets**, so picks only live for this session. Add `GITHUB_TOKEN` (a personal "
               "access token with the *gist* scope) to the Streamlit secrets and picks will persist in a private "
               "gist. Optionally add `NHL_GIST_ID` to point at a specific gist; otherwise one is created for you.")

if "tracker_picks" not in st.session_state:
    try:
        if token:
            picks, gid = tr.load_picks(st.session_state.get("tracker_gist"))
            st.session_state["tracker_gist"] = gid
        else:
            picks = st.session_state.get("tracker_local", [])
        st.session_state["tracker_picks"] = picks
    except Exception as exc:
        st.error(f"Couldn't load picks: {exc}")
        st.session_state["tracker_picks"] = []
picks = st.session_state["tracker_picks"]


def persist(updated):
    st.session_state["tracker_picks"] = updated
    if token:
        st.session_state["tracker_gist"] = tr.save_all(updated, st.session_state.get("tracker_gist"))
    else:
        st.session_state["tracker_local"] = updated


c1, c2, c3 = st.columns(3)
if c1.button("🔄 Reload"):
    st.session_state.pop("tracker_picks", None)
    st.rerun()
open_n = sum(1 for p in picks if p["status"] == tr.OPEN)
if c2.button(f"✅ Settle open picks ({open_n})", disabled=not open_n):
    with st.spinner("Checking results..."):
        try:
            updated, n, errs = tr.settle_open(picks, tr.live_get_board, tr.live_get_skaters)
            if n:
                persist(updated)
            st.success(f"Settled {n} pick(s)." if n else "Nothing to settle yet: games not finished.")
            for e in errs[:5]:
                st.warning(e)
        except Exception as exc:
            st.error(f"Settling failed: {exc}")

up = c3.file_uploader("Import CSV", type="csv", label_visibility="collapsed")
if up is not None and st.session_state.get("tracker_imported") != up.name:
    try:
        rows = pd.read_csv(up).where(lambda d: d.notna(), None).to_dict("records")
        rows = [r for r in rows if r.get("id") and r.get("key")]
        merged, added = tr.merge(picks, rows)
        persist(merged)
        st.session_state["tracker_imported"] = up.name
        st.success(f"Imported {added} new pick(s).")
        picks = merged
    except Exception as exc:
        st.error(f"Couldn't import: {exc}")

if not picks:
    st.info("No saved picks yet. Use **Save these picks** on Suggested Bets, Game Bets or Player Props.")
    st.stop()

df = tr.to_frame(picks)
df["Kickoff"] = df["start"].map(lambda s: uk_time(s) if s else "")
df["Result"] = df["status"].map({"won": "✅", "lost": "❌", "push": "➖", "void": "🚫", "open": "⏳"})

settled = df[df["status"].isin([tr.WON, tr.LOST, tr.PUSH])]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Picks", len(df), f"{open_n} open")
if settled.empty:
    m2.metric("Settled", 0)
else:
    dec = settled[settled["status"].isin([tr.WON, tr.LOST])]
    m2.metric("Hit rate", f"{(dec['status'] == tr.WON).mean() * 100:.1f}%" if len(dec) else "-",
              f"model said {pd.to_numeric(dec['model_pct'], errors='coerce').mean():.1f}%" if len(dec) else None,
              delta_color="off")
    m3.metric("Units (1u flat)", f"{settled['pnl_units'].sum():+.2f}u",
              f"ROI {settled['pnl_units'].sum() / len(settled) * 100:+.1f}%", delta_color="off")
    m4.metric("Cash P/L", f"£{settled['pnl_cash'].sum():+.2f}")
    if len(settled) < 50:
        st.caption(f"Only {len(settled)} settled pick(s). Under ~100 this tells you almost nothing: a few lucky "
                   "or unlucky results swing it. Keep logging.")

    for by, title in (("market", "By market"), ("light", "By light"), ("source", "By page")):
        s = tr.summarize(df, by)
        if not s.empty:
            st.markdown(f"#### {title}")
            st.dataframe(s, width="stretch", hide_index=True)

st.markdown("#### All picks")
cols = ["Result", "game_date", "selection", "game", "Kickoff", "market", "light", "odds", "book", "model_pct",
        "stake", "free_bet", "pnl_cash", "result_note", "source"]
st.dataframe(df.sort_values(["game_date", "start"], ascending=False)[cols], width="stretch", hide_index=True,
             column_config={"Result": st.column_config.TextColumn("", width="small")})

with st.expander("Delete a pick"):
    opts = {f"{p['game_date']} · {p['selection']} ({p['game']})": p["id"] for p in picks}
    pick = st.selectbox("Pick", list(opts))
    if st.button("Delete selected pick"):
        try:
            persist([p for p in picks if p["id"] != opts[pick]])
            st.rerun()
        except Exception as exc:
            st.error(f"Couldn't delete: {exc}")

st.download_button("Download picks CSV (backup)", data=pd.DataFrame(picks).to_csv(index=False).encode("utf-8"),
                   file_name="nhl_picks.csv", mime="text/csv")
