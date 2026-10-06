import pandas as pd
import streamlit as st
from nhl_ui import setup_page, sidebar_date, render_pick_card, sort_picker, uk_time
from common import get_api_key, cached_odds
from nhl_data import get_games, fetch_player_game_log
from nhl_backtest import season_start_year
from nhl_lights import GREEN, AMBER, RED, NONE
from nhl_props import (NB_R, fetch_stats_table, player_table, team_factors, props_selections, likely_players,
                       form_streak)
from odds import fetch_event_props, match_event, COST_LOG

setup_page("NHL Model — Player Props")
sel_date = sidebar_date()

FORM_GAMES = 5

st.title("🎰 Player Props")
st.caption("Shots on goal for now. Explore prop edges (best value against the market) and the most-likely view.")
st.divider()


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


@st.cache_data(ttl=10800, show_spinner=False)          # 3 hours, like the MLB form lookups
def cached_log(player_id, season_id):
    return fetch_player_game_log(player_id, season_id)


def load_tables(day):
    """Skater table (this + last season) and the opponent shot-suppression factors."""
    y = season_start_year(day)
    table = player_table(cached_stats("skater/summary", f"{y}{y + 1}"), cached_stats("skater/summary", f"{y - 1}{y}"))
    try:
        tfac = team_factors(cached_stats("team/summary", f"{y}{y + 1}"), cached_stats("team/summary", f"{y - 1}{y}"))
    except Exception:
        tfac = {}
    return table, tfac


def dispersion():
    bt = st.session_state.get("sog_bt")                 # fitted on the Backtest page when it has been run
    return float(bt["r_on"]) if bt else NB_R


def player_form(player_id, line):
    """(hits, played, symbols) over the last games this season, or None when the lookup fails (never breaks the list)."""
    try:
        y = season_start_year(sel_date)
        hits, played, syms = form_streak(cached_log(int(player_id), f"{y}{y + 1}"), line, FORM_GAMES)
        return (hits, played, syms) if played else None
    except Exception:
        return None


# ---------------------------------------------------------------- Player Prop Edges
st.markdown("## 🎰 Player Prop Edges")
st.caption("Pulls the US books' shots-on-goal props per game (the only source the odds provider has: it carries no UK "
           "prices for them), estimates each player's probability with the opponent's shot suppression factored in, "
           "and surfaces green/amber value. The US prices are the market reference: you can't bet there from the UK, "
           "so check the price at your own bookmaker before betting. Each game analysed costs 1 quota credit.")
pc1, pc2 = st.columns([2, 3])
with pc1:
    prop_max_games = st.slider("Games to analyse (1 credit each)", 1, 15, 8)
with pc2:
    st.caption(f"Projected cost: up to {prop_max_games} credits. Cached 15 min, so re-viewing the same games is free. "
               "The US books usually post these lines a few hours before the game.")

if st.button("Find player prop edges (US books)"):
    api_key = get_api_key()
    if not api_key:
        st.error("No Odds API key found in secrets.")
        st.stop()
    with st.spinner("Pulling props, player stats and opponent factors..."):
        try:
            games = sorted(cached_games(str(sel_date)), key=lambda g: g["start_utc"] or "")
            if not games:
                st.session_state.pop("sog_edges", None)
                st.warning(f"No games found for {sel_date}.")
            else:
                try:
                    events = cached_odds(api_key)
                except Exception as exc:
                    events = []
                    st.warning(f"Odds unavailable: {exc}")
                table, tfac = load_tables(sel_date)
                COST_LOG.clear()
                items, errs = [], []
                for g in games:
                    if len(items) >= prop_max_games:
                        break
                    ev = match_event(events, g["home"], g["away"], g["start_utc"])
                    if ev is None:
                        continue
                    try:
                        items.append((g, cached_props(api_key, ev["id"])))
                    except Exception as exc:
                        errs.append(f"{g['away']} @ {g['home']}: {exc}")
                sel, info = props_selections(items, table, tfac, r=dispersion())
                if not sel.empty:
                    sel["GameLbl"] = sel.apply(lambda r: f"{r['Game']} · {uk_time(r['Start'])}", axis=1)
                # Stored here and rendered from the persistent block below: touching a Sort dropdown or a checkbox
                # reruns the page with this button False again, so nothing nested under it would survive.
                st.session_state["sog_edges"] = {"sel": sel, "info": info, "errs": errs, "n_games": len(games),
                                                 "credits": sum(m["last"] for m in COST_LOG),
                                                 "remaining": COST_LOG[-1]["remaining"] if COST_LOG else None}
        except Exception as exc:
            st.session_state.pop("sog_edges", None)
            st.error(f"Couldn't load props: {exc}")

edges = st.session_state.get("sog_edges")
if edges:
    sel, info = edges["sel"], edges["info"]
    if edges["credits"]:
        st.caption(f"Quota: this run used ~{edges['credits']} credits, {edges['remaining']} remaining")
    if edges["errs"]:
        st.warning("Some games' props failed to load: " + "; ".join(edges["errs"])[:300])
    if sel.empty:
        st.info(f"No US shots-on-goal prices found yet for the {edges['n_games']} game(s) on this date. The US books "
                "usually post them a few hours before the game, so try again closer to the start.")
    else:
        n = sel["Light"].value_counts()
        st.markdown(f"### {GREEN} {n.get(GREEN, 0)} green · {AMBER} {n.get(AMBER, 0)} amber value props")
        show_all = st.checkbox(f"Also show {RED} suspect and {NONE} no-signal selections", value=False, key="edge_show_all")
        show_form = st.checkbox(f"Show last-{FORM_GAMES} games form (slower: one lookup per player)", key="edge_form")
        st.caption("Checks whether each player went over this pick's line in each of his last games this season. Free NHL "
                   "data, but it's one call per player: the first check each session takes a few seconds, then it's "
                   "cached for 3 hours.")
        if info["unmatched"]:
            st.caption(f"Couldn't match {len(info['unmatched'])} name(s) to the stats table: "
                       + ", ".join(info["unmatched"][:6]))
        picked = st.multiselect("Filter by game", sorted(sel["GameLbl"].unique().tolist()), default=[],
                                key="edge_game_filter", help="Leave empty to show every game.")
        view = sel[sel["GameLbl"].isin(picked)] if picked else sel
        if not show_all:
            view = view[view["Light"].isin([GREEN, AMBER])]

        def show_prop_market(tab, label):
            with tab:
                sub = view[view["Market"] == label].copy()
                if sub.empty:
                    st.write("No value bets in this market for the chosen filters.")
                    return
                sub["_edge"] = sub["Edge vs market (pp)"].fillna(sub["Edge (pp)"])
                sub = sort_picker(sub, [("Edge (high to low)", "_edge", False), ("Model % (high to low)", "Model %", False),
                                        ("Odds (high to low)", "Odds", False), ("Start time", "Start", True)],
                                  key=f"sort_prop_{label}")
                for _, r in sub.iterrows():
                    mkt = f"{r['Market %']:.1f}%" if pd.notna(r["Market %"]) else "n/a (1 book)"
                    metrics = [("Model %", f"{r['Model %']:.1f}%"), ("Market %", mkt),
                               ("Edge", f"{r['_edge']:+.1f} pts"), ("Best US price", f"{r['Odds']:.2f}")]
                    detail = f"Model expects {r['Model mean']:.1f} shots" + (
                        f" · {r['Why']}" if r["Light"] in (AMBER, RED) else "")
                    if show_form:
                        f = player_form(r["Key"].split(":")[1], r["Line"])
                        if f:
                            metrics.append(("Form", f"{f[0]}/{f[1]} over"))
                            detail = f"{f[2]}  —  {detail}"
                    render_pick_card(r["Light"], r["Selection"], r["GameLbl"], metrics, reason=detail,
                                     conditions=f"{r['Team']} · {int(r['Books'])} US book(s) in the consensus")

        (sog_tab,) = st.tabs(["🏒 Shots on goal"])
        show_prop_market(sog_tab, "Shots on goal")
        st.caption(f"{GREEN} edge 4–8 pts over the market · {AMBER} above that, or only one book to check against. "
                   f"{RED} (12+) and no-signal (<4) are hidden unless you tick the box above. Model %: our probability · "
                   "Market %: the US books' de-vigged consensus · Best US price: best decimal price across US books. "
                   "The model has not been compared with real prices: paper-trade until it has.")

# ---------------------------------------------------------------- Most Likely
st.markdown("## 🔮 Most Likely: players expected to take the most shots")
st.caption("Ranks skaters by the model's expected shots and their chance of 2+, 3+ and 4+, using this season's rate blended "
           "with last season's, recent ice time and how many shots tonight's opponent allows. This is the 'most likely' "
           "lens (it ignores odds): pair it with Player Prop Edges, the 'best value' lens. Free: no odds or quota used. "
           "The model doesn't know who is injured or scratched, so check the lineups.")
if st.button("Rank most likely players"):
    with st.spinner("Reading the slate, player stats and opponents..."):
        try:
            games = cached_games(str(sel_date))
            if not games:
                st.session_state.pop("sog_likely", None)
                st.warning(f"No games found for {sel_date}.")
            else:
                table, tfac = load_tables(sel_date)
                ml = likely_players(games, table, tfac, r=dispersion())
                if not ml.empty:
                    ml["GameLbl"] = ml.apply(lambda r: f"{r['Game']} · {uk_time(r['Start'])}", axis=1)
                st.session_state["sog_likely"] = {"df": ml, "tfac": bool(tfac)}
        except Exception as exc:
            st.session_state.pop("sog_likely", None)
            st.error(f"Couldn't rank players: {exc}")

likely = st.session_state.get("sog_likely")
if likely:
    ml = likely["df"]
    if not likely["tfac"]:
        st.caption("Opponent shot-suppression data unavailable, so no opponent adjustment was applied.")
    if ml.empty:
        st.warning("No players found for these games.")
    else:
        show_form_ml = st.checkbox(f"Show last-{FORM_GAMES} games form (slower: one lookup per player)", key="ml_form")
        st.caption(f"Checks whether each player took 3+ shots in each of his last {FORM_GAMES} games this season. Free NHL "
                   "data, but it's one call per player: the first check each session takes a few seconds, then it's "
                   "cached for 3 hours.")
        ml_picked = st.multiselect("Filter by game", sorted(ml["GameLbl"].unique().tolist()), default=[],
                                   key="ml_game_filter", help="Leave empty to show every game.")
        if ml_picked:
            ml = ml[ml["GameLbl"].isin(ml_picked)]

        def show_ml(tab):
            with tab:
                sub = sort_picker(ml.copy(), [("Expected shots (high to low)", "Exp shots", False),
                                              ("3+ shots % (high to low)", "3+ %", False),
                                              ("Ice time (high to low)", "Exp TOI", False)], key="sort_ml_sog")
                for _, r in sub.head(40).iterrows():
                    metrics = [("Exp shots", f"{r['Exp shots']:.2f}"), ("2+", f"{r['2+ %']:.0f}%"),
                               ("3+", f"{r['3+ %']:.0f}%"), ("4+", f"{r['4+ %']:.0f}%")]
                    if show_form_ml:
                        f = player_form(r["Player ID"], 2.5)
                        if f:
                            metrics.append(("Form (3+)", f"{f[0]}/{f[1]}  {f[2]}"))
                    season = (f"Season {r['Season shots/gp']:.1f}/gp ({int(r['GP this season'])} gp)"
                              if pd.notna(r["Season shots/gp"]) else "No games yet this season")
                    last = f" · last season {r['Last season shots/gp']:.1f}/gp" if pd.notna(r["Last season shots/gp"]) else ""
                    opp = f" · {r['Opp']} allow {(r['Opp factor'] - 1) * 100:+.0f}% shots vs league" if r["Opp factor"] != 1 else ""
                    render_pick_card(None, r["Player"], f"{r['GameLbl']} · {r['Pos']} · ~{r['Exp TOI']:.0f} min",
                                     metrics, conditions=season + last + opp)

        (ml_tab,) = st.tabs(["🏒 Shots on goal"])
        show_ml(ml_tab)
        st.caption("Most likely is not the same as best bet: a player can be very likely yet fairly priced (no value). "
                   "Cross-reference with Player Prop Edges. Expected shots is the model's mean; the 2+/3+/4+ chances come "
                   "from it with a negative binomial (shots are more variable than a Poisson). Players with no games yet "
                   "this season, or under about 11 minutes a night, are left out.")
