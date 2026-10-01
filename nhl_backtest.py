"""
nhl_backtest.py
Backtest + results logic for the NHL model (no Streamlit imports, so it moves into shared/ later).

No lookahead: each game is modelled with standings as of the DAY BEFORE it was played,
blended with the prior season exactly as the live app does.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd

import model as nhl_model
import nhl_data
from nhl_data import (get_standings, get_completed_games_range, compute_team_strengths,
                      prior_season_end_date, prior_is_valid)
from model import (expected_goals, score_matrix, moneyline_probs, puck_line_probs,
                   puck_line_probs_with_empty_net, totals_probs)
from odds import match_event, best_prices

TEST_LINES = (5.5, 6.5)   # no-push lines, so calibration is clean

# market label -> (model probability column, outcome column)
MARKETS = {
    "Moneyline (home win)": ("ml_home_p", "home_win"),
    "Puck line (home -1.5)": ("pl_home_p", "home_m15"),
    "Puck line (away -1.5)": ("pl_away_p", "away_m15"),
    "Total Over 5.5": ("over55_p", "over55"),
    "Total Over 6.5": ("over65_p", "over65"),
}


# ---------- helpers ----------
def season_start_year(d: date) -> int:
    return d.year if d.month >= 9 else d.year - 1


def prior_table(d: date) -> list[dict]:
    """Previous season's final standings for the season containing d. Raises rather than degrade."""
    teams = get_standings(prior_season_end_date(season_start_year(d)))
    if not prior_is_valid(teams):
        raise RuntimeError("Couldn't load last season's final standings (got a table that isn't a "
                           "completed season), so early-season strengths would be wrong. Try again shortly.")
    return teams


def plausible_gp(d: date, teams: list[dict]) -> bool:
    """
    Guard against the standings endpoint answering a pre-season date with LAST season's final
    table (82 GP in early October) -- that would leak a whole season into the model.
    """
    season_start = date(season_start_year(d), 10, 1)
    max_gp = max((t["games_played"] for t in teams), default=0)
    return max_gp <= (d - season_start).days + 3


def compute_model(home: str, away: str, strengths: dict) -> dict:
    hx, ax = expected_goals(home, away, strengths)
    m = score_matrix(hx, ax)
    ml = moneyline_probs(m, hx, ax)
    raw = puck_line_probs(m)
    return {
        "home_xg": hx, "away_xg": ax, "matrix": m,
        "ml_home_p": ml["home_win_prob"],
        "pl": puck_line_probs_with_empty_net(m),
        "raw_home_2": raw["home_-1.5_raw"], "raw_away_2": raw["away_-1.5_raw"],
        "home_by_1": raw["home_by_1"], "away_by_1": raw["away_by_1"],
    }


def outcome(g: dict) -> dict:
    """Real result. The shootout winner's score includes +1 that isn't a real goal for totals."""
    hg, ag = g["home_score"], g["away_score"]
    if g["last_period"] == "SO":
        if hg > ag:
            hg -= 1
        else:
            ag -= 1
    return {"home_goals": hg, "away_goals": ag, "total_goals": hg + ag,
            "margin": g["home_score"] - g["away_score"]}   # puck line settles on the FINAL score


# ---------- backtest ----------
def _fetch_asof(d_iso: str):
    """Standings as of the day before d_iso. Returns (teams, error_text)."""
    asof = (date.fromisoformat(d_iso) - timedelta(days=1)).isoformat()
    try:
        return get_standings(asof), None
    except Exception as e:                       # get_standings already retried with backoff
        return None, f"{type(e).__name__}: {e}"


def _frame(games, standings, prior_teams, **params):
    """Model + outcome rows for every game that has usable point-in-time standings."""
    strengths_by_date, guard_skipped, rows = {}, set(), []
    for g in games:
        d_iso = g["date"]
        if d_iso not in strengths_by_date:
            teams = standings.get(d_iso)
            ok = bool(teams) and plausible_gp(date.fromisoformat(d_iso), teams)
            strengths_by_date[d_iso] = compute_team_strengths(teams, prior_teams, **params) if ok else None
            if teams and not ok:
                guard_skipped.add(d_iso)
        strengths = strengths_by_date[d_iso]
        if not strengths or g["home"] not in strengths or g["away"] not in strengths:
            continue
        mdl, o = compute_model(g["home"], g["away"], strengths), outcome(g)
        pl, m = mdl["pl"], mdl["matrix"]
        rows.append({
            "date": d_iso, "home": g["home"], "away": g["away"],
            "home_xg": mdl["home_xg"], "away_xg": mdl["away_xg"],
            "home_goals": o["home_goals"], "away_goals": o["away_goals"],
            "ml_home_p": mdl["ml_home_p"], "home_win": o["margin"] > 0,
            "pl_home_p": pl["home_-1.5"], "home_m15": o["margin"] >= 2,
            "pl_away_p": pl["away_-1.5"], "away_m15": o["margin"] <= -2,
            "raw_home_2": mdl["raw_home_2"], "raw_away_2": mdl["raw_away_2"],
            "home_by_1": mdl["home_by_1"], "away_by_1": mdl["away_by_1"],
            "over55_p": totals_probs(m, 5.5)["over"], "over55": o["total_goals"] > 5.5,
            "over65_p": totals_probs(m, 6.5)["over"], "over65": o["total_goals"] > 6.5,
        })
    return pd.DataFrame(rows), guard_skipped


def build_game_frame(start: date, end: date, progress=None):
    """
    One row per completed game: model probabilities (point-in-time) + real outcomes.
    Returns (df, notes, ctx). ctx holds the fetched data so the settings sweep can re-score
    the same games without touching the API again.
    """
    prior_teams = prior_table(start)
    games = get_completed_games_range(start, end)
    dates = sorted({g["date"] for g in games})

    # Pass 1: modest parallelism (8 workers tripped the API's rate limit on a full season)
    standings, errors = {}, {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (d_iso, (teams, err)) in enumerate(zip(dates, ex.map(_fetch_asof, dates))):
            standings[d_iso] = teams
            if err:
                errors[d_iso] = err
            if progress:
                progress(0.8 * (i + 1) / max(len(dates), 1))
    # Pass 2: retry whatever failed, one at a time, after a pause
    failed = [d for d in dates if standings.get(d) is None]
    if failed:
        time.sleep(2)
        for i, d_iso in enumerate(failed):
            teams, err = _fetch_asof(d_iso)
            standings[d_iso] = teams
            if err:
                errors[d_iso] = err
            else:
                errors.pop(d_iso, None)
            if progress:
                progress(0.8 + 0.2 * (i + 1) / len(failed))

    df, guard_skipped = _frame(games, standings, prior_teams)
    notes = [f"Fetched {len(games)} completed games over {len(dates)} dates; scored {len(df)}."]
    still_failed = sorted(d for d in dates if standings.get(d) is None)
    if still_failed:
        sample = errors.get(still_failed[0], "unknown error")
        lost = sum(1 for g in games if g["date"] in still_failed)
        notes.append(f"⚠ {len(still_failed)} date(s) ({lost} games) could not be fetched even after retries, so "
                     f"they are MISSING from this backtest (first: {still_failed[0]}; last: {still_failed[-1]}). "
                     f"Error: {sample[:160]}. Re-run before trusting these numbers.")
    if guard_skipped:
        notes.append(f"Skipped {len(guard_skipped)} date(s) where the standings looked like a previous "
                     "season's table (too early in the season).")
    ctx = {"games": games, "standings": standings, "prior": prior_teams}
    return df, notes, ctx


def summarize_settings(df: pd.DataFrame) -> dict:
    """Brier vs naive per market (negative = beats always-predicting-the-base-rate), plus the slope."""
    def delta(p_col, y_col):
        p, y = df[p_col].astype(float), df[y_col].astype(float)
        return ((p - y) ** 2).mean() - y.mean() * (1 - y.mean())
    d = {name: delta(*cols) for name, cols in MARKETS.items()}
    xd, ad = df["home_xg"] - df["away_xg"], df["home_goals"] - df["away_goals"]
    days = sorted(df["date"].unique())
    mid = days[len(days) // 2]                      # split by calendar, so both halves span many dates
    halves = [h for h in (df[df["date"] < mid], df[df["date"] >= mid]) if len(h)]
    half_avg = [sum(((h[p].astype(float) - h[y].astype(float)) ** 2).mean() - h[y].mean() * (1 - h[y].mean())
                    for p, y in MARKETS.values()) / len(MARKETS) for h in halves]
    return {"ML": d["Moneyline (home win)"],
            "Puck": (d["Puck line (home -1.5)"] + d["Puck line (away -1.5)"]) / 2,
            "Totals": (d["Total Over 5.5"] + d["Total Over 6.5"]) / 2,
            "Avg": sum(d.values()) / len(d), "Slope": xd.cov(ad) / xd.var(),
            "1st half": half_avg[0] if half_avg else float("nan"),
            "2nd half": half_avg[1] if len(half_avg) > 1 else float("nan")}


def sweep_settings(ctx: dict, ks=(20, 40, 70, 100), regressions=(0.25, 0.5), progress=None) -> pd.DataFrame:
    """Re-score the SAME fetched games under different prior-weight / regression settings."""
    combos = [(k, r) for k in ks for r in regressions]
    cur = (nhl_data.PRIOR_WEIGHT_GAMES, nhl_data.PRIOR_REGRESSION)
    if cur not in combos:
        combos.append(cur)
    rows = []
    for n, (k, r) in enumerate(combos):
        df, _ = _frame(ctx["games"], ctx["standings"], ctx["prior"], k=k, regression=r)
        if df.empty:
            continue
        rows.append({"PRIOR_WEIGHT_GAMES": k, "PRIOR_REGRESSION": r,
                     "current?": "<- current" if (k, r) == cur else "", **summarize_settings(df)})
        if progress:
            progress((n + 1) / len(combos))
    return pd.DataFrame(rows).sort_values("Avg").reset_index(drop=True)


def calibration(df: pd.DataFrame, p_col: str, y_col: str):
    if df.empty:
        return None
    p, y = df[p_col].astype(float), df[y_col].astype(float)
    base = y.mean()
    band = pd.cut(p, bins=[i / 10 for i in range(11)], include_lowest=True)
    buckets = []
    for b, grp in df.groupby(band, observed=True):
        if len(grp) < 10:          # too thin to read
            continue
        buckets.append((f"{b.left:.0%}-{b.right:.0%}", len(grp),
                        round(p[grp.index].mean() * 100, 1), round(y[grp.index].mean() * 100, 1)))
    return {"brier": round(((p - y) ** 2).mean(), 4),
            "brier_naive": round(base * (1 - base), 4),   # always predicting the base rate
            "acc": round(((p >= 0.5) == (y == 1)).mean() * 100, 1),
            "base_rate": round(base * 100, 1), "mean_p": round(p.mean() * 100, 1),
            "buckets": buckets}


def tuning_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Model vs reality on the quantities the model's tunable constants control."""
    mt, at = (df["home_xg"] + df["away_xg"]).mean(), (df["home_goals"] + df["away_goals"]).mean()
    mh, ah, ma, aa = df["home_xg"].mean(), df["home_goals"].mean(), df["away_xg"].mean(), df["away_goals"].mean()
    rows = [
        ("Avg total goals", f"{mt:.2f}", f"{at:.2f}",
         f"GOALS_SCALE {nhl_model.GOALS_SCALE:.2f} -> {nhl_model.GOALS_SCALE * at / mt:.3f}"),
        ("Avg home goals", f"{mh:.2f}", f"{ah:.2f}", ""),
        ("Avg away goals", f"{ma:.2f}", f"{aa:.2f}", ""),
        ("Home/away goals ratio", f"{mh / ma:.3f}", f"{ah / aa:.3f}",
         f"HOME_ICE_GOAL_FACTOR {nhl_model.HOME_ICE_GOAL_FACTOR:.3f} -> "
         f"{nhl_model.HOME_ICE_GOAL_FACTOR * (ah / aa) / (mh / ma):.3f}"),
        ("Home win % (incl. OT/SO)", f"{df['ml_home_p'].mean() * 100:.1f}%", f"{df['home_win'].mean() * 100:.1f}%", ""),
    ]
    xd, ad = df["home_xg"] - df["away_xg"], df["home_goals"] - df["away_goals"]
    slope = xd.cov(ad) / xd.var()
    rows.append(("Goal-diff slope (1.0 = team spread right)", "1.00", f"{slope:.2f}",
                 f"STRENGTH_SHRINK {nhl_data.STRENGTH_SHRINK:.2f} -> {nhl_data.STRENGTH_SHRINK * slope:.2f}"
                 "  (noisy: expect +/-0.35 on ~650 games)"))
    by1 = (df["home_by_1"] + df["away_by_1"]).sum()
    if by1:
        actual_2 = df["home_m15"].sum() + df["away_m15"].sum()
        raw_2 = (df["raw_home_2"] + df["raw_away_2"]).sum()
        rows.append(("Empty-net rate (1-goal -> 2-goal)", f"{nhl_model.EMPTY_NET_1_GOAL_TO_2_GOAL_RATE:.2f}",
                     f"{(actual_2 - raw_2) / by1:.2f}",
                     "Fit this AFTER fixing goals scale / home ice - it absorbs their errors"))
    return pd.DataFrame(rows, columns=["Check", "Model", "Actual", "Suggested change"])


# ---------- results (per date) ----------
def selection_rows(g: dict, mdl: dict, prices=None, fixed_line=5.5) -> list[dict]:
    """
    Every selection the model prices for one finished game, settled against the real result.
    prices=None -> model-only (Totals at fixed_line); otherwise only priced selections are kept,
    with the best available odds, edge and 1u P/L.
    """
    h, a, pl = g["home"], g["away"], mdl["pl"]
    o = outcome(g)
    margin, tot = o["margin"], o["total_goals"]
    score = f"{a} {g['away_score']} - {g['home_score']} {h}" + (
        f" ({g['last_period']})" if g["last_period"] != "REG" else "")

    def offer(mkt, key):
        if not prices:
            return None
        return prices[mkt][key] if mkt != "totals" else (prices["totals"] or {}).get(key)

    p_home = mdl["ml_home_p"]
    sels = [
        ("Moneyline", f"{h} ML", "", p_home, margin > 0, False, offer("h2h", "home")),
        ("Moneyline", f"{a} ML", "", 1 - p_home, margin < 0, False, offer("h2h", "away")),
        ("Puck line", f"{h} -1.5", -1.5, pl["home_-1.5"], margin >= 2, False, offer("spreads", "home_-1.5")),
        ("Puck line", f"{a} +1.5", 1.5, pl["away_+1.5"], margin < 2, False, offer("spreads", "away_+1.5")),
        ("Puck line", f"{a} -1.5", -1.5, pl["away_-1.5"], margin <= -2, False, offer("spreads", "away_-1.5")),
        ("Puck line", f"{h} +1.5", 1.5, pl["home_+1.5"], margin > -2, False, offer("spreads", "home_+1.5")),
    ]
    line = fixed_line if prices is None else (prices["totals"]["line"] if prices and prices["totals"] else None)
    if line is not None:
        t = totals_probs(mdl["matrix"], line)
        decided = t["over"] + t["under"]          # condition on no push
        push = tot == line
        sels += [
            ("Total", f"Over {line:g}", line, t["over"] / decided, tot > line, push, offer("totals", "over")),
            ("Total", f"Under {line:g}", line, t["under"] / decided, tot < line, push, offer("totals", "under")),
        ]

    rows = []
    for market, label, ln, p, hit, push, off in sels:
        row = {"Market": market, "Selection": label, "Line": ln, "Model %": round(p * 100, 1),
               "Hit": bool(hit) and not push, "Push": bool(push), "Game": f"{a} @ {h}", "Score": score}
        if prices is not None:
            if not off:
                continue
            price, book = off
            row.update({"Odds": price, "Book": book, "Implied %": round(100 / price, 1),
                        "Edge (pp)": round(p * 100 - 100 / price, 1),
                        "P/L (1u)": 0.0 if push else (round(price - 1, 2) if hit else -1.0)})
        rows.append(row)
    return rows


def _day_setup(day: date):
    prior_teams = prior_table(day)
    games = get_completed_games_range(day, day)
    if not games:
        return None, None, "No completed regular-season games found for that date."
    teams = get_standings((day - timedelta(days=1)).isoformat())
    if not plausible_gp(day, teams):
        return None, None, "Point-in-time standings unavailable for that date (too early in the season)."
    return games, compute_team_strengths(teams, prior_teams), ""


def build_free_results(day: date):
    games, strengths, err = _day_setup(day)
    if games is None:
        return None, err
    rows = []
    for g in games:
        if g["home"] in strengths and g["away"] in strengths:
            rows += selection_rows(g, compute_model(g["home"], g["away"], strengths))
    return pd.DataFrame(rows), (f"{len(games)} completed games on {day}. Model reads use standings as of the "
                                "day before (no lookahead); totals are tested at a fixed 5.5 line.")


def pick_games(games: list[dict], max_games: int) -> list[dict]:
    return sorted(games, key=lambda g: g["start_utc"] or "")[:max_games]


def snapshot_ts(start_utc: str, lead_minutes: int = 60) -> str:
    t = datetime.fromisoformat(start_utc.replace("Z", "+00:00")) - timedelta(minutes=lead_minutes)
    t = t.replace(minute=t.minute - t.minute % 5, second=0, microsecond=0)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_priced_results(day: date, max_games: int, fetch_snapshot, lead_minutes: int = 60):
    """
    Reconstruct each game's priced selections from the odds snapshot `lead_minutes` before ITS
    OWN start. fetch_snapshot(ts) -> list of events (caching lives in the page).
    """
    games, strengths, err = _day_setup(day)
    if games is None:
        return None, err
    chosen, rows, no_odds = pick_games(games, max_games), [], 0
    for g in chosen:
        if not g["start_utc"] or g["home"] not in strengths or g["away"] not in strengths:
            continue
        event = match_event(fetch_snapshot(snapshot_ts(g["start_utc"], lead_minutes)),
                            g["home"], g["away"], g["start_utc"])
        if not event:
            no_odds += 1
            continue
        rows += selection_rows(g, compute_model(g["home"], g["away"], strengths), prices=best_prices(event))
    note = (f"{len(chosen)} game(s) reconstructed on {day}, priced from the snapshot {lead_minutes} min "
            "before each game's own start.")
    if no_odds:
        note += f" {no_odds} game(s) had no odds in the snapshot."
    return pd.DataFrame(rows), note
