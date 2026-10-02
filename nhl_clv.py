"""
nhl_clv.py
Closing Line Value (CLV) report for the NHL model. No Streamlit imports.

For each game: take the best price available `pick_lead` minutes before the start (what you could
have bet), and compare it with the de-vigged consensus at the close. If the model's flagged picks
systematically beat the close, the market moved toward the model: the best evidence of a real edge
that doesn't depend on thousands of noisy win/loss results.

Two controls keep this honest (both were needed: a no-edge test world gave false positives without them):
  1. Picks are FLAGGED on the model's edge over the CONSENSUS price at pick time, not over the best price.
     Flagging on the best price over-selects one book's outlier price, which reverts by the close and
     looks like skill.
  2. Results are always flagged picks vs ALL priced selections (the baseline), with a bootstrap over games,
     because best-price shopping beats the average close even for random picks.
Two measures: "market move" = how far the consensus moved toward the pick (did the market learn what the
model claims?) and "CLV" = closing fair prob minus the implied prob of the best price you could bet.
"""

from datetime import date

import numpy as np
import pandas as pd

from nhl_data import get_completed_games_range, compute_team_strengths
from nhl_backtest import (prior_table, plausible_gp, fetch_standings_for, compute_model,
                          selection_rows, snapshot_ts)
from odds import match_event, best_prices, fair_probs

CREDITS_PER_SNAPSHOT = 30      # 10 per market x 3 markets x 1 region
MIN_CLOSE_BOOKS = 2            # closing consensus needs at least this many books quoting the market


def plan_snapshots(games: list[dict], pick_lead: int, close_lead: int, max_snapshots: int):
    """
    Which games fit under the snapshot cap (chronological), and what the full range would need.
    Returns (games_covered, distinct_snapshots_needed_for_all, distinct_snapshots_used).
    """
    games = sorted((g for g in games if g.get("start_utc")), key=lambda g: g["start_utc"])
    all_ts = {t for g in games for t in (snapshot_ts(g["start_utc"], pick_lead),
                                         snapshot_ts(g["start_utc"], close_lead))}
    used, covered = set(), []
    for g in games:
        need = {snapshot_ts(g["start_utc"], pick_lead), snapshot_ts(g["start_utc"], close_lead)}
        if len(used | need) <= max_snapshots:
            used |= need
            covered.append(g)
    return covered, len(all_ts), len(used)


def build_clv_frame(start: date, end: date, fetch_snapshot, pick_lead: int = 180, close_lead: int = 5,
                    max_snapshots: int = 30, progress=None):
    """
    One row per priced selection: model %, best pick-time odds/edge, closing fair %, CLV.
    fetch_snapshot(ts) -> list of odds events (caching lives in the page).
    Returns (df, notes).
    """
    prior_teams = prior_table(start)
    games = get_completed_games_range(start, end)
    covered, n_all, n_used = plan_snapshots(games, pick_lead, close_lead, max_snapshots)
    notes = []
    if len(covered) < len([g for g in games if g.get("start_utc")]):
        first_cut = min((g["date"] for g in games if g["id"] not in {c["id"] for c in covered}), default=None)
        notes.append(f"Snapshot cap ({max_snapshots}) reached: scored {len(covered)} of {len(games)} games, "
                     f"stopping before {first_cut}. Raise the cap or shorten the date range for the rest.")

    dates = sorted({g["date"] for g in covered})
    standings, errors, failed = fetch_standings_for(dates, lambda f: progress(0.25 * f) if progress else None)
    if failed:
        notes.append(f"⚠ Standings missing for {len(failed)} date(s) ({failed[0]} ...): those games are excluded. "
                     f"Error: {errors.get(failed[0], 'unknown')[:120]}")
    strengths_by_date = {}
    for d_iso in dates:
        teams = standings.get(d_iso)
        ok = bool(teams) and plausible_gp(date.fromisoformat(d_iso), teams)
        strengths_by_date[d_iso] = compute_team_strengths(teams, prior_teams) if ok else None

    rows, no_odds, dropped, no_str = [], 0, 0, 0
    for n, g in enumerate(covered):
        strengths = strengths_by_date.get(g["date"])
        if not strengths or g["home"] not in strengths or g["away"] not in strengths:
            no_str += 1
            continue
        ev_pick = match_event(fetch_snapshot(snapshot_ts(g["start_utc"], pick_lead)), g["home"], g["away"], g["start_utc"])
        ev_close = match_event(fetch_snapshot(snapshot_ts(g["start_utc"], close_lead)), g["home"], g["away"], g["start_utc"])
        if progress:
            progress(0.25 + 0.75 * (n + 1) / len(covered))
        if not ev_pick or not ev_close:
            no_odds += 1
            continue
        prices = best_prices(ev_pick)
        line = prices["totals"]["line"] if prices["totals"] else None
        fair_close = fair_probs(ev_close, line=line)   # totals only matched at the SAME line we'd have bet
        fair_pick = fair_probs(ev_pick, line=line)
        for r in selection_rows(g, compute_model(g["home"], g["away"], strengths), prices=prices):
            fc, fp = fair_close.get(r["Key"]), fair_pick.get(r["Key"])
            if not fc or not fp or fc[1] < MIN_CLOSE_BOOKS or fp[1] < MIN_CLOSE_BOOKS:
                dropped += 1
                continue
            r.update({"Date": g["date"], "Game ID": g["id"],
                      "Min GP": min(strengths[g["home"]]["games_played"], strengths[g["away"]]["games_played"]),
                      "Pick fair %": round(fp[0] * 100, 1), "Close fair %": round(fc[0] * 100, 1),
                      "Close books": fc[1],
                      "Edge vs consensus (pp)": round(r["Model %"] - fp[0] * 100, 2),
                      "Market move (pp)": round((fc[0] - fp[0]) * 100, 2),
                      "CLV (pp)": round((fc[0] - 1 / r["Odds"]) * 100, 2),
                      "Close EV %": round((r["Odds"] * fc[0] - 1) * 100, 2)})
            rows.append(r)
    notes.append(f"{len(covered)} games planned ({n_used} snapshots); {len(set(r['Game ID'] for r in rows))} had "
                 f"usable pick + closing odds. {no_odds} had no odds in a snapshot; {dropped} selections dropped "
                 "(consensus from 2+ books missing at pick or close, or the totals line moved).")
    if no_str:
        notes.append(f"{no_str} game(s) skipped: no point-in-time standings.")
    return pd.DataFrame(rows), notes


def _vs_baseline(df, flag, col, n_boot=2000, seed=0):
    """Mean of `col` for flagged picks vs all selections, with a bootstrap CI (resampling whole games)."""
    d = df.assign(_f=flag, _fv=df[col].where(flag, 0.0))
    g = d.groupby("Game ID").agg(fs=("_fv", "sum"), fc=("_f", "sum"), as_=(col, "sum"), ac=(col, "size"))
    fs, fc, as_, ac = (g[c].to_numpy(float) for c in ("fs", "fc", "as_", "ac"))
    rng, diffs = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(g), len(g))
        if fc[i].sum() > 0:
            diffs.append(fs[i].sum() / fc[i].sum() - as_[i].sum() / ac[i].sum())
    f_mean = df.loc[flag, col].mean() if flag.any() else float("nan")
    a_mean = df[col].mean()
    return {"flagged": f_mean, "all": a_mean, "diff": f_mean - a_mean,
            "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if len(diffs) > 50 else None,
            "beat": (df.loc[flag, col] > 0).mean() * 100 if flag.any() else float("nan")}


def clv_summary(df: pd.DataFrame, min_edge: float = 3.0, model_floor: float = 0.0) -> dict:
    """Flagged picks (edge over CONSENSUS at pick time) vs the all-selections baseline."""
    flag = (df["Edge vs consensus (pp)"] >= min_edge) & (df["Model %"] >= model_floor)
    out = {"n_all": len(df), "n_flag": int(flag.sum()),
           "move": _vs_baseline(df, flag, "Market move (pp)"),
           "clv": _vs_baseline(df, flag, "CLV (pp)")}

    by_market = []
    for mk, sub in df.groupby("Market"):
        f = flag.loc[sub.index]
        by_market.append({"Market": mk, "Flagged n": int(f.sum()),
                          "Flagged move (pp)": sub.loc[f, "Market move (pp)"].mean() if f.any() else float("nan"),
                          "All move (pp)": sub["Market move (pp)"].mean(),
                          "Flagged CLV (pp)": sub.loc[f, "CLV (pp)"].mean() if f.any() else float("nan"),
                          "All CLV (pp)": sub["CLV (pp)"].mean()})
    out["by_market"] = pd.DataFrame(by_market)

    bands = pd.cut(df["Edge vs consensus (pp)"], [-100, 0, 3, 6, 100], labels=["< 0", "0 to 3", "3 to 6", "6+"], right=False)
    out["by_edge"] = (df.groupby(bands, observed=True)
                        .agg(n=("Market move (pp)", "size"), avg_move_pp=("Market move (pp)", "mean"),
                             avg_clv_pp=("CLV (pp)", "mean"),
                             pct_moved_toward=("Market move (pp)", lambda x: (x > 0).mean() * 100))
                        .reset_index().rename(columns={"Edge vs consensus (pp)": "Model edge vs consensus at pick (pp)"}))
    return out
