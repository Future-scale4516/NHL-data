"""
nhl_goalies.py
Goalie value test: does knowing the starting goalie improve the model? No Streamlit imports.

Method
  1. Pull every box score for the season -> one row per goalie per game (who started, shots, saves).
  2. As of each date (using ONLY earlier dates), rate each goalie by save% regressed toward the league
     average, and compare him with his own team's pooled goalie save%: that pooled figure is what the
     team's goals-against rating already assumes, so only the DIFFERENCE is new information.
        ratio = (1 - goalie_sv) / (1 - team_sv)     >1 = worse than the team's usual goalie
  3. Scale the opponent's expected goals by 1 + weight * (ratio - 1), using the goalie who ACTUALLY
     started. That is the best case (perfect foresight of the starter): if it doesn't help here, a live
     starter feed won't either.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from nhl_data import get_boxscore, get_completed_games_range
import nhl_backtest as nb

SV_PRIOR_SHOTS = 1200          # regression strength: ~40 games of shots toward the league average
MULT_CLAMP = (0.8, 1.25)       # never move expected goals more than this for a goalie
LEAGUE_SV_DEFAULT = 0.905
MIN_TEAM_STARTS_FOR_BACKUP = 5 # don't call anyone a "backup" in a team's first few games
_BOX_CACHE = {}                # game_id -> goalie rows. Finished games never change.


# ---------- parsing ----------
def _toi_seconds(toi) -> int:
    try:
        m, s = str(toi).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return 0


def parse_goalies(payload: dict) -> list[dict]:
    """Goalies who actually played, per team. Starter = explicit flag, else the goalie with most minutes."""
    out = []
    stats = payload.get("playerByGameStats") or {}
    for side in ("homeTeam", "awayTeam"):
        team = (payload.get(side) or {}).get("abbrev")
        rows = []
        for gl in (stats.get(side) or {}).get("goalies") or []:
            shots, saves = gl.get("shotsAgainst"), gl.get("saves")
            if shots is None or saves is None:
                ss = gl.get("saveShotsAgainst")                     # "saves/shots", e.g. "21/22"
                if isinstance(ss, str) and "/" in ss:
                    saves, shots = (int(x) for x in ss.split("/"))
            toi = _toi_seconds(gl.get("toi"))
            if shots is None or (not toi and not shots):             # dressed but never played
                continue
            rows.append({"team": team, "side": "home" if side == "homeTeam" else "away",
                         "goalie_id": gl.get("playerId"), "name": (gl.get("name") or {}).get("default"),
                         "starter": gl.get("starter"), "toi": toi, "shots": int(shots), "saves": int(saves or 0)})
        if rows and not any(r["starter"] for r in rows):
            max(rows, key=lambda r: r["toi"])["starter"] = True
        for r in rows:
            r["starter"] = bool(r["starter"])
        out += rows
    return out


def parse_skaters(payload: dict) -> list[dict]:
    """Skaters who played (TOI > 0): shots on goal and ice time, tagged forward/defence."""
    out = []
    stats = payload.get("playerByGameStats") or {}
    for side, other in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
        team, opp = (payload.get(side) or {}).get("abbrev"), (payload.get(other) or {}).get("abbrev")
        for grp in ("forwards", "defense"):
            for pl in (stats.get(side) or {}).get(grp) or []:
                toi = _toi_seconds(pl.get("toi"))
                if not toi:
                    continue
                out.append({"team": team, "opp": opp, "player_id": pl.get("playerId"),
                            "name": (pl.get("name") or {}).get("default"), "pos": "D" if grp == "defense" else "F",
                            "toi": toi, "sog": int(pl.get("sog") or 0)})
    return out


# ---------- fetching ----------
def fetch_boxscores(games: list[dict], progress=None):
    """
    One box score per game -> (goalie rows, skater rows, failed {game_id: error}). Cached per game, so a
    re-run only retries failures, and the goalie test and the shots-on-goal backtest share the same fetch.
    Modest parallelism + a slow second pass, because the NHL API rate-limits bursts (429).
    """
    date_of = {g["id"]: g["date"] for g in games}
    ids = list(date_of)

    def one(gid, tries=4):
        if gid in _BOX_CACHE:
            return gid, _BOX_CACHE[gid], None
        try:
            payload = get_boxscore(gid, tries=tries)
        except Exception as e:
            return gid, None, f"{type(e).__name__}: {e}"
        rows = parse_goalies(payload)
        if not rows:
            pbs = payload.get("playerByGameStats") or {}
            return gid, None, (f"no goalie rows parsed; top-level keys {list(payload)[:12]}; "
                               f"playerByGameStats keys {list(pbs)[:6]}")
        _BOX_CACHE[gid] = {"goalies": rows, "skaters": parse_skaters(payload)}
        return gid, _BOX_CACHE[gid], None

    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (gid, rec, err) in enumerate(ex.map(one, ids)):
            if err:
                errors[gid] = err
            else:
                results[gid] = rec
            if progress:
                progress(0.85 * (i + 1) / len(ids))
    retry = [g for g in ids if g not in results]
    if retry:
        time.sleep(3)
        for i, gid in enumerate(retry):
            if i:
                time.sleep(3)
            _, rec, err = one(gid, tries=6)
            if err:
                errors[gid] = err
            else:
                results[gid] = rec
                errors.pop(gid, None)
            if progress:
                progress(0.85 + 0.15 * (i + 1) / len(retry))

    goalies = [dict(r, game_id=gid, date=date_of[gid]) for gid, rec in results.items() for r in rec["goalies"]]
    skaters = [dict(r, game_id=gid, date=date_of[gid]) for gid, rec in results.items() for r in rec["skaters"]]
    return pd.DataFrame(goalies), pd.DataFrame(skaters), {g: e for g, e in errors.items() if g not in results}


def fetch_goalie_log(games: list[dict], progress=None):
    """Goalie rows only. Returns (DataFrame, failed {game_id: error})."""
    goalies, _, failed = fetch_boxscores(games, progress)
    return goalies, failed


# ---------- point-in-time goalie ratings ----------
def goalie_ratios(games: list[dict], log: pd.DataFrame, prior_shots: float = SV_PRIOR_SHOTS) -> pd.DataFrame:
    """
    Per game: how the starting goalie on each side compares with his team's usual goalie, using only
    games on EARLIER dates. ratio_* = (1 - goalie_sv) / (1 - team_sv); backup_* = not the team's most-used
    starter so far. Games without goalie data are simply absent.
    """
    if log.empty:
        return pd.DataFrame(columns=["game_id", "ratio_home", "ratio_away", "backup_home", "backup_away"])
    by_game = {gid: grp for gid, grp in log.groupby("game_id")}
    goalie, team, starts = {}, {}, {}        # id -> [saves, shots]; team -> [saves, shots]; team -> {id: n}
    lg = [0, 0]
    rows = []
    for d_iso in sorted({g["date"] for g in games}):
        day = [g for g in games if g["date"] == d_iso and g["id"] in by_game]
        lg_sv = lg[0] / lg[1] if lg[1] >= 2000 else LEAGUE_SV_DEFAULT

        def sv(stat):
            return (stat[0] + prior_shots * lg_sv) / (stat[1] + prior_shots)

        for g in day:                         # rate BEFORE any of today's results are added
            grp = by_game[g["id"]]
            rec = {"game_id": g["id"]}
            for side, tm in (("home", g["home"]), ("away", g["away"])):
                st = grp[(grp["side"] == side) & grp["starter"]]
                if st.empty:
                    rec[f"ratio_{side}"], rec[f"backup_{side}"] = 1.0, False
                    continue
                gid_ = st.iloc[0]["goalie_id"]
                g_sv, t_sv = sv(goalie.get(gid_, [0, 0])), sv(team.get(tm, [0, 0]))
                rec[f"ratio_{side}"] = (1 - g_sv) / (1 - t_sv)
                tmstarts = starts.get(tm, {})
                usual = max(tmstarts, key=tmstarts.get) if tmstarts else None
                rec[f"backup_{side}"] = bool(sum(tmstarts.values()) >= MIN_TEAM_STARTS_FOR_BACKUP and gid_ != usual)
            rows.append(rec)
        for g in day:                         # now fold today's games into the state
            for _, r in by_game[g["id"]].iterrows():
                gs, ts = goalie.setdefault(r["goalie_id"], [0, 0]), team.setdefault(r["team"], [0, 0])
                gs[0] += r["saves"]; gs[1] += r["shots"]; ts[0] += r["saves"]; ts[1] += r["shots"]
                lg[0] += r["saves"]; lg[1] += r["shots"]
                if r["starter"]:
                    starts.setdefault(r["team"], {}).setdefault(r["goalie_id"], 0)
                    starts[r["team"]][r["goalie_id"]] += 1
    return pd.DataFrame(rows)


def multipliers(ratios: pd.DataFrame, weight: float) -> dict:
    """{game_id: (home_xg_mult, away_xg_mult)}. Home scores against the AWAY goalie, and vice versa."""
    lo, hi = MULT_CLAMP
    out = {}
    for r in ratios.itertuples():
        out[r.game_id] = (float(np.clip(1 + weight * (r.ratio_away - 1), lo, hi)),
                          float(np.clip(1 + weight * (r.ratio_home - 1), lo, hi)))
    return out


# ---------- the test ----------
def _per_game_brier(df: pd.DataFrame) -> np.ndarray:
    cols = [((df[p].astype(float) - df[y].astype(float)) ** 2).to_numpy() for p, y in nb.MARKETS.values()]
    return np.mean(cols, axis=0)


def goalie_test(ctx: dict, log: pd.DataFrame, weights=(0.0, 0.5, 1.0, 1.5), n_boot: int = 1500, seed: int = 0):
    """Re-run the backtest games with goalie adjustments of increasing weight (0 = off)."""
    ratios = goalie_ratios(ctx["games"] if ctx.get("all_games") is None else ctx["all_games"], log)
    frames = {}
    for w in weights:
        df, _ = nb._frame(ctx["games"], ctx["standings"], ctx["prior"], goalie_mults=multipliers(ratios, w))
        frames[w] = df
    base = frames[weights[0]]
    n = len(base)
    rng, idx = np.random.default_rng(seed), None
    b0 = _per_game_brier(base)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]

    summary = []
    for w in weights:
        s = nb.summarize_settings(frames[w])
        bw = _per_game_brier(frames[w])
        diff = bw - b0
        cis = [diff[i].mean() for i in boots] if w != weights[0] else [0.0]
        summary.append({"goalie weight": w, **{k: s[k] for k in ("ML", "Puck", "Totals", "Avg", "Slope", "1st half", "2nd half")},
                        "Avg vs off": float(diff.mean()),
                        "95% CI low": float(np.percentile(cis, 2.5)), "95% CI high": float(np.percentile(cis, 97.5))})

    # Where does it matter? Games where a backup was in net on either side vs the rest.
    flags = ratios.set_index("game_id")[["backup_home", "backup_away"]].any(axis=1)
    sub_rows = []
    for name, mask_fn in (("A backup started (either team)", lambda f: f),
                          ("Both usual starters", lambda f: ~f)):
        for w in weights:
            f = frames[w]
            m = f["game_id"].map(flags).fillna(False).astype(bool)
            m = mask_fn(m)
            if m.sum() == 0:
                continue
            sub = f[m]
            ml = ((sub["ml_home_p"] - sub["home_win"].astype(float)) ** 2).mean()
            tot = np.mean([((sub[p] - sub[y].astype(float)) ** 2).mean() for p, y in
                           (("over55_p", "over55"), ("over65_p", "over65"))])
            sub_rows.append({"Games": name, "n": int(m.sum()), "goalie weight": w,
                             "ML Brier": float(ml), "Totals Brier": float(tot)})
    cov = {"games_scored": n, "games_with_goalie_data": int(base["game_id"].isin(ratios["game_id"]).sum()),
           "backup_games": int(flags.reindex(base["game_id"]).fillna(False).sum())}
    return {"summary": pd.DataFrame(summary), "subsets": pd.DataFrame(sub_rows), "coverage": cov}


def run_goalie_test(ctx: dict, progress=None):
    """Fetch the whole season's box scores (from Oct 1, so early-window games have goalie history), then test."""
    first = min(g["date"] for g in ctx["games"]); last = max(g["date"] for g in ctx["games"])
    start = date(nb.season_start_year(date.fromisoformat(first)), 10, 1)
    all_games = get_completed_games_range(start, date.fromisoformat(last))
    log, failed = fetch_goalie_log(all_games, progress)
    if log.empty:
        sample = next(iter(failed.values()), "no data")
        raise RuntimeError(f"No goalie rows could be read from the box scores. Example: {sample[:300]}")
    ctx = dict(ctx, all_games=all_games)
    res = goalie_test(ctx, log)
    res["coverage"].update({"boxscores_fetched": len(all_games) - len(failed), "boxscores_failed": len(failed),
                            "first_error": next(iter(failed.values()), None)})
    res["log"] = log
    return res
