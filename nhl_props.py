"""
nhl_props.py
Shots-on-goal props: an as-of model, a free calibration backtest, and live pricing. No Streamlit.

Model, per player-game
    shots per minute (season to date, regressed toward the position average)
  x expected ice time (exponentially weighted recent games)
  x how many shots the opponent allows (as-of, regressed toward the league average)
  -> mean mu, then a negative binomial (shots are over-dispersed vs Poisson) for P(over a line).
Everything is as-of: a game only ever sees EARLIER dates.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from nhl_data import _get_json
from odds import ABBREV_TO_NAME

NHL_STATS_BASE = "https://api.nhle.com/stats/rest/en"

NB_R = 12.0                      # dispersion: var = mu + mu^2 / r. Refit from the backtest (page shows the value)
OPP_WEIGHT = 1.0                 # exponent on the opponent shot-suppression factor (1 = full, 0 = off)
PRIOR_MINUTES = 400.0            # shots-per-minute prior worth ~20 games of ice time
PRIOR_TEAM_GAMES = 15.0          # team shots-allowed prior worth 15 games
TOI_SPAN = 10                    # EWM span (games) for expected ice time
LEAGUE_SPM = {"F": 0.105, "D": 0.070}   # fallbacks before any data (shots per minute)
LEAGUE_TOI = {"F": 15.5, "D": 19.5}     # fallback minutes
LINES = (0.5, 1.5, 2.5, 3.5)


# ---------- distribution ----------
def p_over(mu, line: float, r: float = NB_R):
    """P(shots > line) for a negative binomial with mean mu. Whole-number lines condition on no push."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    prob = r / (r + mu)
    if float(line).is_integer():
        over, under = nbinom.sf(line, r, prob), nbinom.cdf(line - 1, r, prob)
        return over / (over + under)
    return nbinom.sf(np.floor(line), r, prob)


def fit_dispersion(df: pd.DataFrame, lo: float = 2.0, hi: float = 200.0) -> float:
    """Method of moments: Var(X - mu) = mu + mu^2 / r."""
    num = float((df["mu"] ** 2).sum())
    den = float((((df["sog"] - df["mu"]) ** 2) - df["mu"]).sum())
    return float(np.clip(num / den, lo, hi)) if den > 0 else hi


# ---------- as-of model (backtest) ----------
def as_of_model(skl: pd.DataFrame, opp_weight: float = OPP_WEIGHT) -> pd.DataFrame:
    """Adds mu to every skater-game row using ONLY earlier dates. skl needs date, game_id, player_id, team,
    opp, pos (F/D), toi (seconds), sog."""
    d = skl.sort_values(["date", "game_id"]).reset_index(drop=True).copy()
    d["min"] = d["toi"] / 60.0
    g = d.groupby("player_id")
    d["prev_games"] = g.cumcount()
    d["prev_shots"] = g["sog"].cumsum() - d["sog"]
    d["prev_min"] = g["min"].cumsum() - d["min"]

    # league shots/minute and minutes per game by position, through the PREVIOUS date only
    daily = d.groupby(["pos", "date"]).agg(s=("sog", "sum"), m=("min", "sum"), n=("sog", "size")).reset_index()
    daily = daily.sort_values(["pos", "date"])
    for c in ("s", "m", "n"):
        daily[f"c{c}"] = daily.groupby("pos")[c].cumsum() - daily[c]
    daily["lg_spm"] = np.where(daily["cm"] > 0, daily["cs"] / daily["cm"].replace(0, np.nan), np.nan)
    daily["lg_toi"] = np.where(daily["cn"] > 0, daily["cm"] / daily["cn"].replace(0, np.nan), np.nan)
    d = d.merge(daily[["pos", "date", "lg_spm", "lg_toi"]], on=["pos", "date"], how="left")
    d["lg_spm"] = d["lg_spm"].fillna(d["pos"].map(LEAGUE_SPM))
    d["lg_toi"] = d["lg_toi"].fillna(d["pos"].map(LEAGUE_TOI))

    spm = (d["prev_shots"] + PRIOR_MINUTES * d["lg_spm"]) / (d["prev_min"] + PRIOR_MINUTES)
    ewm = d.groupby("player_id")["min"].transform(lambda s: s.shift().ewm(span=TOI_SPAN, min_periods=1).mean())
    w = np.minimum(d["prev_games"], 5) / 5.0                      # trust a player's own ice time after ~5 games
    toi_exp = w * ewm.fillna(d["lg_toi"]) + (1 - w) * d["lg_toi"]

    # opponent shot suppression, as-of
    tg = d.groupby(["game_id", "team"], as_index=False).agg(date=("date", "first"), opp=("opp", "first"), sf=("sog", "sum"))
    tg = tg.merge(tg[["game_id", "team", "sf"]].rename(columns={"team": "opp", "sf": "sa"}), on=["game_id", "opp"])
    tg = tg.sort_values(["date", "game_id"]).reset_index(drop=True)
    gt = tg.groupby("team")
    tg["prev_g"], tg["prev_sa"] = gt.cumcount(), gt["sa"].cumsum() - tg["sa"]
    dl = tg.groupby("date").agg(sf=("sf", "sum"), n=("sf", "size")).sort_index()
    dl["lg"] = (dl["sf"].cumsum() - dl["sf"]) / (dl["n"].cumsum() - dl["n"]).replace(0, np.nan)
    tg["lg_sf"] = tg["date"].map(dl["lg"]).fillna(30.0)
    tg["opp_factor"] = ((tg["prev_sa"] + PRIOR_TEAM_GAMES * tg["lg_sf"]) / (tg["prev_g"] + PRIOR_TEAM_GAMES)) / tg["lg_sf"]
    d = d.merge(tg[["game_id", "team", "opp_factor"]].rename(columns={"team": "opp"}), on=["game_id", "opp"], how="left")
    d["opp_factor"] = d["opp_factor"].fillna(1.0)

    d["mu"] = spm * toi_exp * d["opp_factor"] ** opp_weight
    return d


# ---------- calibration backtest ----------
def _brier_rows(df, line, p_col):
    y = (df["sog"] > line).astype(float)
    return (df[p_col] - y) ** 2, y


def sog_backtest(skl: pd.DataFrame, lines=LINES, n_boot: int = 800, seed: int = 0, min_games: int = 0):
    """
    Calibration of P(over line) against real box scores, with and without the opponent factor.
    Naive benchmark = always predicting the base rate of that line (per position).
    """
    base0, base1 = as_of_model(skl, 0.0), as_of_model(skl, 1.0)
    r0, r1 = fit_dispersion(base0), fit_dispersion(base1)
    if min_games:
        keep = base1["prev_games"] >= min_games
        base0, base1 = base0[keep], base1[keep]
    rng, rows, compare = np.random.default_rng(seed), [], []
    mid = sorted(base1["date"].unique())[len(base1["date"].unique()) // 2]
    for line in lines:
        for variant, frame, r in (("opponent factor off", base0, r0), ("opponent factor on", base1, r1)):
            f = frame.assign(p=p_over(frame["mu"], line, r))
            for pos in ("All", "F", "D"):
                s = f if pos == "All" else f[f["pos"] == pos]
                y = (s["sog"] > line).astype(float)
                if len(s) < 200 or y.nunique() < 2:
                    continue
                naive = np.where(s["pos"].eq("F"), (y[s["pos"] == "F"].mean() if (s["pos"] == "F").any() else y.mean()),
                                 (y[s["pos"] == "D"].mean() if (s["pos"] == "D").any() else y.mean()))
                b_m, b_n = ((s["p"] - y) ** 2).mean(), ((naive - y) ** 2).mean()
                late = s["date"] >= mid
                rows.append({"line": line, "variant": variant, "pos": pos, "n": len(s), "base rate": round(y.mean(), 3),
                             "model avg": round(s["p"].mean(), 3), "Brier": round(b_m, 4), "naive": round(b_n, 4),
                             "vs naive": round(b_m - b_n, 4),
                             "1st half": round(((s["p"] - y)[~late] ** 2).mean() - ((naive - y)[~late] ** 2).mean(), 4),
                             "2nd half": round(((s["p"] - y)[late] ** 2).mean() - ((naive - y)[late] ** 2).mean(), 4)})
        # paired bootstrap over games: does the opponent factor help?
        y = (base1["sog"] > line).astype(float)
        d = ((p_over(base1["mu"], line, r1) - y) ** 2 - (p_over(base0["mu"], line, r0) - y) ** 2)
        per = d.groupby(base1["game_id"]).agg(["sum", "size"])
        sums, ns = per["sum"].to_numpy(), per["size"].to_numpy()
        boots = [sums[i].sum() / ns[i].sum() for i in (rng.integers(0, len(per), len(per)) for _ in range(n_boot))]
        compare.append({"line": line, "Brier change from opp factor": round(d.mean(), 5),
                        "95% CI low": round(np.percentile(boots, 2.5), 5), "95% CI high": round(np.percentile(boots, 97.5), 5)})
    return {"r_off": r0, "r_on": r1, "summary": pd.DataFrame(rows), "opp_compare": pd.DataFrame(compare),
            "frame": base1.assign(p25=p_over(base1["mu"], 2.5, r1)), "n_rows": len(base1)}


# ---------- live: stats-API tables ----------
def fetch_stats_table(report: str, season_id: str) -> list[dict]:
    """e.g. report='skater/summary' or 'team/summary'; one call returns every row."""
    return _get_json(f"{NHL_STATS_BASE}/{report}?limit=-1&cayenneExp=seasonId={season_id}%20and%20gameTypeId=2",
                     timeout=30).get("data", [])


def norm_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(jr|sr|ii|iii)\b\.?", "", n)
    return re.sub(r"[^a-z ]", "", n.replace("-", " ")).strip()


def player_table(cur_rows: list[dict], prior_rows: list[dict]) -> pd.DataFrame:
    """One row per skater: current + prior season games, shots, minutes. Fails loudly on an unexpected format."""
    need = {"playerId", "skaterFullName", "gamesPlayed", "shots", "timeOnIcePerGame"}
    for rows, label in ((cur_rows, "current"), (prior_rows, "prior")):
        if rows and not need <= set(rows[0]):
            raise RuntimeError(f"Unexpected skater stats format ({label} season): missing {sorted(need - set(rows[0]))}; "
                               f"keys seen: {sorted(rows[0])[:20]}")

    def frame(rows, suffix):
        out = pd.DataFrame([{"player_id": r["playerId"], "name": r["skaterFullName"],
                             "team": str(r.get("teamAbbrevs", "")).split(",")[-1].strip(),
                             "pos": "D" if r.get("positionCode") == "D" else "F",
                             f"gp_{suffix}": r["gamesPlayed"], f"shots_{suffix}": r["shots"],
                             f"min_{suffix}": r["gamesPlayed"] * r["timeOnIcePerGame"] / 60.0} for r in rows])
        return out

    cur, pri = frame(cur_rows, "cur"), frame(prior_rows, "pri")
    if cur.empty and pri.empty:
        raise RuntimeError("No skater stats returned for either season.")
    t = pri.merge(cur.drop(columns=["name", "pos"]), on="player_id", how="outer", suffixes=("", "_c"))
    t["team"] = t["team_c"].fillna(t["team"]) if "team_c" in t else t["team"]
    t = t.drop(columns=[c for c in t.columns if c.endswith("_c")])
    for c in ("gp_cur", "shots_cur", "min_cur", "gp_pri", "shots_pri", "min_pri"):
        t[c] = t.get(c, 0).fillna(0) if c in t else 0
    # players new this season (no prior row) still need a name/pos
    if t["name"].isna().any():
        t = t.merge(cur[["player_id", "name", "pos"]], on="player_id", how="left", suffixes=("", "_n"))
        t["name"], t["pos"] = t["name"].fillna(t["name_n"]), t["pos"].fillna(t["pos_n"])
        t = t.drop(columns=["name_n", "pos_n"])
    t["norm"] = t["name"].map(norm_name)
    return t


def live_parts(row, lg_spm: dict, opp_factor: float = 1.0, opp_weight: float = OPP_WEIGHT) -> dict:
    """Expected shots for tonight, blending this season (primary) with last season (prior), regressed.
    Returns the mean plus the pieces behind it (expected ice time in minutes, shots per minute)."""
    pos = row["pos"]
    spm_prior = (row["shots_pri"] + PRIOR_MINUTES * lg_spm[pos]) / (row["min_pri"] + PRIOR_MINUTES)
    spm = (row["shots_cur"] + PRIOR_MINUTES * 1.25 * spm_prior) / (row["min_cur"] + PRIOR_MINUTES * 1.25)
    toi_cur = row["min_cur"] / row["gp_cur"] if row["gp_cur"] else None
    toi_pri = row["min_pri"] / row["gp_pri"] if row["gp_pri"] else LEAGUE_TOI[pos]
    toi = ((row["gp_cur"] * toi_cur + 8 * toi_pri) / (row["gp_cur"] + 8)) if toi_cur else toi_pri
    return {"mu": float(spm * toi * opp_factor ** opp_weight), "toi": float(toi), "spm": float(spm)}


def live_mu(row, lg_spm: dict, opp_factor: float = 1.0, opp_weight: float = OPP_WEIGHT) -> float:
    return live_parts(row, lg_spm, opp_factor, opp_weight)["mu"]


def team_factors(cur_rows: list[dict], prior_rows: list[dict]) -> dict:
    """{team abbrev: shots-allowed factor vs league (1.0 = average)} from the stats API. {} if unavailable."""
    name_to_abbrev = {norm_name(v): k for k, v in ABBREV_TO_NAME.items()}
    def per_game(rows):
        out = {}
        for r in rows:
            ab = name_to_abbrev.get(norm_name(r.get("teamFullName", "")))
            if ab and r.get("gamesPlayed") and r.get("shotsAgainstPerGame") is not None:
                out[ab] = (r["gamesPlayed"], float(r["shotsAgainstPerGame"]))
        return out
    cur, pri = per_game(cur_rows), per_game(prior_rows)
    if not cur and not pri:
        return {}
    lg = np.mean([v[1] for v in (pri or cur).values()])
    f = {}
    for ab in set(cur) | set(pri):
        gc, sc = cur.get(ab, (0, lg)); gp, sp = pri.get(ab, (82, lg))
        sa = (gc * sc + PRIOR_TEAM_GAMES * (0.5 * sp + 0.5 * lg)) / (gc + PRIOR_TEAM_GAMES)
        f[ab] = sa / lg
    return f


def match_player(desc: str, teams: tuple, table: pd.DataFrame):
    """Odds-feed player name -> stats row, restricted to the two teams in the game. None if no safe match."""
    n = norm_name(desc)
    pool = table[table["team"].isin(teams)]
    hit = pool[pool["norm"] == n]
    if len(hit) == 1:
        return hit.iloc[0]
    parts = n.split()
    if len(parts) >= 2:                                              # "t j oshie" style / initials: last name + first initial
        last, first = parts[-1], parts[0][0]
        hit = pool[pool["norm"].map(lambda x: x.split()[-1] == last and x[0] == first if x else False)]
        if len(hit) == 1:
            return hit.iloc[0]
    return None


def league_spm(table: pd.DataFrame) -> dict:
    """League shots per minute by position, from both seasons' totals (live fallback prior)."""
    out = {}
    for pos in ("F", "D"):
        t = table[table["pos"] == pos]
        mins = t["min_cur"].sum() + t["min_pri"].sum()
        out[pos] = float((t["shots_cur"].sum() + t["shots_pri"].sum()) / mins) if mins > 0 else LEAGUE_SPM[pos]
    return out


def props_selections(items: list, table: pd.DataFrame, tfac: dict, r: float = NB_R, opp_weight: float = OPP_WEIGHT):
    """
    items: [(slate_game_row, odds_event_json_with_props), ...]. One row per player x line x side, priced against the
    best book and checked against the multi-book consensus. Returns (DataFrame, {'unmatched': [...], 'props': n}).
    """
    from nhl_lights import assign_light
    from odds import props_table

    lg = league_spm(table)
    recs, unmatched, n_props = [], set(), 0
    for g, event in items:
        pt = props_table(event)
        if pt.empty:
            continue
        for (player, line), grp in pt.groupby(["player", "line"]):
            n_props += 1
            row = match_player(player, (g["home"], g["away"]), table)
            if row is None:
                unmatched.add(player)
                continue
            opp = g["away"] if row["team"] == g["home"] else g["home"]
            mu = live_mu(row, lg, tfac.get(opp, 1.0), opp_weight)
            p_o = float(p_over(mu, line, r))
            # consensus: de-vig every book that quotes BOTH sides of this exact line
            fair_o = []
            for _, bk in grp.groupby("book"):
                ov, un = bk[bk["side"] == "over"]["price"], bk[bk["side"] == "under"]["price"]
                if len(ov) and len(un):
                    io, iu = 1 / ov.iloc[0], 1 / un.iloc[0]
                    if io + iu >= 0.97:
                        fair_o.append(io / (io + iu))
            for side, p_sel in (("over", p_o), ("under", 1 - p_o)):
                sd = grp[grp["side"] == side]
                if sd.empty:
                    continue
                best = sd.loc[sd["price"].idxmax()]
                cons = (np.mean(fair_o) if side == "over" else 1 - np.mean(fair_o)) if fair_o else None
                books = len(fair_o)
                edge_best = (p_sel - 1 / best["price"]) * 100
                edge_cons = (p_sel - cons) * 100 if cons is not None and books >= 2 else None
                light, why = assign_light("Shots on goal", p_sel, edge_cons if edge_cons is not None else edge_best,
                                          books, None)
                recs.append({
                    "Game ID": g["id"], "Game": f"{g['away']} @ {g['home']}", "Start": g["start_utc"],
                    "Player": row["name"], "Team": row["team"], "Market": "Shots on goal",
                    "Selection": f"{row['name']} {side.title()} {line:g}", "Line": line,
                    "Key": f"sog:{int(row['player_id'])}:{side}:{line:g}", "Model mean": round(mu, 2),
                    "Model %": round(p_sel * 100, 1), "Odds": float(best["price"]), "Book": best["book"],
                    "Implied %": round(100 / best["price"], 1), "Edge (pp)": round(edge_best, 1),
                    "Market %": round(cons * 100, 1) if cons is not None else None, "Books": books,
                    "Edge vs market (pp)": round(edge_cons, 1) if edge_cons is not None else None,
                    "EV %": round((p_sel * best["price"] - 1) * 100, 1), "Light": light, "Why": why})
    return pd.DataFrame(recs), {"unmatched": sorted(unmatched), "props": n_props}


def likely_players(games: list, table: pd.DataFrame, tfac: dict, r: float = NB_R, min_toi: float = 11.0) -> pd.DataFrame:
    """
    Every regular skater in tonight's games with the model's expected shots and P(2+/3+/4+). Model-only: no odds.
    The stats feed knows nothing about injuries or scratches, so players who haven't played this season (once their
    team has) are left out, along with anyone expected to play under `min_toi` minutes.
    """
    lg = league_spm(table)
    recs = []
    for g in games:
        for team, opp in ((g["home"], g["away"]), (g["away"], g["home"])):
            pool = table[table["team"] == team]
            if pool.empty:
                continue
            started = pool["gp_cur"].max() >= 1
            for _, row in pool.iterrows():
                if (started and row["gp_cur"] < 1) or (not started and row["gp_pri"] < 20):
                    continue
                parts = live_parts(row, lg, tfac.get(opp, 1.0))
                if parts["toi"] < min_toi:
                    continue
                mu = parts["mu"]
                recs.append({"Game ID": g["id"], "Game": f"{g['away']} @ {g['home']}", "Start": g["start_utc"],
                             "Player": row["name"], "Player ID": int(row["player_id"]), "Team": team, "Opp": opp,
                             "Pos": row["pos"], "Exp shots": round(mu, 2),
                             "2+ %": round(float(p_over(mu, 1.5, r)) * 100, 1),
                             "3+ %": round(float(p_over(mu, 2.5, r)) * 100, 1),
                             "4+ %": round(float(p_over(mu, 3.5, r)) * 100, 1),
                             "Exp TOI": round(parts["toi"], 1), "GP this season": int(row["gp_cur"]),
                             "Season shots/gp": round(row["shots_cur"] / row["gp_cur"], 2) if row["gp_cur"] else None,
                             "Last season shots/gp": round(row["shots_pri"] / row["gp_pri"], 2) if row["gp_pri"] else None,
                             "Opp factor": round(tfac.get(opp, 1.0), 3)})
    return pd.DataFrame(recs)


def form_streak(log: list, line: float, games: int = 5):
    """How often a player went OVER `line` shots in his most recent `games` games. log is newest-first
    [{'date','shots'}]. Returns (hits, played, symbols) or (None, 0, '') with no games."""
    recent = log[:games]
    if not recent:
        return None, 0, ""
    over = [g["shots"] > line for g in recent]
    return sum(over), len(recent), "".join("✅" if o else "❌" for o in over)
