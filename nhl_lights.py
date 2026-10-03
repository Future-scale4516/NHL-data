"""
nhl_lights.py
Traffic lights: when to distrust an apparent edge. No Streamlit imports.

A light is a SANITY CHECK, not a prediction. Green = "passes every check we know how to make", not
"will win". Almost everything the model flags is noise, so the rules mostly decide when NOT to trust it.
Every threshold below comes from the 2025-26 backtest (1,217 games) and one week of real closing prices:

  edge_min / edge_green_max / edge_red   scaled to the observed noise of model-minus-market disagreement:
        moneyline SD 4.1pp, totals SD 5.5pp  ->  min ~0.75 SD, green ceiling ~1.5 SD, red ~2 SD.
        Past ~2 SD the gap is more likely a model blind spot (goalie, injury) than value.
  q_lo   floor on the less-likely side's probability. Below it the model is more extreme than anything it
        was calibrated on (puck line: where it said <=22%, only 11% happened).
  EARLY_SEASON_GP   the model beat the base rate only in the second half of last season (Brier vs naive
        -0.0033 vs -0.0002), so before ~game 40 nothing is better than amber.
  consensus   an edge over a single book's price can't be checked against the market -> amber at best.

Thresholds are provisional for the puck line and shots-on-goal (little or no market data yet). The Results
and CLV pages break outcomes down by light so these can be revisited with evidence.
"""

import pandas as pd

GREEN, AMBER, RED, NONE = "🟢", "🟡", "🔴", "⚪"

LIGHT_CONFIG = {
    "Moneyline":     {"edge_min": 3.0, "edge_green_max": 6.0, "edge_red": 8.0,  "q_lo": 0.36},
    "Puck line":     {"edge_min": 3.0, "edge_green_max": 6.0, "edge_red": 8.0,  "q_lo": 0.22},
    "Total":         {"edge_min": 4.0, "edge_green_max": 8.0, "edge_red": 11.0, "q_lo": 0.33},
    "Shots on goal": {"edge_min": 4.0, "edge_green_max": 8.0, "edge_red": 12.0, "q_lo": 0.08},
}
EARLY_SEASON_GP = 40
MIN_CONSENSUS_BOOKS = 2
ORDER = {GREEN: 0, AMBER: 1, RED: 2, NONE: 3}


def assign_light(market: str, model_p: float, edge_pp: float, books: int = None, min_gp: int = None):
    """
    model_p: probability of THIS selection (0-1). edge_pp: model% minus the consensus % (or the best
    price's implied % if no consensus exists). books: number of books behind the consensus
    (None/<2 = no consensus). min_gp: fewer games than this for either team = early season.
    Returns (light, reason).
    """
    cfg = LIGHT_CONFIG[market]
    if edge_pp is None or edge_pp < cfg["edge_min"]:
        return NONE, f"no edge (under {cfg['edge_min']:g}pp)"
    q = min(model_p, 1 - model_p)
    reds = []
    if q < cfg["q_lo"]:
        reds.append(f"model gives the less likely side only {q:.0%}, below the {cfg['q_lo']:.0%} floor "
                    "it was calibrated on")
    if edge_pp >= cfg["edge_red"]:
        reds.append(f"edge {edge_pp:.1f}pp is over {cfg['edge_red']:g}pp: more likely a model blind spot than value")
    if reds:
        return RED, "; ".join(reds)
    ambers = []
    if edge_pp > cfg["edge_green_max"]:
        ambers.append(f"edge {edge_pp:.1f}pp is above the {cfg['edge_green_max']:g}pp range")
    if books is None or books < MIN_CONSENSUS_BOOKS:
        ambers.append("no multi-book consensus to check against")
    if min_gp is not None and min_gp < EARLY_SEASON_GP:
        ambers.append(f"early season ({min_gp} GP): model not yet shown to beat the base rate")
    if ambers:
        return AMBER, "; ".join(ambers)
    return GREEN, "passes all checks"


def add_lights(df: pd.DataFrame, p_col="Model %", edge_col="Edge vs consensus (pp)", books_col="Close books",
               gp_col=None, market_col="Market") -> pd.DataFrame:
    """Adds Light and Why columns to a frame of selections (Model % is in percent)."""
    out = df.copy()
    res = [assign_light(r[market_col], r[p_col] / 100,
                        r[edge_col] if edge_col in out.columns else None,
                        r[books_col] if books_col in out.columns and pd.notna(r[books_col]) else None,
                        r[gp_col] if gp_col and gp_col in out.columns and pd.notna(r[gp_col]) else None)
           for _, r in out.iterrows()]
    out["Light"] = [x[0] for x in res]
    out["Why"] = [x[1] for x in res]
    return out


def lights_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Outcomes by light, so thresholds can be judged on evidence. Uses whichever columns exist."""
    rows = []
    for light in (GREEN, AMBER, RED, NONE):
        s = df[df["Light"] == light]
        if s.empty:
            continue
        row = {"Light": light, "n": len(s), "Avg model %": round(s["Model %"].mean(), 1)}
        if "Hit" in s and "Push" in s:
            s2 = s[~s["Push"]]
            row["Hit %"] = round(s2["Hit"].mean() * 100, 1) if len(s2) else float("nan")
        if "P/L (1u)" in s:
            row["P/L (1u)"] = round(s["P/L (1u)"].sum(), 2)
            row["ROI %"] = round(s["P/L (1u)"].sum() / len(s) * 100, 1)
        if "Market move (pp)" in s:
            row["Avg market move (pp)"] = round(s["Market move (pp)"].mean(), 2)
        if "CLV (pp)" in s:
            row["Avg CLV (pp)"] = round(s["CLV (pp)"].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)
