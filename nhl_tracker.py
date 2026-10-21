"""
nhl_tracker.py
Pick tracker: save the picks you actually take, settle them against real results, and summarise.

Storage is one private GitHub Gist holding a single JSON file. Needs two optional Streamlit secrets:
  GITHUB_TOKEN  a personal access token with the "gist" scope (names tried: GITHUB_TOKEN, GH_TOKEN, github_token)
  NHL_GIST_ID   the gist id (names tried: NHL_GIST_ID, nhl_gist_id). If absent, the tracker looks for a gist
                it created earlier (description "NHL picks tracker"), and creates one on first save.
Without a token everything still works for the session, and the CSV download/upload keeps your log safe.
The token is only ever sent to api.github.com and is never printed or put in an error message.
"""

import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import requests

GIST_FILE = "nhl_picks.json"
GIST_DESC = "NHL picks tracker"
API = "https://api.github.com"
TOKEN_NAMES = ("GITHUB_TOKEN", "GH_TOKEN", "github_token")
GIST_ID_NAMES = ("NHL_GIST_ID", "nhl_gist_id")
OPEN, WON, LOST, PUSH, VOID = "open", "won", "lost", "push", "void"
FINAL = (WON, LOST, PUSH, VOID)
FIELDS = ["id", "saved_at", "game_date", "game_id", "game", "start", "market", "selection", "key", "line",
          "odds", "book", "model_pct", "market_pct", "edge_pp", "light", "units", "stake", "free_bet",
          "source", "status", "result_note", "settled_at"]


# ---------- secrets ----------
def _secret(names):
    try:
        import streamlit as st
        for n in names:
            if n in st.secrets:
                return str(st.secrets[n]).strip()
    except Exception:
        pass
    return None


def has_token() -> bool:
    return bool(_secret(TOKEN_NAMES))


def _headers():
    return {"Authorization": f"Bearer {_secret(TOKEN_NAMES)}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _fail(resp, what):
    """Error text that can never contain the token."""
    hint = {401: " (token rejected or expired)", 403: " (token lacks the 'gist' scope, or rate limited)",
            404: " (gist not found, or the token can't see it)"}.get(resp.status_code, "")
    return RuntimeError(f"GitHub {what} failed: HTTP {resp.status_code}{hint}")


# ---------- gist storage ----------
def _find_gist_id():
    gid = _secret(GIST_ID_NAMES)
    if gid:
        return gid
    page = 1
    while page <= 5:
        r = requests.get(f"{API}/gists", headers=_headers(), params={"per_page": 100, "page": page}, timeout=15)
        if r.status_code != 200:
            raise _fail(r, "listing gists")
        batch = r.json()
        for g in batch:
            if g.get("description") == GIST_DESC and GIST_FILE in (g.get("files") or {}):
                return g["id"]
        if len(batch) < 100:
            break
        page += 1
    return None


def load_picks(gist_id=None):
    """Returns (list of pick dicts, gist_id or None). Raises RuntimeError on a GitHub problem."""
    if not has_token():
        return [], None
    gid = gist_id or _find_gist_id()
    if not gid:
        return [], None
    r = requests.get(f"{API}/gists/{gid}", headers=_headers(), timeout=15)
    if r.status_code != 200:
        raise _fail(r, "reading the gist")
    f = (r.json().get("files") or {}).get(GIST_FILE)
    if not f:
        return [], gid
    content = f.get("content")
    if f.get("truncated") and f.get("raw_url"):
        content = requests.get(f["raw_url"], headers=_headers(), timeout=15).text
    try:
        data = json.loads(content or "[]")
    except json.JSONDecodeError:
        raise RuntimeError("The picks gist isn't valid JSON. Open it on GitHub and check nhl_picks.json.")
    return (data if isinstance(data, list) else []), gid


def write_picks(picks, gist_id=None):
    """Overwrites the gist file (creating a private gist if there isn't one). Returns the gist id."""
    body = {"files": {GIST_FILE: {"content": json.dumps(picks, indent=1)}}}
    if gist_id:
        r = requests.patch(f"{API}/gists/{gist_id}", headers=_headers(), json=body, timeout=20)
        if r.status_code != 200:
            raise _fail(r, "saving to the gist")
        return gist_id
    r = requests.post(f"{API}/gists", headers=_headers(), timeout=20,
                      json={**body, "description": GIST_DESC, "public": False})
    if r.status_code != 201:
        raise _fail(r, "creating the gist")
    return r.json()["id"]


def merge(existing, new):
    """Upsert by id. Existing picks win (so a settled pick is never reopened by a re-save)."""
    by_id = {p["id"]: p for p in existing}
    added = 0
    for p in new:
        if p["id"] not in by_id:
            by_id[p["id"]] = p
            added += 1
    return list(by_id.values()), added


def save_new(new, gist_id=None):
    """Load-merge-write so two devices don't overwrite each other. Returns (all picks, added, gist_id)."""
    existing, gid = load_picks(gist_id)
    merged, added = merge(existing, new)
    if added:
        gid = write_picks(merged, gid)
    return merged, added, gid


def save_all(picks, gist_id=None):
    """Replace the stored list wholesale (used after settling or deleting)."""
    return write_picks(picks, gist_id)


# ---------- building records ----------
def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def make_records(df: pd.DataFrame, source: str, game_date, free_bet: bool = False, default_stake: float = 0.0):
    """
    df: selection rows (columns as produced by nhl_live.selections / nhl_props.props_selections / pick_bets).
    One record per selection. The id is deterministic (game + selection + line), so saving twice is harmless.
    """
    out = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for _, r in df.iterrows():
        line = _clean(r.get("Line"))
        line = None if line in ("", None) else float(line)
        edge = _clean(r.get("Edge vs market (pp)"))
        if edge is None:
            edge = _clean(r.get("Edge (pp)"))
        stake = _clean(r.get("Stake"))
        out.append({
            "id": f"{int(r['Game ID'])}|{r['Key']}|{'' if line is None else f'{line:g}'}",
            "saved_at": now, "game_date": str(game_date), "game_id": int(r["Game ID"]), "game": r["Game"],
            "start": _clean(r.get("Start")), "market": r["Market"], "selection": r["Selection"], "key": r["Key"],
            "line": line, "odds": float(r["Odds"]), "book": _clean(r.get("Book")),
            "model_pct": _clean(r.get("Model %")), "market_pct": _clean(r.get("Market %")), "edge_pp": edge,
            "light": _clean(r.get("Light")), "units": _clean(r.get("Units")),
            "stake": float(stake) if stake is not None else float(default_stake),
            "free_bet": bool(free_bet), "source": source, "status": OPEN, "result_note": "", "settled_at": None})
    return out


# ---------- settling ----------
def pick_pnl(p: dict) -> float:
    """Cash P/L in £. A free bet's stake isn't returned, so a loss costs nothing and a win pays stake x (odds-1)."""
    s, o, st = p.get("status"), p.get("odds") or 0, p.get("stake") or 0
    if s == WON:
        return st * (o - 1)
    if s == LOST:
        return 0.0 if p.get("free_bet") else -st
    return 0.0


def pick_units(p: dict) -> float:
    """Flat 1-unit P/L, ignoring stake size: the cleanest read on whether the model is right."""
    s, o = p.get("status"), p.get("odds") or 0
    return {WON: o - 1, LOST: -1.0}.get(s, 0.0)


def _settle_game(p, board):
    from nhl_bets import settle
    g = board.get(p["game_id"])
    if not g:
        return None, "game not found on that date's schedule"
    res = settle(p["key"], p.get("line"), g)
    if res in ("pending", "live"):
        return None, ""
    note = f"{g['away']} {g['away_score']} - {g['home_score']} {g['home']}" + \
           (f" ({g['last_period']})" if g.get("last_period") not in (None, "REG") else "")
    return res, note


def _settle_sog(p, skaters):
    """skaters: parsed box-score rows (only players with ice time). Absent = didn't play = void."""
    _, pid, side, line = p["key"].split(":")
    row = next((s for s in skaters if str(s["player_id"]) == pid), None)
    if row is None:
        return VOID, "did not play"
    line = float(line)
    sog = row["sog"]
    if sog == line:
        return PUSH, f"{sog} shots"
    return (WON if (sog > line) == (side == "over") else LOST), f"{sog} shots"


def settle_open(picks, get_board, get_skaters):
    """
    Settles every open pick whose game has finished. get_board(date_iso) -> {game_id: game dict};
    get_skaters(game_id) -> parsed skater rows, or None if the box score isn't available yet.
    Returns (picks, n_settled, errors). Never reopens a settled pick.
    """
    boards, boxes, errors, n = {}, {}, [], 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for p in picks:
        if p.get("status") != OPEN:
            continue
        try:
            d = p["game_date"]
            if d not in boards:
                boards[d] = {g["id"]: g for g in get_board(d)}
            board = boards[d]
            if p["key"].startswith("sog:"):
                g = board.get(p["game_id"])
                if not g or g.get("state") not in ("OFF", "FINAL"):
                    continue
                if p["game_id"] not in boxes:
                    boxes[p["game_id"]] = get_skaters(p["game_id"])
                sk = boxes[p["game_id"]]
                if not sk:
                    continue
                res, note = _settle_sog(p, sk)
            else:
                res, note = _settle_game(p, board)
            if res:
                p.update(status=res, result_note=note, settled_at=now)
                n += 1
        except Exception as exc:
            errors.append(f"{p.get('selection', '?')}: {type(exc).__name__}: {exc}")
    return picks, n, errors


def live_get_board(day_iso):
    from nhl_data import get_scoreboard
    return get_scoreboard(day_iso)


def live_get_skaters(game_id):
    from nhl_data import get_boxscore
    from nhl_goalies import parse_skaters
    return parse_skaters(get_boxscore(game_id))


# ---------- summaries ----------
def to_frame(picks) -> pd.DataFrame:
    if not picks:
        return pd.DataFrame(columns=FIELDS)
    df = pd.DataFrame(picks)
    for c in FIELDS:
        if c not in df.columns:
            df[c] = None
    df["pnl_cash"] = [pick_pnl(p) for p in picks]
    df["pnl_units"] = [pick_units(p) for p in picks]
    return df


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Rows per group: picks, settled, W-L-P, hit rate, model avg, units P/L, ROI per unit, cash P/L."""
    d = df[df["status"].isin([WON, LOST, PUSH])]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for k, g in d.groupby(by, dropna=False):
        decided = g[g["status"].isin([WON, LOST])]
        w = int((g["status"] == WON).sum())
        l = int((g["status"] == LOST).sum())
        rows.append({by: k if k is not None else "?", "Settled": len(g), "W-L-P": f"{w}-{l}-{len(g) - w - l}",
                     "Hit rate %": round(w / len(decided) * 100, 1) if len(decided) else None,
                     "Model avg %": round(float(pd.to_numeric(decided["model_pct"], errors="coerce").mean()), 1)
                     if len(decided) else None,
                     "Units": round(float(g["pnl_units"].sum()), 2),
                     "ROI %": round(float(g["pnl_units"].sum()) / len(g) * 100, 1),
                     "Cash £": round(float(g["pnl_cash"].sum()), 2)})
    return pd.DataFrame(rows)


# ---------- Streamlit helpers (imported lazily so the logic above stays testable without Streamlit) ----------
def save_button(df, source: str, game_date, key: str, default_stake: float = 0.0, label: str = "💾 Save these picks"):
    """Drop-in widget: a free-bet toggle + save button. Saved picks are settled on the My Picks page."""
    import streamlit as st
    if df is None or df.empty:
        return
    free = st.checkbox("These are free bets (stake not returned)", key=f"{key}_free")
    if st.button(label, key=f"{key}_save"):
        recs = make_records(df, source, game_date, free_bet=free, default_stake=default_stake)
        if has_token():
            try:
                _, added, gid = save_new(recs, st.session_state.get("tracker_gist"))
                st.session_state["tracker_gist"] = gid
                st.session_state.pop("tracker_picks", None)
                st.success(f"Saved {added} new pick(s)" + (f" ({len(recs) - added} already saved)." if added < len(recs) else "."))
            except Exception as exc:
                st.error(f"Couldn't save to GitHub: {exc}")
        else:
            cur = st.session_state.setdefault("tracker_local", [])
            st.session_state["tracker_local"], added = merge(cur, recs)
            st.warning(f"No GitHub token in secrets, so {added} pick(s) are saved for this session only. "
                       "Add GITHUB_TOKEN to keep them, or download the CSV on the My Picks page.")


def save_picker(df, source: str, game_date, key: str):
    """Pick which of the shown selections you actually bet, set a stake, save. Used on the browse pages."""
    import streamlit as st
    if df is None or df.empty:
        return
    with st.expander("💾 Log the bets you actually place"):
        d = df.reset_index(drop=True)
        labels = [f"{r['Selection']} @ {r['Odds']:.2f} · {r['Game']}" for _, r in d.iterrows()]
        chosen = st.multiselect("Selections you backed", range(len(d)), format_func=lambda i: labels[i], key=f"{key}_sel")
        stake = st.number_input("Stake per pick (£)", min_value=0.0, value=5.0, step=1.0, key=f"{key}_stake")
        if chosen:
            save_button(d.iloc[chosen].assign(Stake=stake), source, game_date, key, default_stake=stake)
