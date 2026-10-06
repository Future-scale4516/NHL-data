"""
nhl_ui.py
Shared page look and card helpers, mirroring the MLB app: one sidebar (date + Clear Cache) on every page,
compact mobile-friendly pick cards, and a sort dropdown instead of wide tables.
"""

from datetime import date, datetime

import streamlit as st


def setup_page(title="NHL Model"):
    """Per-page config + shared styling. Must be the first Streamlit call on a page."""
    st.set_page_config(page_title=title, page_icon="🏒", layout="wide")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    section[data-testid="stSidebar"]{background:#1c1b19;}
    section[data-testid="stSidebar"] *{color:#cdccca !important;}
    </style>
    """, unsafe_allow_html=True)


def nhl_today(now=None) -> date:
    """
    Today's NHL slate date. The NHL lists games by US date, so a game that kicks off at 00:07 UK time on a Sunday
    belongs to SATURDAY's slate. US Eastern keeps the default right for a UK user at any hour: at 02:00 UK time you
    still see the games in progress, and tomorrow's slate appears at about 05:00 UK. (The server runs on UTC, so
    date.today() would flip to the next day at 01:00 UK time.)
    """
    try:
        from zoneinfo import ZoneInfo
        return (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(ZoneInfo("America/New_York")).date()
    except Exception:
        return date.today()


def sidebar_date():
    """Shared sidebar: date picker (kept across pages) + clear cache. Returns the chosen date."""
    with st.sidebar:
        st.markdown("## 🏒 NHL Model")
        sel = st.date_input("Slate Date", value=nhl_today(), key="sel_date_picker",
                            help="The NHL's own game date (US time). Games that start after midnight UK time belong to "
                                 "the previous day's slate. Kickoff times are shown in UK time.")
        if st.button("Clear Cache", key="clear_cache_btn"):
            st.cache_data.clear()
            st.rerun()
    return sel


def uk_time(iso: str) -> str:
    """'2026-10-03T23:07:00Z' -> 'Sun 4 Oct 00:07' in UK time (what bet365 shows). Falls back to UTC."""
    if not iso:
        return ""
    t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    try:
        from zoneinfo import ZoneInfo
        t = t.astimezone(ZoneInfo("Europe/London"))
        suffix = ""
    except Exception:
        suffix = " UTC"
    return f"{t.strftime('%a')} {t.day} {t.strftime('%b %H:%M')}{suffix}"


def render_pick_card(light, title, subtitle, metrics, reason=None, conditions=None):
    """One pick as a compact card: a light + title header, one dense line of metrics, optional reason below."""
    with st.container(border=True):
        st.markdown(f"{light} **{title}**" if light else f"**{title}**")
        if subtitle:
            st.caption(subtitle)
        st.caption("  ·  ".join(f"{label}: {value}" for label, value in metrics))
        if reason:
            st.caption(reason)
        if conditions:
            st.caption(f"📍 {conditions}")


def sort_picker(df, sort_options, key):
    """Selectbox for how a card list is sorted. sort_options = [(label, column, ascending), ...]; first is default."""
    labels = [lbl for lbl, _, _ in sort_options]
    choice = st.selectbox("Sort by", labels, key=key)
    _, col, asc = next(o for o in sort_options if o[0] == choice)
    return df.sort_values(col, ascending=asc, kind="stable").reset_index(drop=True)
