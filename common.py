"""common.py - small helpers shared by the app pages."""
import streamlit as st
from odds import fetch_odds


def get_api_key():
    try:
        for name in ("ODDS_API_KEY", "odds_api_key", "THE_ODDS_API_KEY", "API_KEY"):
            if name in st.secrets:
                return st.secrets[name]
    except Exception:          # no secrets file configured at all: report "no key" instead of crashing
        pass
    return None


@st.cache_data(ttl=900, show_spinner=False)   # 15 min; shared by every page so odds are only paid for once
def cached_odds(api_key: str):
    return fetch_odds(api_key)
