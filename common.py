"""common.py - small helpers shared by the app pages."""
import streamlit as st


def get_api_key():
    for name in ("ODDS_API_KEY", "odds_api_key", "THE_ODDS_API_KEY", "API_KEY"):
        if name in st.secrets:
            return st.secrets[name]
    return None
