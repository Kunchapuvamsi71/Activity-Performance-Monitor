"""Shared look and feel for the Streamlit app (navy sidebar, gold highlights)."""
import streamlit as st

ACCENT = "#0B6B57"
GOLD = "#E9C46A"


def apply_theme():
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Playfair+Display:wght@600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Playfair Display', serif !important; color: #14213D; }}
[data-testid="stSidebar"] {{ background: #0F1A2E; }}
[data-testid="stSidebar"] * {{ color: #E8EAF0 !important; }}
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #E6E1D6; border-radius: 14px; padding: 16px; }}
div.stButton > button, div.stDownloadButton > button {{ background: {ACCENT}; color: #fff; border: 0; border-radius: 8px; min-height: 44px; font-weight: 600; }}
</style>
""",
        unsafe_allow_html=True,
    )


def sidebar_brand(title, footer=""):
    st.sidebar.markdown(
        f"<div style='font-family:Playfair Display,serif;font-size:22px;font-weight:700;color:{GOLD} !important'>{title}</div>"
        f"<div style='height:2px;width:34px;background:{GOLD};margin:8px 0 18px'></div>",
        unsafe_allow_html=True,
    )
    return footer


def hero(title, text):
    st.markdown(f"## {title}")
    st.write(text)
