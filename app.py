import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# ============================================================
# Config
# ============================================================
# Your published Google Sheet CSV link (tested pattern)
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5-VF0xH9XbunX7DWFgtVlN_OUC9zwhYjD0ycs3anPgPgcnzYq3gcKLIMY_YonEzkDq44hbCLKgc8K/pub?gid=0&single=true&output=csv"

st.set_page_config(page_title="CEO Marketing Dashboard", layout="wide")

# --- subtle styling for a more polished look ---
st.markdown("""
<style>
.block-container {padding-top: 0.8rem;}
h1, h2, h3 { letter-spacing: 0.2px; }
hr {border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 1rem 0 1.2rem;}
.card {padding: 0.6rem 0.8rem; border-radius: 12px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);}
.section-wrap {padding: 0.8rem 1rem; border-radius: 14px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); margin-bottom: 1.2rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data loading and prep
# ============================================================
@st.cache_data(ttl=600)
def load_data(url=SHEET_CSV_URL):
    # Validate link first
    if not isinstance(url, str) or not url.strip().lower().startswith(("http://", "https://")):
        st.error("SHEET_CSV_URL is not a valid http(s) link. "
                 "Open app.py and set SHEET_CSV_URL to your Google Sheet CSV URL.")
        st.stop()

    # Read CSV (will show a friendly error if it fails)
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(
            "Couldn't read CSV from the provided URL.\n\n"
            "Quick checks:\n"
            "• Open the URL in a new tab — it should download or show plain CSV.\n"
            "• If it asks for permission, publish the sheet to the web (CSV) or set 'Anyone with the link: Viewer'.\n"
            "• Make sure the URL ends with `...export?format=csv` or `...output=csv`.\n\n"
            f"Original error: {e}"
        )
        st.stop()

    # normalize column names -> snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # text columns as strings
    for col in ["section", "metric", "source", "notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # numeric cleanup: handle "3:1" -> 3, remove commas and 'x'
    if "numerical_value" in df.columns:
        nv = df["numerical_value"].astype(str).str.strip()
        nv = nv.str.replace(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*[0-9]+.*$", r"\1", regex=True)  # 3:1 -> 3
        nv = nv.str.replace(",", "", regex=False).str.replace("x", "", case=False, regex=False)
        df["numerical_value"] = pd.to_numeric(nv, errors="coerce")

    # parse dates
    for col in ["week_start", "week_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # drop bad rows
    bad = {"", "nan", "none"}
    if "section" in df.columns and "metric" in df.columns:
        df = df[
            (~df["section"].str.lower().isin(bad)) &
            (~df["metric"].str.lower().isin(bad))
        ]

    df = df.dropna(subset=["numerical_value", "week_end"])

    # keep latest if duplicate (section, metric, week_end)
    df = df.sort_values("week_end").drop_duplicates(
        subset=["section", "metric", "week_end"], keep="last"
    )
    return df


def prepare_timeseries(df, section, metric):
    ts = df[(df.section == section) & (df.metric == metric)].copy()
    ts = ts.sort_values("week_end")
    return ts[["week_end", "numerical_value"]]

# ============================================================
# Helpers
# ============================================================
def format_value(metric, val):
    if pd.isna(val):
        return "—"
    name = metric.lower()
    if any(x in name for x in ["revenue", "aov", "gmv"]):
        return f"₦{val:,.0f}"
    if any(x in name for x in ["%", "rate", "index", "share of voice", "sov"]):
        return f"{val:.1f}%"
    if "roas" in name:
        return f"{val:.1f}x"
    if "nps" in name or "promoter" in name:
        return f"{int(round(val))} pts"
    if any(x in name for x in ["orders","customers","pr hits","active users","views","reach","interactions","follows","mentions"]):
        return f"{int(round(val)):,}"
    return f"{val:,.1f}"

def higher_is_better(metric):
    name = metric.lower()
    return not any(k in name for k in ["cac", "churn"])

def compute_delta(series, mode="WoW"):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None, None, None
    cur = s.iloc[-1]
    if mode == "MoM":
        if len(s) < 8:
            return None, None, cur
        cur_avg, prev_avg = s.iloc[-4:].mean(), s.iloc[-8:-4].mean()
        if pd.isna(prev_avg) or prev_avg == 0:
            return None, None, cur_avg
        delta = cur_avg - prev_avg
        pct = (delta / abs(prev_avg)) * 100
        return delta, pct, cur_avg
    else:
        i
