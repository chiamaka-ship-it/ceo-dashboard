import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import re

# ============================================================
# Config
# ============================================================
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5-VF0xH9XbunX7DWFgtVlN_OUC9zwhYjD0ycs3anPgPgcnzYq3gcKLIMY_YonEzkDq44hbCLKgc8K/pub?gid=0&single=true&output=csv"
st.set_page_config(page_title="CEO Marketing Dashboard", layout="wide")

# ============================================================
# Data loading and prep
# ============================================================
@st.cache_data(ttl=600)
def load_data(url=SHEET_CSV_URL):
    df = pd.read_csv(url)

    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # make text columns safe strings (avoid NaN strings leaking into UI)
    for col in ["section", "metric", "source", "notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # pre-clean numeric column: handle "3:1" style for ROAS to become 3
    if "numerical_value" in df.columns:
        # keep the original as string for parsing
        nv_str = df["numerical_value"].astype(str).str.strip()

        # if looks like "3:1" -> take the left side as number
        nv_str = nv_str.str.replace(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*[0-9]+.*$", r"\1", regex=True)
        # remove commas and trailing x/X
        nv_str = nv_str.str.replace(",", "", regex=False).str.replace("x", "", case=False, regex=False)

        df["numerical_value"] = pd.to_numeric(nv_str, errors="coerce")

    # parse dates
    for col in ["week_start", "week_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # drop rows with bad/blank section/metric or missing essentials
    bad_tokens = {"", "nan", "none"}
    if "section" in df.columns and "metric" in df.columns:
        df = df[
            (~df["section"].str.lower().isin(bad_tokens)) &
            (~df["metric"].str.lower().isin(bad_tokens))
        ]
    df = df.dropna(subset=["numerical_value", "week_end"])

    # keep latest if duplicates on (section, metric, week_end)
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

    # currency-like
    if any(x in name for x in ["revenue", "aov", "gmv"]):
        return f"₦{val:,.0f}"

    # percent-like
    if any(x in name for x in ["%", "rate", "index", "share of voice", "sov"]):
        return f"{val:.1f}%"

    # roas ratio
    if "roas" in name:
        return f"{val:.1f}x"

    # nps
    if "nps" in name or "promoter" in name:
        return f"{int(val)} pts"

    # integer-ish counts
    if any(x in name for x in [
        "orders", "customers", "pr hits", "active users", "views", "reach",
        "interactions", "follows", "mentions"
    ]):
        return f"{int(round(val)):,}"

    # default
    return f"{val:,.1f}"


def higher_is_better(metric):
    name = metric.lower()
    # lower is better for CAC and churn
    if any(k in name for k in ["cac", "churn"]):
        return False
    return True


def compute_delta(series, mode="WoW"):
    """
    Returns (delta_value, pct_change, current_value)
    pct_change is a float or None. Handles NaNs safely.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None, None, None

    cur = s.iloc[-1]

    if mode == "MoM":
        # need at least 8 points for 4w vs prior 4w
        if len(s) < 8:
            return None, None, cur
        current_avg = s.iloc[-4:].mean()
        prev_avg = s.iloc[-8:-4].mean()
        if pd.isna(prev_avg) or prev_avg == 0:
            return None, None, cur
        delta = current_avg - prev_avg
        pct = (delta / abs(prev_avg)) * 100
        return delta, pct, current_avg
    else:  # WoW
        if len(s) < 2:
            return None, None, cur
        prev = s.iloc[-2]
        if pd.isna(prev) or prev == 0:
            return None, None, cur
        delta = cur - prev
        pct = (delta / abs(prev)) * 100
        return delta, pct, cur


def delta_color_mode(metric):
    """
    For metrics where lower is better, flip colors (Streamlit 'inverse').
    """
    return "inverse" if not higher_is_better(metric) else "normal"


# ============================================================
# Drawing
# ============================================================
def draw_metric_card(section, metric, ts, mode):
    delta, pct, cur = compute_delta(ts["numerical_value"], mode=mode)
    value_str = format_value(metric, cur if cur is not None else np.nan)

    delta_str = "—" if (pct is None or pd.isna(pct)) else f"{pct:+.1f}%"
    st.metric(
        metric,
        value_str,
        delta_str,
        delta_color=delta_color_mode(metric),
    )

    if len(ts) >= 8:
        chart = (
            alt.Chart(ts)
            .mark_line()
            .encode(
                x=alt.X("week_end:T", title=""),
                y=alt.Y("numerical_value:Q", title=""),
            )
            .properties(height=60)
        )
        st.altair_chart(chart, use_container_width=True)


def draw_section(df, section, mode):
    subset = df[df.section == section]
    if subset.empty:
        st.info(f"No data for {section}")
        return

    st.subheader(section)
    metrics = subset.metric.unique()
    cols = st.columns(4)
    for i, metric in enumerate(metrics):
        with cols[i % 4]:
            ts = prepare_timeseries(df, section, metric)
            draw_metric_card(section, metric, ts, mode)

    # Section-level combined trend of top 3 variable metrics (if enough)
    pivot = subset.pivot_table(
        index="week_end", columns="metric", values="numerical_value", aggfunc="mean"
    )
    if pivot.shape[1] >= 3 and pivot.shape[0] >= 2:
        top3 = pivot.std().sort_values(ascending=False).head(3).index
        melted = pivot[top3].reset_index().melt("week_end", var_name="metric", value_name="value")
        chart = (
            alt.Chart(melted)
            .mark_line()
            .encode(x="week_end:T", y="value:Q", color="metric:N")
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)


# ============================================================
# Main
# ============================================================
def main():
    df = load_data()
    if df.empty:
        st.error("No data loaded. Please check the SHEET_CSV_URL.")
        return

    min_date, max_date = df.week_end.min(), df.week_end.max()

    st.title("📊 CEO Marketing Dashboard")
    st.caption(f"Last updated: {max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else '—'}")
    st.markdown("**Legend:** Green delta = improvement, Red delta = decline.")

    # Controls
    st.sidebar.header("Controls")
    date_range = st.sidebar.date_input("Date range", [min_date.date(), max_date.date()])
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start, end = min_date, max_date

    # Clean section options (remove '', 'nan', 'none')
    section_series = df["section"].astype(str).str.strip()
    section_opts = sorted(s for s in section_series.unique() if s and s.lower() not in ("nan", "none"))
    selected_sections = st.sidebar.multiselect("Sections", section_opts, default=section_opts)

    mode = st.sidebar.radio("Comparison mode", ["WoW", "MoM"])

    st.markdown(f"_Comparison mode = **{mode}**._")

    # Filter
    mask = (df.week_end >= pd.to_datetime(start)) & (df.week_end <= pd.to_datetime(end))
    df_filtered = df[mask]

    for section in selected_sections:
        draw_section(df_filtered, section, mode)


if __name__ == "__main__":
    main()
