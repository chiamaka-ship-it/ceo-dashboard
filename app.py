import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# ============================================================
# Config
# ============================================================
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5-VF0xH9XbunX7DWFgtVlN_OUC9zwhYjD0ycs3anPgPgcnzYq3gcKLIMY_YonEzkDq44hbCLKgc8K/pub?gid=0&single=true&output=csv"  # 
st.set_page_config(page_title="CEO Marketing Dashboard", layout="wide")

# ============================================================
# Data loading and prep
# ============================================================
@st.cache_data(ttl=600)
def load_data(url=SHEET_CSV_URL):
    df = pd.read_csv(url)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df["numerical_value"] = (
        pd.to_numeric(df["numerical_value"].astype(str).str.replace(",", ""), errors="coerce")
    )
    df = df.sort_values("week_end")
    df = df.drop_duplicates(subset=["section", "metric", "week_end"], keep="last")
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
    if any(x in name for x in ["revenue", "aov"]):
        return f"₦{val:,.0f}"
    elif any(x in name for x in ["%", "rate", "index"]):
        return f"{val:.1f}%"
    elif "roas" in name:
        return f"{val:.1f}x"
    elif any(x in name for x in ["orders", "customers", "pr hits", "active users", "social", "follows"]):
        return f"{int(val):,}"
    elif "nps" in name:
        return f"{int(val)} pts"
    else:
        return f"{val:,.1f}"


def compute_delta(series, mode="WoW"):
    if series.empty:
        return None, "—", None
    if mode == "WoW":
        if len(series) < 2:
            return None, "—", None
        cur, prev = series.iloc[-1], series.iloc[-2]
    else:  # MoM = compare last 4 vs prev 4
        if len(series) < 8:
            return None, "—", None
        cur = series.iloc[-4:].mean()
        prev = series.iloc[-8:-4].mean()
    delta = cur - prev
    if prev == 0 or pd.isna(prev):
        pct = None
    else:
        pct = delta / prev * 100
    return delta, pct, cur


def delta_color(metric, delta):
    if delta is None:
        return "off"
    name = metric.lower()
    # lower is better for CAC and churn
    if any(x in name for x in ["cac", "churn"]):
        return "inverse"
    return "normal"


# ============================================================
# Drawing
# ============================================================
def draw_metric_card(section, metric, ts, mode):
    delta, pct, cur = compute_delta(ts["numerical_value"], mode=mode)
    value_str = format_value(metric, cur if cur is not None else np.nan)
    if pct is None or pd.isna(pct):
        delta_str = "—"
    else:
        sign = "+" if pct >= 0 else ""
        delta_str = f"{sign}{pct:.1f}%"
    invert = delta_color(metric, delta)
    if invert == "inverse":
        delta_str = delta_str
        st.metric(metric, value_str, delta_str, delta_color="inverse")
    else:
        st.metric(metric, value_str, delta_str)

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

    # Section-level combined trend of top 3 variable metrics
    pivot = subset.pivot_table(
        index="week_end", columns="metric", values="numerical_value", aggfunc="mean"
    )
    if pivot.shape[1] >= 3:
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

    # Controls
    st.sidebar.header("Controls")
    date_range = st.sidebar.date_input("Date range", [min_date, max_date])
    if isinstance(date_range, tuple) or isinstance(date_range, list):
        start, end = date_range
    else:
        start, end = min_date, max_date
    section_opts = df.section.unique().tolist()
    selected_sections = st.sidebar.multiselect("Sections", section_opts, default=section_opts)
    mode = st.sidebar.radio("Comparison mode", ["WoW", "MoM"])

    st.markdown(
        f"**Legend:** Green delta = improvement, Red delta = decline. Comparison mode = **{mode}**."
    )

    # Filter
    mask = (df.week_end >= pd.to_datetime(start)) & (df.week_end <= pd.to_datetime(end))
    df_filtered = df[mask]

    for section in selected_sections:
        draw_section(df_filtered, section, mode)


if __name__ == "__main__":
    main()


