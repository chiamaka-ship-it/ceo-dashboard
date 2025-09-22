import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# ============================================================
# Config
# ============================================================
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ5-VF0xH9XbunX7DWFgtVlN_OUC9zwhYjD0ycs3anPgPgcnzYq3gcKLIMY_YonEzkDq44hbCLKgc8K/pub?gid=0&single=true&output=csv"
st.set_page_config(page_title="CEO Marketing Dashboard", layout="wide")

# --- light styling so it’s not bland ---
st.markdown("""
<style>
/* tighten layout a bit */
.block-container {padding-top: 1rem;}
/* subtle section dividers */
hr {border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.2rem 0;}
/* nicer subheaders */
h3 {margin-top: 0.6rem;}
/* table font size */
[data-testid="stDataFrame"] div[data-testid="stVerticalBlock"] {font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data loading and prep
# ============================================================
@st.cache_data(ttl=600)
def load_data(url=SHEET_CSV_URL):
    df = pd.read_csv(url)

    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # make text columns safe strings
    for col in ["section", "metric", "source", "notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # numeric cleanup: handle "3:1" like ROAS -> 3, remove commas/x
    if "numerical_value" in df.columns:
        nv = df["numerical_value"].astype(str).str.strip()
        nv = nv.str.replace(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*[0-9]+.*$", r"\1", regex=True)
        nv = nv.str.replace(",", "", regex=False).str.replace("x", "", case=False, regex=False)
        df["numerical_value"] = pd.to_numeric(nv, errors="coerce")

    # parse dates
    for col in ["week_start", "week_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # drop bad rows
    bad = {"", "nan", "none"}
    df = df.dropna(subset=["numerical_value", "week_end"])
    if "section" in df and "metric" in df:
        df = df[
            (~df["section"].str.lower().isin(bad)) &
            (~df["metric"].str.lower().isin(bad))
        ]

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
        if len(s) < 2:
            return None, None, cur
        prev = s.iloc[-2]
        if pd.isna(prev) or prev == 0:
            return None, None, cur
        delta = cur - prev
        pct = (delta / abs(prev)) * 100
        return delta, pct, cur

def delta_color_mode(metric):
    return "inverse" if not higher_is_better(metric) else "normal"

# ============================================================
# Drawing
# ============================================================
def draw_metric_card(metric, ts, mode):
    delta, pct, cur = compute_delta(ts["numerical_value"], mode=mode)
    value_str = format_value(metric, cur if cur is not None else np.nan)
    delta_str = "—" if (pct is None or pd.isna(pct)) else f"{pct:+.1f}%"
    st.metric(metric, value_str, delta_str, delta_color=delta_color_mode(metric))
    if len(ts) >= 8:
        chart = (
            alt.Chart(ts)
            .mark_line()
            .encode(x=alt.X("week_end:T", title=""), y=alt.Y("numerical_value:Q", title=""))
            .properties(height=60)
        )
        st.altair_chart(chart, use_container_width=True)

def section_table(df_section):
    """Return a table that shows ALL columns for the filtered section."""
    tbl = df_section.copy()
    # Add a formatted value column but keep original numeric too
    tbl["value_formatted"] = tbl.apply(
        lambda r: format_value(r["metric"], r["numerical_value"]), axis=1
    )
    # order columns for readability
    cols = ["section","metric","numerical_value","value_formatted","source","notes","week_start","week_end"]
    cols = [c for c in cols if c in tbl.columns]
    tbl = tbl[cols].sort_values(["metric","week_end"])
    # nicer date display
    for dcol in ["week_start","week_end"]:
        if dcol in tbl.columns:
            tbl[dcol] = pd.to_datetime(tbl[dcol], errors="coerce").dt.date
    return tbl

def draw_section(df, section, mode, show_detail_tables=True):
    subset = df[df.section == section]
    if subset.empty:
        return
    st.subheader(section)

    # KPI grid
    metrics = subset.metric.unique()
    cols = st.columns(4)
    for i, m in enumerate(metrics):
        with cols[i % 4]:
            ts = prepare_timeseries(subset, section, m)
            draw_metric_card(m, ts, mode)

    # Section detail table (ALL columns)
    if show_detail_tables:
        st.markdown("**Details**")
        st.dataframe(section_table(subset), use_container_width=True, hide_index=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# Main
# ============================================================
def main():
    df = load_data()
    if df.empty:
        st.error("No data loaded. Please check the CSV link and sheet structure.")
        return

    min_date, max_date = df.week_end.min(), df.week_end.max()
    st.title("📊 CEO Marketing Dashboard")
    st.caption(f"Last updated: {max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else '—'}")
    st.markdown("**Legend:** Green delta = improvement, Red delta = decline. Use the date range to control which rows appear in cards and tables.")

    # Sidebar controls
    st.sidebar.header("Controls")
    date_range = st.sidebar.date_input("Date range", [min_date.date(), max_date.date()])
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start, end = min_date, max_date

    # Clean section options
    secs = df["section"].astype(str).str.strip()
    section_opts = sorted(s for s in secs.unique() if s and s.lower() not in ("nan","none"))
    selected_sections = st.sidebar.multiselect("Sections", section_opts, default=section_opts)

    mode = st.sidebar.radio("Comparison mode", ["WoW", "MoM"], index=0, horizontal=False)
    show_detail_tables = st.sidebar.checkbox("Show section detail tables", value=True)
    show_raw = st.sidebar.checkbox("Show raw data (filtered)", value=False)

    # Filter by date
    mask = (df.week_end >= pd.to_datetime(start)) & (df.week_end <= pd.to_datetime(end))
    df_filtered = df[mask]

    # Render sections
    for sec in selected_sections:
        draw_section(df_filtered, sec, mode, show_detail_tables=show_detail_tables)

    # Raw data table + download
    if show_raw:
        st.header("Raw data (filtered)")
        raw_tbl = df_filtered.copy().sort_values(["section","metric","week_end"])
        for dcol in ["week_start","week_end"]:
            if dcol in raw_tbl.columns:
                raw_tbl[dcol] = pd.to_datetime(raw_tbl[dcol], errors="coerce").dt.date
        st.dataframe(raw_tbl, use_container_width=True, hide_index=True)

        csv = raw_tbl.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv, file_name="dashboard_filtered.csv", mime="text/csv")

if __name__ == "__main__":
    main()
