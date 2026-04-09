
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path


# CONFIG

DATA_PATH = Path(__file__).parent / "../data/ZRH_Flights_with_Weather_2025.csv"

# Brand colours (consistent with analysis script)
C_BLUE   = "#2563EB"
C_ORANGE = "#F59E0B"
C_RED    = "#EF4444"
C_GREEN  = "#10B981"
C_PURPLE = "#8B5CF6"
C_CYAN   = "#06B6D4"
C_PANEL  = "#1E293B"
C_TEXT   = "#E2E8F0"
C_MUTED  = "#94A3B8"

PALETTE  = [C_BLUE, C_ORANGE, C_RED, C_GREEN, C_PURPLE, C_CYAN, "#EC4899", "#84CC16"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=C_TEXT, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
)

#layout() calls that need a legend
LEGEND_STYLE = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1)


# PAGE CONFIG  (must be first Streamlit call)

st.set_page_config(
    page_title="ZRH Delay Dashboard · CIP 111",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# GLOBAL CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F172A;
    color: #E2E8F0;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #1E293B;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2563EB;
    margin-top: 1.4rem;
    margin-bottom: 0.4rem;
}

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background: #1E293B;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 20px 14px;
}
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #E2E8F0 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #2563EB;
    border-bottom: 1px solid rgba(37,99,235,0.25);
    padding-bottom: 6px;
    margin-bottom: 16px;
    margin-top: 30px;
}

/* ── chart panels ── */
.chart-panel {
    background: #1E293B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px;
}

/* ── tab bar ── */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #94A3B8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
}

/* ── title banner ── */
.title-banner {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 60%, #1a1040 100%);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.title-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(37,99,235,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.title-banner h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #E2E8F0;
    margin: 0 0 4px 0;
}
.title-banner p {
    font-size: 0.85rem;
    color: #94A3B8;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.35);
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #60A5FA;
    padding: 3px 12px;
    margin-right: 6px;
    margin-top: 10px;
}

/* ── prediction card ── */
.pred-card {
    background: linear-gradient(135deg, #1E293B, #162032);
    border: 1px solid rgba(37,99,235,0.25);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
}
.pred-value {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: #F59E0B;
    line-height: 1;
}
.pred-label {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-top: 6px;
}

/* ── insight box ── */
.insight-box {
    background: rgba(37,99,235,0.08);
    border-left: 3px solid #2563EB;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.85rem;
    color: #CBD5E1;
    line-height: 1.5;
}

/* ── model comparison table ── */
.model-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.model-table th {
    background: rgba(37,99,235,0.2);
    color: #93C5FD;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    padding: 10px 14px;
    text-align: left;
}
.model-table td { padding: 9px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #E2E8F0; }
.model-table tr:hover td { background: rgba(255,255,255,0.03); }
.best-row td { color: #10B981 !important; font-weight: 600; }
.best-row td:first-child::after { content: ' ★'; color: #F59E0B; }

/* ── hide streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# DATA LAYER  (Application Tier – cached)

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["scheduled_utc"])
    df = df[df["is_outlier"] == False].copy()
    df = df[df["delay_min"].between(-60, 300)].copy()
    df["date"] = df["scheduled_utc"].dt.date
    df["is_delayed"] = (df["delay_min"] > 5).astype(int)
    # Readable day names
    df["dow_name"] = df["dep_dow"].map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )
    df["month_name"] = df["dep_month"].map({10: "October", 11: "November", 12: "December"})
    return df


@st.cache_resource(show_spinner=False)
def run_models(_df):
    features_base    = ["dep_hour", "dep_dow", "dep_month", "is_weekend", "route_avg_delay"]
    features_weather = features_base + ["temperature_C", "precipitation_mm", "wind_speed_kmh"]

    df_m = df[features_weather + ["delay_min"]].dropna()
    X_b, X_w, y = df_m[features_base], df_m[features_weather], df_m["delay_min"]

    X_tr_b, X_te_b, y_tr, y_te = train_test_split(X_b, y, test_size=0.2, random_state=42)
    X_tr_w, X_te_w, _,    _   = train_test_split(X_w, y, test_size=0.2, random_state=42)

    results = []
    models  = {}
    for name, model, Xtr, Xte in [
        ("Linear Regression\n(Baseline)",     LinearRegression(),                               X_tr_b, X_te_b),
        ("Linear Regression\n(+ Weather)",    LinearRegression(),                               X_tr_w, X_te_w),
        ("Random Forest\n(Baseline)",         RandomForestRegressor(200, random_state=42, n_jobs=-1), X_tr_b, X_te_b),
        ("Random Forest\n(+ Weather)",        RandomForestRegressor(200, random_state=42, n_jobs=-1), X_tr_w, X_te_w),
    ]:
        model.fit(Xtr, y_tr)
        preds = model.predict(Xte)
        mae   = mean_absolute_error(y_te, preds)
        rmse  = mean_squared_error(y_te, preds) ** 0.5
        r2    = r2_score(y_te, preds)
        results.append({"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R²": round(r2, 4)})
        models[name] = (model, preds, y_te)

    # Feature importances from best RF
    rf_w = models["Random Forest\n(+ Weather)"][0]
    feat_imp = pd.Series(rf_w.feature_importances_, index=[
        "Departure Hour", "Day of Week", "Month", "Weekend Flag",
        "Route Avg Delay", "Temperature [°C]", "Precipitation [mm]", "Wind Speed [km/h]"
    ]).sort_values(ascending=True)

    return pd.DataFrame(results), models, feat_imp, y_te


# LOAD DATA

with st.spinner("Loading flight data…"):
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"❌ CSV not found at `{DATA_PATH}`. Place **ZRH_Flights_with_Weather_2025.csv** in the same folder as this script.")
        st.stop()

with st.spinner("Training models…"):
    results_df, models, feat_imp, y_te = run_models(df)



# SIDEBAR FILTERS

with st.sidebar:
    st.markdown("## ✈️ ZRH Delay\nCIP Group 111")
    st.markdown("---")

    st.markdown("## Filters")

    # Month filter
    month_options = {10: "October", 11: "November", 12: "December"}
    selected_months = st.multiselect(
        "Month", options=list(month_options.keys()),
        default=list(month_options.keys()),
        format_func=lambda x: month_options[x]
    )

    # Aircraft family
    all_families = sorted(df["aircraft_family"].dropna().unique())
    selected_families = st.multiselect(
        "Aircraft Family", options=all_families, default=all_families
    )

    # Min flights for airline chart
    min_flights = st.slider("Min. flights per airline", 10, 200, 50, step=10)

    # Weekend toggle
    show_weekend = st.checkbox("Weekend flights only", value=False)

    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        "<small style='color:#94A3B8'>Departure delays at Zurich Airport "
        "(ZRH), Oct–Dec 2025. Data: Apify API + Open-Meteo.</small>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<small style='color:#475569'>Hanieh Jebeli · Elif Gürçinar · Silan Cihaner</small>",
        unsafe_allow_html=True
    )

# Apply sidebar filters
mask = (
    df["dep_month"].isin(selected_months) &
    df["aircraft_family"].isin(selected_families)
)
if show_weekend:
    mask &= df["is_weekend"] == 1
dff = df[mask].copy()


# HEADER BANNER

st.markdown(f"""
<div class="title-banner">
    <h1>✈ ZRH Flight Delay Dashboard</h1>
    <p>Analysis and Prediction of Departure Delays · Zurich Airport · Oct – Dec 2025</p>
    <span class="badge">CIP Group 111</span>
    <span class="badge">{len(df):,} flights</span>
    <span class="badge">3 months</span>
    <span class="badge">4 ML models</span>
</div>
""", unsafe_allow_html=True)


st.info("🖥️ Best viewed in full screen for the clearest charts — press **F11** or maximise your browser window.")
# KPI METRICS ROW

m1, m2, m3, m4, m5 = st.columns(5)
overall_mean = df["delay_min"].mean()
filtered_mean = dff["delay_min"].mean() if len(dff) > 0 else 0
delta = filtered_mean - overall_mean

with m1:
    st.metric("Flights (filtered)", f"{len(dff):,}", f"{len(dff)-len(df):,} vs all")
with m2:
    st.metric("Avg Delay", f"{filtered_mean:.1f} min",
              f"{delta:+.1f} vs overall", delta_color="inverse")
with m3:
    on_time_pct = (dff["delay_min"] <= 0).mean() * 100 if len(dff) > 0 else 0
    st.metric("On-Time Rate", f"{on_time_pct:.1f}%")
with m4:
    airlines_n = dff["airline_name"].nunique()
    st.metric("Airlines", f"{airlines_n}")
with m5:
    routes_n = dff["route"].nunique()
    st.metric("Routes", f"{routes_n}")

st.markdown("<br>", unsafe_allow_html=True)



# TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Delay Explorer",
    "🌤 Weather Analysis",
    "🤖 Predictive Models",
    "🔮 Live Predictor",
])



# TAB 1 — DELAY EXPLORER

with tab1:
    st.markdown('<p class="section-header">Delay Distribution</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="medium")

    with col_a:
        # Histogram + KDE
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=dff["delay_min"], nbinsx=70,
            marker_color=C_BLUE, opacity=0.75,
            histnorm="probability density",
            name="Flights",
        ))
        # KDE overlay using plotly
        from scipy.stats import gaussian_kde
        if len(dff) > 10:
            kde_x = np.linspace(-65, 200, 300)
            kde = gaussian_kde(dff["delay_min"].dropna(), bw_method=0.15)
            fig_hist.add_trace(go.Scatter(
                x=kde_x, y=kde(kde_x),
                mode="lines", line=dict(color=C_RED, width=2.5),
                name="KDE",
            ))
        fig_hist.add_vline(x=0, line_dash="dot", line_color="white", line_width=1,
                           annotation_text="On time", annotation_font_color=C_MUTED)
        fig_hist.add_vline(x=filtered_mean, line_dash="dash", line_color=C_ORANGE,
                           line_width=1.8, annotation_text=f"Mean {filtered_mean:.1f} min",
                           annotation_font_color=C_ORANGE)
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            title="Departure Delay Distribution",
            xaxis_title="Delay [min]",
            yaxis_title="Density",
            xaxis_range=[-65, 200],
            bargap=0.04,
            height=340,
        )
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        # Delay category donut
        bins   = [-60, 0, 15, 60, 300]
        labels = ["Early / On-time", "Minor (1–15 min)", "Moderate (16–60 min)", "Severe (>60 min)"]
        colors = [C_GREEN, C_BLUE, C_ORANGE, C_RED]
        dff2 = dff.copy()
        dff2["delay_cat"] = pd.cut(dff2["delay_min"], bins=bins, labels=labels)
        cat_counts = dff2["delay_cat"].value_counts().reindex(labels).fillna(0)

        fig_donut = go.Figure(go.Pie(
            labels=cat_counts.index,
            values=cat_counts.values,
            hole=0.56,
            marker_colors=colors,
            textinfo="percent",
            textfont_size=12,
        ))
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            title="Delay Category Breakdown",
            legend=dict(orientation="v", x=1.02, y=0.5,
                        bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            height=340,
        )
        st.plotly_chart(fig_donut, width='stretch')

    # ── Hour of day & Day of week
    st.markdown('<p class="section-header">Temporal Patterns</p>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2, gap="medium")

    with col_c:
        hour_avg = dff.groupby("dep_hour")["delay_min"].mean().reset_index()
        fig_hour = go.Figure(go.Bar(
            x=hour_avg["dep_hour"], y=hour_avg["delay_min"],
            marker_color=C_BLUE, opacity=0.85,
            text=hour_avg["delay_min"].round(1),
            textposition="outside", textfont_size=9,
        ))
        fig_hour.add_hline(y=filtered_mean, line_dash="dash", line_color=C_ORANGE,
                           line_width=1.5,
                           annotation_text=f"Avg {filtered_mean:.1f} min",
                           annotation_font_color=C_ORANGE)
        fig_hour.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Delay by Departure Hour (UTC)",
            xaxis_title="Hour of Day",
            yaxis_title="Avg Delay [min]",
            height=320,
        )
        st.plotly_chart(fig_hour, width='stretch')

    with col_d:
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_avg = dff.groupby("dow_name")["delay_min"].mean().reindex(day_order).reset_index()
        colors_dow = [C_RED if v == dow_avg["delay_min"].max() else C_BLUE
                      for v in dow_avg["delay_min"]]
        fig_dow = go.Figure(go.Bar(
            x=dow_avg["dow_name"], y=dow_avg["delay_min"],
            marker_color=colors_dow, opacity=0.88,
            text=dow_avg["delay_min"].round(1),
            textposition="outside", textfont_size=9,
        ))
        fig_dow.add_hline(y=filtered_mean, line_dash="dash", line_color=C_ORANGE,
                          line_width=1.5,
                          annotation_text=f"Avg {filtered_mean:.1f} min",
                          annotation_font_color=C_ORANGE)
        fig_dow.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Delay by Day of Week",
            xaxis_title="Day",
            yaxis_title="Avg Delay [min]",
            height=320,
        )
        st.plotly_chart(fig_dow, width='stretch')

    # ── Airlines & Aircraft
    st.markdown('<p class="section-header">Airlines & Aircraft</p>', unsafe_allow_html=True)
    col_e, col_f = st.columns([3, 2], gap="medium")

    with col_e:
        airline_stats = (
            dff.groupby("airline_name")["delay_min"]
            .agg(["mean", "count"])
            .query(f"count >= {min_flights}")
            .sort_values("mean", ascending=False)
            .head(15)
        )
        bar_colors = [C_RED if v > filtered_mean * 1.5 else C_BLUE
                      for v in airline_stats["mean"]]
        fig_airline = go.Figure(go.Bar(
            y=airline_stats.index[::-1],
            x=airline_stats["mean"][::-1],
            orientation="h",
            marker_color=bar_colors[::-1],
            text=[f"n={int(c):,}  {v:.1f} min"
                  for c, v in zip(airline_stats["count"][::-1], airline_stats["mean"][::-1])],
            textposition="outside", textfont_size=9,
        ))
        fig_airline.add_vline(x=filtered_mean, line_dash="dash", line_color=C_ORANGE,
                              line_width=1.5)
        fig_airline.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Top Airlines by Avg Delay (≥ {min_flights} flights)",
            xaxis_title="Avg Delay [min]",
            height=420,
        )
        st.plotly_chart(fig_airline, width='stretch')

    with col_f:
        fam_stats = (
            dff.groupby("aircraft_family")["delay_min"]
            .agg(["mean", "count"])
            .query("count >= 20")
            .sort_values("mean", ascending=False)
        )
        fig_fam = go.Figure(go.Bar(
            x=fam_stats.index,
            y=fam_stats["mean"],
            marker_color=PALETTE[:len(fam_stats)],
            text=fam_stats["mean"].round(1),
            textposition="outside", textfont_size=10,
        ))
        fig_fam.add_hline(y=filtered_mean, line_dash="dash", line_color=C_ORANGE,
                          line_width=1.5,
                          annotation_text=f"Avg {filtered_mean:.1f} min",
                          annotation_font_color=C_ORANGE)
        fig_fam.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Delay by Aircraft Family",
            yaxis_title="Avg Delay [min]",
            height=420,
        )
        st.plotly_chart(fig_fam, width='stretch')

    # ── Monthly heatmap: hour × day
    st.markdown('<p class="section-header">Delay Heatmap: Hour × Day of Week</p>', unsafe_allow_html=True)
    pivot = dff.pivot_table(index="dep_hour", columns="dow_name", values="delay_min", aggfunc="mean")
    pivot = pivot.reindex(columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, "#0F172A"], [0.3, "#1E3A6E"], [0.6, C_ORANGE], [1, C_RED]],
        colorbar=dict(title="min", tickfont=dict(color=C_TEXT)),
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
        textfont_size=9,
    ))
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        title="Average Departure Delay [min] — Hour of Day × Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Departure Hour (UTC)",
        height=420,
    )
    st.plotly_chart(fig_heat, width='stretch')



# TAB 2 — WEATHER ANALYSIS

with tab2:
    st.markdown('<p class="section-header">Weather Variables vs Delay</p>', unsafe_allow_html=True)

    col_w1, col_w2 = st.columns(2, gap="medium")

    with col_w1:
        # Correlation heatmap
        weather_cols = ["delay_min", "temperature_C", "precipitation_mm", "wind_speed_kmh"]
        readable = {
            "delay_min": "Delay [min]",
            "temperature_C": "Temperature [°C]",
            "precipitation_mm": "Precipitation [mm]",
            "wind_speed_kmh": "Wind Speed [km/h]",
        }
        corr = dff[weather_cols].dropna().rename(columns=readable).corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns, y=corr.index,
            colorscale="RdBu_r", zmid=0,
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont_size=12,
            colorbar=dict(title="r", tickfont=dict(color=C_TEXT)),
        ))
        fig_corr.update_layout(
            **PLOTLY_LAYOUT,
            title="Pearson Correlation Matrix",
            height=380,
        )
        st.plotly_chart(fig_corr, width='stretch')

    with col_w2:
        # Scatter: wind speed vs delay
        sample = dff[["wind_speed_kmh", "delay_min", "temperature_C"]].dropna().sample(
            min(2000, len(dff)), random_state=42
        )
        fig_scatter = px.scatter(
            sample, x="wind_speed_kmh", y="delay_min",
            color="temperature_C",
            color_continuous_scale="RdYlBu_r",
            labels={
                "wind_speed_kmh": "Wind Speed [km/h]",
                "delay_min": "Delay [min]",
                "temperature_C": "Temp [°C]",
            },
            opacity=0.4,
            title="Delay vs Wind Speed (coloured by Temperature)",
        )
        fig_scatter.update_traces(marker_size=5)
        fig_scatter.update_layout(**PLOTLY_LAYOUT, legend=LEGEND_STYLE, height=380)
        st.plotly_chart(fig_scatter, width='stretch')

    # ── Grouped bar charts
    col_w3, col_w4, col_w5 = st.columns(3, gap="medium")

    with col_w3:
        dff["wind_bin"] = pd.cut(dff["wind_speed_kmh"],
                                  bins=[0, 10, 20, 30, 999],
                                  labels=["0–10", "10–20", "20–30", "30+"])
        wind_avg = dff.groupby("wind_bin", observed=True)["delay_min"].agg(["mean", "count"])
        fig_wind = go.Figure(go.Bar(
            x=wind_avg.index.astype(str),
            y=wind_avg["mean"],
            marker_color=PALETTE[:4],
            text=[f"{v:.1f}" for v in wind_avg["mean"]],
            textposition="outside",
        ))
        fig_wind.add_hline(y=filtered_mean, line_dash="dash", line_color=C_ORANGE, line_width=1.5)
        fig_wind.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Delay by Wind [km/h]",
            xaxis_title="Wind Speed [km/h]",
            yaxis_title="Avg Delay [min]",
            height=320,
        )
        st.plotly_chart(fig_wind, width='stretch')

    with col_w4:
        dff["temp_bin"] = pd.cut(dff["temperature_C"],
                                  bins=[-99, -5, 0, 5, 10, 99],
                                  labels=["<−5", "−5 to 0", "0 to 5", "5 to 10", ">10"])
        temp_avg = dff.groupby("temp_bin", observed=True)["delay_min"].agg(["mean", "count"])
        fig_temp = go.Figure(go.Bar(
            x=temp_avg.index.astype(str),
            y=temp_avg["mean"],
            marker_color=PALETTE[:5],
            text=[f"{v:.1f}" for v in temp_avg["mean"]],
            textposition="outside",
        ))
        fig_temp.add_hline(y=filtered_mean, line_dash="dash", line_color=C_ORANGE, line_width=1.5)
        fig_temp.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Delay by Temperature [°C]",
            xaxis_title="Temperature [°C]",
            yaxis_title="Avg Delay [min]",
            height=320,
        )
        st.plotly_chart(fig_temp, width='stretch')

    with col_w5:
        dff["precip_bin"] = pd.cut(dff["precipitation_mm"],
                                    bins=[-0.1, 0, 0.5, 1, 999],
                                    labels=["None", "Trace", "Light", "Moderate+"])
        prec_avg = dff.groupby("precip_bin", observed=True)["delay_min"].agg(["mean", "count"])
        fig_prec = go.Figure(go.Bar(
            x=prec_avg.index.astype(str),
            y=prec_avg["mean"],
            marker_color=[C_GREEN, C_BLUE, C_ORANGE, C_RED],
            text=[f"{v:.1f}" for v in prec_avg["mean"]],
            textposition="outside",
        ))
        fig_prec.add_hline(y=filtered_mean, line_dash="dash", line_color=C_ORANGE, line_width=1.5)
        fig_prec.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Delay by Precipitation [mm]",
            xaxis_title="Precipitation [mm]",
            yaxis_title="Avg Delay [min]",
            height=320,
        )
        st.plotly_chart(fig_prec, width='stretch')

    st.markdown(
        '<div class="insight-box">💡 <b>Key Insight:</b> All weather variables show very low '
        'linear correlation with departure delay (|r| < 0.05). Higher wind speed categories show '
        'marginally elevated delays. Weather is only one of many delay drivers — airline-network '
        'cascading effects dominate at ZRH during Oct–Dec 2025.</div>',
        unsafe_allow_html=True
    )



# TAB 3 — PREDICTIVE MODELS

with tab3:
    st.markdown('<p class="section-header">Model Performance — RQ1 & RQ2</p>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([2, 3], gap="medium")

    with col_m1:
        # Model table
        table_html = """
        <table class="model-table">
            <tr><th>Model</th><th>MAE [min]</th><th>RMSE [min]</th><th>R²</th></tr>
        """
        best_r2_idx = results_df["R²"].idxmax()
        for i, row in results_df.iterrows():
            cls = "best-row" if i == best_r2_idx else ""
            name_clean = row["Model"].replace("\n", " ")
            table_html += (
                f'<tr class="{cls}">'
                f"<td>{name_clean}</td>"
                f"<td>{row['MAE']}</td>"
                f"<td>{row['RMSE']}</td>"
                f"<td>{row['R²']:.4f}</td>"
                f"</tr>"
            )
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        best_row = results_df.loc[best_r2_idx]
        best_name = best_row["Model"].replace("\n", " ")
        best_r2 = best_row["R²"]
        st.markdown(
            f'<div class="insight-box">★ <b>Best model:</b> {best_name} '
            f'(R² = {best_r2:.3f}). Weather adds only marginal improvement (+1.3 pp R²).<br><br>'
            '⚠️ Low overall R² (~0.05–0.07) is expected: most delay variance comes from '
            'cascading network effects not observable before departure.</div>',
            unsafe_allow_html=True
        )

    with col_m2:
        # Grouped bar: MAE / RMSE / R²
        model_names = [r["Model"].replace("\n", " ") for _, r in results_df.iterrows()]
        fig_cmp = make_subplots(rows=1, cols=3,
                                subplot_titles=["MAE [min]", "RMSE [min]", "R² [-]"])
        for col_idx, metric in enumerate(["MAE", "RMSE", "R²"], start=1):
            fig_cmp.add_trace(
                go.Bar(
                    x=model_names, y=results_df[metric],
                    marker_color=PALETTE[:4],
                    text=[f"{v:.3f}" for v in results_df[metric]],
                    textposition="outside", textfont_size=9,
                    showlegend=False,
                ),
                row=1, col=col_idx
            )
        fig_cmp.update_layout(
            **PLOTLY_LAYOUT,
            legend=LEGEND_STYLE,
            title="Model Comparison: Baseline vs +Weather  |  LR vs Random Forest",
            height=380,
        )
        fig_cmp.update_xaxes(tickangle=15, tickfont_size=9)
        st.plotly_chart(fig_cmp, width='stretch')

    # ── Feature importance + Actual vs Predicted
    st.markdown('<p class="section-header">Feature Importance & Prediction Quality — RQ3</p>',
                unsafe_allow_html=True)
    col_m3, col_m4 = st.columns(2, gap="medium")

    with col_m3:
        fig_fi = go.Figure(go.Bar(
            y=feat_imp.index,
            x=feat_imp.values,
            orientation="h",
            marker=dict(
                color=feat_imp.values,
                colorscale=[[0, C_BLUE], [0.6, C_PURPLE], [1, C_ORANGE]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in feat_imp.values],
            textposition="outside", textfont_size=9,
        ))
        fig_fi.update_layout(
            **PLOTLY_LAYOUT,
            title="Feature Importance — Random Forest (+Weather)",
            xaxis_title="Importance [-]",
            height=360,
        )
        st.plotly_chart(fig_fi, width='stretch')

    with col_m4:
        rf_model, rf_preds, y_te_ref = models["Random Forest\n(+ Weather)"]
        n = min(1500, len(y_te_ref))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_te_ref), size=n, replace=False)
        y_sample = np.array(y_te_ref)[idx]
        p_sample  = rf_preds[idx]

        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_sample, y=p_sample,
            mode="markers",
            marker=dict(color=C_BLUE, size=5, opacity=0.35),
            name="Flights",
        ))
        lim = max(abs(y_sample).max(), abs(p_sample).max())
        fig_avp.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim],
            mode="lines",
            line=dict(color=C_RED, width=1.8, dash="dash"),
            name="Perfect prediction",
        ))
        fig_avp.update_layout(
            **PLOTLY_LAYOUT,
            legend=LEGEND_STYLE,
            title="Actual vs Predicted Delay — RF + Weather",
            xaxis_title="Actual Delay [min]",
            yaxis_title="Predicted Delay [min]",
            height=360,
        )
        st.plotly_chart(fig_avp, width='stretch')

    st.markdown(
        '<div class="insight-box">💡 <b>RQ3 Answer:</b> Route average historical delay is the '
        'single strongest predictor, followed by departure hour. Weather variables rank lowest — '
        'confirming that time-of-day and route-level history matter far more than meteorology '
        'for pre-departure delay prediction at ZRH.</div>',
        unsafe_allow_html=True
    )



# TAB 4 — LIVE PREDICTOR

with tab4:
    st.markdown('<p class="section-header">Predict Delay for a New Flight</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Fill in the flight parameters below. The **Linear Regression + Weather** model "
        "(best R²) will estimate the expected departure delay in real time.",
        unsafe_allow_html=True
    )

    p_col1, p_col2, p_col3 = st.columns(3, gap="large")

    with p_col1:
        st.markdown("**🕐 Temporal**")
        dep_hour  = st.slider("Departure Hour (UTC)", 0, 23, 10)
        dep_dow   = st.selectbox("Day of Week",
                                  options=list(range(7)),
                                  format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
        dep_month = st.selectbox("Month", sorted(df["dep_month"].unique()),
                                 format_func=lambda x: {10: "October", 11: "November", 12: "December"}.get(x, str(x)))
        is_weekend = 1 if dep_dow >= 5 else 0
        st.caption(f"Weekend: {'Yes ✓' if is_weekend else 'No'}")

    with p_col2:
        st.markdown("**✈️ Route**")
        route_avg_min = float(df["route_avg_delay"].min())
        route_avg_max = float(df["route_avg_delay"].max())
        route_avg_delay = st.slider(
            "Route Historical Avg Delay [min]",
            min_value=round(route_avg_min, 1),
            max_value=round(min(route_avg_max, 60.0), 1),
            value=5.0, step=0.5
        )
        st.caption("Higher = historically delay-prone route")

    with p_col3:
        st.markdown("**🌤 Weather at ZRH**")
        temperature_C    = st.slider("Temperature [°C]", -15, 25, 8)
        precipitation_mm = st.slider("Precipitation [mm]", 0.0, 5.0, 0.0, step=0.1)
        wind_speed_kmh   = st.slider("Wind Speed [km/h]", 0, 60, 15)

    # Run prediction
    X_input = pd.DataFrame([[
        dep_hour, dep_dow, dep_month, is_weekend, route_avg_delay,
        temperature_C, precipitation_mm, wind_speed_kmh
    ]], columns=["dep_hour", "dep_dow", "dep_month", "is_weekend", "route_avg_delay",
                 "temperature_C", "precipitation_mm", "wind_speed_kmh"])

    lr_w_model = models["Linear Regression\n(+ Weather)"][0]
    rf_w_model = models["Random Forest\n(+ Weather)"][0]
    pred_lr = lr_w_model.predict(X_input)[0]
    pred_rf = rf_w_model.predict(X_input)[0]
    pred_avg = (pred_lr + pred_rf) / 2

    st.markdown("<br>", unsafe_allow_html=True)

    res1, res2, res3, res4 = st.columns(4, gap="medium")
    with res1:
        st.markdown(
            f'<div class="pred-card">'
            f'<div class="pred-value">{pred_lr:.1f}</div>'
            f'<div class="pred-label">LR + Weather [min]</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with res2:
        st.markdown(
            f'<div class="pred-card">'
            f'<div class="pred-value">{pred_rf:.1f}</div>'
            f'<div class="pred-label">RF + Weather [min]</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with res3:
        colour = C_RED if pred_avg > 15 else (C_ORANGE if pred_avg > 5 else C_GREEN)
        st.markdown(
            f'<div class="pred-card" style="border-color:{colour}40">'
            f'<div class="pred-value" style="color:{colour}">{pred_avg:.1f}</div>'
            f'<div class="pred-label">Ensemble Avg [min]</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with res4:
        status = "🔴 Likely delayed" if pred_avg > 5 else "🟢 Likely on time"
        conf   = "Low confidence (R² ≈ 6%)"
        st.markdown(
            f'<div class="pred-card">'
            f'<div style="font-size:1.4rem;font-weight:700;color:{C_TEXT};line-height:1.2">{status}</div>'
            f'<div class="pred-label" style="margin-top:10px">{conf}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge chart
    gauge_val = max(-30, min(pred_avg, 90))
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        delta={"reference": filtered_mean, "suffix": " min vs fleet avg",
               "increasing": {"color": C_RED}, "decreasing": {"color": C_GREEN}},
        number={"suffix": " min", "font": {"size": 42, "family": "Space Mono", "color": C_TEXT}},
        gauge={
            "axis": {"range": [-30, 90], "tickwidth": 1, "tickcolor": C_MUTED,
                     "tickfont": {"color": C_MUTED}},
            "bar": {"color": C_ORANGE, "thickness": 0.25},
            "bgcolor": C_PANEL,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [-30, 0],  "color": "rgba(16,185,129,0.15)"},
                {"range": [0, 15],   "color": "rgba(37,99,235,0.15)"},
                {"range": [15, 60],  "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 90],  "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": C_RED, "width": 3},
                "thickness": 0.8,
                "value": 15,
            },
        },
        title={"text": "Predicted Departure Delay", "font": {"color": C_TEXT, "size": 15}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=C_TEXT,
        height=320,
        margin=dict(l=40, r=40, t=20, b=10),
    )
    c_gauge, _ = st.columns([2, 1])
    with c_gauge:
        st.plotly_chart(fig_gauge, width='stretch')

    st.markdown(
        '<div class="insight-box">⚠️ <b>Disclaimer:</b> The models explain only ~6% of delay '
        'variance (R² ≈ 0.05–0.07). Predictions are indicative only. Most departure delay is '
        'caused by cascading network effects (aircraft rotation, crew positioning, inbound delay) '
        'that are not captured in pre-departure features alone.</div>',
        unsafe_allow_html=True
    )