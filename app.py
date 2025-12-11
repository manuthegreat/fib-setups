import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import run_engine, generate_trading_summary
from updater import load_all_market_data




# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Momentum Dashboard",
    layout="wide",
)

st.title("📈 Momentum & Fib Retracement Dashboard")


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Settings")



min_readiness = st.sidebar.slider(
    "Min Readiness Score",
    min_value=0,
    max_value=100,
    value=80,
    step=1,
)

min_pressure = st.sidebar.slider(
    "Min Breakout Pressure",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
)

show_only_insights = st.sidebar.checkbox(
    "Show only insight names (INSIGHT_TAGS != '')",
    value=True,
)

lookback_days = st.sidebar.slider(
    "Chart lookback (days)",
    min_value=60,
    max_value=500,
    value=180,
    step=10,
)

# -----------------------------
# Insight Tag Filter
# -----------------------------
st.sidebar.subheader("Filter by Insight Tags")

INSIGHT_OPTIONS = [
    "🔥 PRIME",
    "⚡ BOS_IMMINENT",
    "🎯 PERFECT_ENTRY",
    "🌀 STRUCTURE_STRONG",
    "📉 SQUEEZE",
    "💥 MACD_THRUST",
    "📈 EARLY_BOS",
    "🔋 ENERGY_BUILDUP",
    "🔄 REVERSAL_CONFIRM",
    "🛑 EXTENDED"
]

selected_insights = st.sidebar.multiselect(
    "Show tickers with any selected tags:",
    INSIGHT_OPTIONS,
    default=[]
)

st.sidebar.write("---")
st.sidebar.write("Run this daily after market close / before open.")




# -----------------------------
# Cache Engine Run
# -----------------------------
@st.cache_data(show_spinner=True)
def compute_dashboard():
    df_all, combined, insight_df = run_engine()
    return df_all, combined, insight_df

df_all, combined, insight_df = compute_dashboard()

if combined.empty:
    st.error("No names in watchlist / combined. Check data or parameters.")
    st.stop()


# ---------------------------------------------------------
# Apply Basic Filters
# ---------------------------------------------------------
df_view = combined.copy()
df_view = df_view[df_view["READINESS_SCORE"] >= min_readiness]
df_view = df_view[df_view["BREAKOUT_PRESSURE"] >= min_pressure]

if show_only_insights:
    df_view = df_view[df_view["INSIGHT_TAGS"] != ""]

# Apply insight tag filtering
if selected_insights:
    df_view = df_view[
        df_view["INSIGHT_TAGS"].apply(
            lambda tags: any(tag in tags for tag in selected_insights)
        )
    ]

if df_view.empty:
    st.warning("No tickers match current filters.")
    st.stop()


# ---------------------------------------------------------
# Summary Metrics
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tickers (filtered)", len(df_view))
with col2:
    st.metric("BUY signals", (df_view["FINAL_SIGNAL"] == "BUY").sum())
with col3:
    st.metric("WATCH signals", (df_view["FINAL_SIGNAL"] == "WATCH").sum())
with col4:
    st.metric("Avg Readiness", f"{df_view['READINESS_SCORE'].mean():.1f}")


# ---------------------------------------------------------
# Data Table
# ---------------------------------------------------------
st.write("### Ranked Dashboard (Filtered)")
st.dataframe(
    df_view[[
        "Ticker", "FINAL_SIGNAL", "Shape",
        "BREAKOUT_PRESSURE", "PERFECT_ENTRY", "READINESS_SCORE",
        "INSIGHT_TAGS", "NEXT_ACTION"
    ]].reset_index(drop=True),
    use_container_width=True,
)


# ---------------------------------------------------------
# Enhanced Chart Function (stable: price+FIB, then MACD+RSI)
# ---------------------------------------------------------
def plot_ticker_chart(df_all, row, lookback_days=180):
    import numpy as np

    ticker = row["Ticker"]

    # -----------------------------
    # 1. Load full history & compute indicators
    # -----------------------------
    df_full = df_all[df_all["Ticker"] == ticker].sort_values("Date").copy()
    if df_full.empty:
        st.write("No price data found.")
        return

    # Moving averages
    df_full["SMA10"] = df_full["Close"].rolling(10).mean()
    df_full["EMA20"] = df_full["Close"].ewm(span=20).mean()
    df_full["EMA50"] = df_full["Close"].ewm(span=50).mean()

    # MACD
    df_full["EMA12"] = df_full["Close"].ewm(span=12).mean()
    df_full["EMA26"] = df_full["Close"].ewm(span=26).mean()
    df_full["MACD"] = df_full["EMA12"] - df_full["EMA26"]
    df_full["Signal"] = df_full["MACD"].ewm(span=9).mean()
    df_full["MACDH"] = df_full["MACD"] - df_full["Signal"]

    # RSI
    delta = df_full["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df_full["RSI"] = 100 - (100 / (1 + rs))

    # Slice for display
    df_t = df_full.tail(lookback_days).copy()

    # =====================================================
    # 2. PRICE + FIB + MAs (figure 1)
    # =====================================================
    fig_price = go.Figure()

    # Candlesticks
    fig_price.add_trace(
        go.Candlestick(
            x=df_t["Date"],
            open=df_t["Open"],
            high=df_t["High"],
            low=df_t["Low"],
            close=df_t["Close"],
            name=ticker
        )
    )

    # MAs
    fig_price.add_trace(go.Scatter(x=df_t["Date"], y=df_t["SMA10"], name="SMA10"))
    fig_price.add_trace(go.Scatter(x=df_t["Date"], y=df_t["EMA20"], name="EMA20"))
    fig_price.add_trace(go.Scatter(x=df_t["Date"], y=df_t["EMA50"], name="EMA50"))

    # FIB levels
    swing_low = row["SwingLow"]
    swing_high = row["SwingHigh"]

    if pd.notna(swing_low) and pd.notna(swing_high):
        swing = swing_high - swing_low
        fib_levels = {
            "100%": swing_high,
            "78.6%": swing_high - 0.786 * swing,
            "61.8%": swing_high - 0.618 * swing,
            "50%":  swing_high - 0.500 * swing,
            "38.2%": swing_high - 0.382 * swing,
            "0%": swing_low,
        }

        x0 = df_t["Date"].iloc[0]
        x1 = df_t["Date"].iloc[-1]

        for label, level in fib_levels.items():
            fig_price.add_shape(
                type="line",
                x0=x0, x1=x1,
                y0=level, y1=level,
                line=dict(color="green", width=1, dash="dot")
            )
            fig_price.add_annotation(
                x=x1,
                y=level,
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=10, color="green")
            )

    fig_price.update_layout(
        height=480,
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=10),
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,   # 🔴 turn off mini candle chart
    )

    st.plotly_chart(fig_price, use_container_width=True)

    # =====================================================
    # 3. MACD + RSI (figure 2, two subplots, NO OHLC)
    # =====================================================
    fig_osc = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.45],
    )

    dates = df_t["Date"]

    # --- MACD panel (row 1) ---
    fig_osc.add_hline(y=0, line=dict(color="white", width=1), row=1, col=1)

    fig_osc.add_trace(
        go.Bar(
            x=dates,
            y=df_t["MACDH"],
            marker_color=df_t["MACDH"].apply(lambda v: "green" if v >= 0 else "red"),
            opacity=0.45,
            name="MACDH",
        ),
        row=1, col=1
    )

    fig_osc.add_trace(
        go.Scatter(x=dates, y=df_t["MACD"], name="MACD"),
        row=1, col=1
    )
    fig_osc.add_trace(
        go.Scatter(x=dates, y=df_t["Signal"], name="Signal"),
        row=1, col=1
    )

    # --- RSI panel (row 2) ---
    fig_osc.add_trace(
        go.Scatter(x=dates, y=df_t["RSI"], name="RSI"),
        row=2, col=1
    )
    fig_osc.add_hline(y=70, line=dict(color="red", dash="dot"), row=2, col=1)
    fig_osc.add_hline(y=30, line=dict(color="green", dash="dot"), row=2, col=1)

    fig_osc.update_yaxes(title_text="MACD", row=1, col=1)
    fig_osc.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    fig_osc.update_layout(
        height=320,
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=20),
        xaxis_rangeslider_visible=False,   # 🔴 also off here for cleanliness
    )

    st.plotly_chart(fig_osc, use_container_width=True)

# ---------------------------------------------------------
# Enhanced Trading Summary Card
# ---------------------------------------------------------
def render_summary_card(row):
    summary = generate_trading_summary(row)

    st.markdown("### 📘 Trading Summary")

    st.markdown(f"""
    <div style="background-color:#111;padding:15px;border-radius:10px;border:1px solid #444;">

    <h3 style="color:#4CC9F0;">🎯 Overview</h3>
    <b>Ticker:</b> {row['Ticker']}<br>
    <b>Signal:</b> {row['FINAL_SIGNAL']}<br>
    <b>Shape:</b> {row['Shape']}<br>
    <b>Insights:</b> {row['INSIGHT_TAGS']}<br>
    <b>Next Action:</b> {row['NEXT_ACTION']}<br><br>

    <h3 style="color:#4CC9F0;">📈 Interpretation</h3>
    {format_section(summary, "Interpretation:", "Your Trading Plan")}

    <h3 style="color:#4CC9F0;">📝 Trade Plan</h3>
    {format_section(summary, "Primary Entry:", "No-Trade Conditions:")}

    <h3 style="color:#F72585;">⚠️ Risk Conditions</h3>
    {format_section(summary, "No-Trade Conditions:", None)}

    </div>
    """, unsafe_allow_html=True)


def format_section(summary_text, start, end):
    try:
        section = summary_text.split(start)[1]
        if end:
            section = section.split(end)[0]
        lines = [f"• {x.strip()}" for x in section.split("\n") if x.strip()]
        return "<br>".join(lines)
    except:
        return "N/A"


# ---------------------------------------------------------
# Ticker Drilldown
# ---------------------------------------------------------
st.write("### Ticker Drilldown")

default_ticker = df_view["Ticker"].iloc[0]

selected_ticker = st.selectbox(
    "Select a ticker",
    options=df_view["Ticker"].unique(),
    index=list(df_view["Ticker"].unique()).index(default_ticker),
)

row_sel = df_view[df_view["Ticker"] == selected_ticker].iloc[0]

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Signal", row_sel["FINAL_SIGNAL"])
with colB:
    st.metric("Readiness", f"{row_sel['READINESS_SCORE']:.2f}")
with colC:
    st.metric("Breakout Pressure", f"{row_sel['BREAKOUT_PRESSURE']:.2f}")
with colD:
    st.metric("Perfect Entry", f"{row_sel['PERFECT_ENTRY']:.2f}" if pd.notna(row_sel["PERFECT_ENTRY"]) else "N/A")

st.write(f"**Shape:** {row_sel['Shape']}  |  **Insights:** {row_sel['INSIGHT_TAGS']}")

plot_ticker_chart(df_all, row_sel, lookback_days=lookback_days)

render_summary_card(row_sel)


# ---------------------------------------------------------
# Insight Summaries List
# ---------------------------------------------------------
#st.write("### All Insight Tickers – Summaries")

#for _, r in df_view.iterrows():
#    with st.expander(f"{r['Ticker']}  |  {r['INSIGHT_TAGS']}"):
#        render_summary_card(r)
