"""
streamlit_sensex_dashboard.py
Interactive Streamlit dashboard for sector-wise Sensex analysis.

Requirements:
pip install streamlit yfinance matplotlib pandas numpy scikit-learn mplfinance
Run:
streamlit run streamlit_sensex_dashboard.py
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import mplfinance as mpf
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Sensex Sector Analyzer")

SECTORS = {
    "Bank": "^BSEBANK",
    "Auto": "^BSEAUTO",
    "FMCG": "^BSEFMCG",
    "IT": "^BSEIT",
    "Healthcare": "^BSEHEALTH",
    "Metal": "^BSEMETAL",
}

@st.cache_data(ttl=300)
def fetch_df(symbol, period_days=180):
    df = yf.download(symbol, period=f"{period_days}d", progress=False)
    return df

def plot_scatter_ax(ax, current, future, sector):
    x = np.array(current).reshape(-1, 1)
    y = np.array(future)
    model = LinearRegression().fit(x, y)
    y_line = model.predict(x)
    ax.scatter(x, y, s=18, alpha=0.6)
    ax.plot(x, y_line, linewidth=1.5)
    ax.set_title(f"{sector}: Current vs Next-day Close")
    ax.set_xlabel("Current Close")
    ax.set_ylabel("Next-day Close")
    ax.grid(True)
    return model

def plot_combined_ax(ax, data_dict):
    for sector, series in data_dict.items():
        ax.scatter(series[:-1], series[1:], s=12, alpha=0.6, label=sector)
    ax.set_title("All Sectors: Current vs Next-day Close")
    ax.set_xlabel("Current Close")
    ax.set_ylabel("Next-day Close")
    ax.legend()
    ax.grid(True)

def plot_candlestick_bytes(df, title):
    df_local = df.copy()
    df_local.index.name = 'Date'
    df_local = df_local[['Open','High','Low','Close','Volume']].dropna()
    df_local['MA50'] = df_local['Close'].rolling(50).mean()
    df_local['MA100'] = df_local['Close'].rolling(100).mean()
    # Use mplfinance to generate a PNG in-memory
    buf = BytesIO()
    mpf.plot(df_local, type='candle', style='yahoo', volume=True, mav=(50,100),
             title=title, savefig=buf)
    buf.seek(0)
    return buf

def prepare_lag_features(series, n_lags=5):
    df = pd.DataFrame({'Close': series})
    for i in range(1, n_lags+1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    X = df[[f'lag_{i}' for i in range(1, n_lags+1)]].values
    y = df['Close'].values
    return X, y, df

def predict_next(series, n_lags=5):
    X, y, df = prepare_lag_features(series, n_lags=n_lags)
    if len(X) < 10:
        return None, None
    model = LinearRegression()
    model.fit(X, y)
    last_vals = series[-n_lags:][::-1]
    feat = np.array(last_vals).reshape(1, -1)
    pred = model.predict(feat)[0]
    return pred, model

# Sidebar
st.sidebar.header("Controls")
sector_choice = st.sidebar.selectbox("Sector", list(SECTORS.keys()))
days = st.sidebar.slider("Days to fetch", min_value=60, max_value=720, value=300, step=30)
show_combined = st.sidebar.checkbox("Show combined sectors scatter", value=True)
show_candlestick = st.sidebar.checkbox("Show candlestick + MAs", value=True)
n_lags = st.sidebar.slider("ML model: number of lags", min_value=3, max_value=10, value=5)
predict_button = st.sidebar.button("Predict Next Day Close")

# Fetch data for selected sector
symbol = SECTORS[sector_choice]
with st.spinner(f"Fetching {sector_choice} data ({symbol}) ..."):
    df_sector = fetch_df(symbol, period_days=days)

st.title("Sensex Sector Analyzer")
st.markdown(f"**Sector:** {sector_choice} — symbol: `{symbol}` — rows fetched: {len(df_sector)}")

if df_sector.empty:
    st.error("No data fetched. Try increasing 'Days to fetch' or check your internet connection.")
    st.stop()

# Show latest rows
st.subheader("Latest data (last 5 rows)")
st.dataframe(df_sector.tail(5))

# Scatter for selected sector
st.subheader("Scatter: Current vs Next-day Close (with regression)")
col1, col2 = st.columns([2,1])
with col1:
    fig, ax = plt.subplots(figsize=(6,4))
    closes = df_sector['Close'].dropna()
    if len(closes) >= 2:
        model = plot_scatter_ax(ax, closes[:-1], closes[1:], sector_choice)
        st.pyplot(fig)
        st.caption("Scatter plotted using Close(t) vs Close(t+1). Linear regression line shown.")
    else:
        st.write("Not enough points to plot scatter.")

# Candlestick
with col2:
    if show_candlestick and len(df_sector) >= 60:
        buf = plot_candlestick_bytes(df_sector[-180:], f"{sector_choice} - Last {min(180, len(df_sector))} days")
        st.image(buf)
    else:
        st.write("Candlestick not shown (toggle on sidebar or need at least 60 rows).")

# Combined scatter
if show_combined:
    st.subheader("Combined sectors: Current vs Next-day Close")
    # fetch closes for all sectors (cached)
    data_closes = {}
    for s, sym in SECTORS.items():
        df = fetch_df(sym, period_days=days)
        if not df.empty:
            data_closes[s] = df['Close'].dropna()
    fig2, ax2 = plt.subplots(figsize=(8,5))
    plot_combined_ax(ax2, data_closes)
    st.pyplot(fig2)

# Prediction
st.subheader("Machine Learning: Predict Next Day Close (simple lag-based LinearRegression)")
if predict_button:
    with st.spinner("Training model and predicting..."):
        pred, model = predict_next(df_sector['Close'].values, n_lags=n_lags)
        if pred is None:
            st.warning("Not enough historical data to train model. Increase 'Days to fetch'.")
        else:
            last_date = df_sector.index[-1].date()
            st.success(f"Predicted next-day close for {sector_choice} (based on data up to {last_date}): {pred:.2f}")
            st.write("Model coefficients:", model.coef_, "Intercept:", model.intercept_)
            # show simple diagnostics: last real vs predicted for the train set
            X, y, df_train = prepare_lag_features(df_sector['Close'].values, n_lags=n_lags)
            preds_train = model.predict(X)
            import sklearn.metrics as metrics
            mae = metrics.mean_absolute_error(y, preds_train)
            rmse = np.sqrt(metrics.mean_squared_error(y, preds_train))
            st.write(f"Train MAE: {mae:.3f}, RMSE: {rmse:.3f}")

st.markdown("---")
st.caption("Notes: This app uses simple models and daily index close prices from yfinance. Use predictions only for demonstration/educational purpose; not financial advice.")
