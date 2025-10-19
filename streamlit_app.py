# app.py
# Trading AI Assistant v3.1 — oprava chyby + moderní design
# pip install streamlit yfinance plotly scikit-learn pandas numpy

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -------------------------
# ⚙️ Nastavení stránky
# -------------------------
st.set_page_config(page_title="Trading AI Assistant v3.1", layout="wide")

st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at top right, #10131A, #0D0F13 70%);
        color: #E0E0E0;
    }
    .stApp {
        background: linear-gradient(180deg, #0E1014 0%, #1A1C22 100%);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(#00FFD1, #00AEEF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .metric-card {
        background-color: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Trading AI Assistant — Beta 3.1")
st.markdown("Interaktivní graf + ensemble predikce (EMA, RSI, Slope, Linear Regression).") 
**Asistent — ne investiční rada.**")

# -------------------------
# 📊 Pomocné funkce
# -------------------------
@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df[['Close']].dropna()

def add_indicators(df: pd.DataFrame):
    df = df.copy()
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['slope_6'] = df['Close'].pct_change().rolling(6).mean()
    return df

def linreg_predict(series: pd.Series, steps: int):
    if len(series) < 3:
        return np.array([series.iloc[-1]] * steps)
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression().fit(X, y)
    future_idx = np.arange(len(series), len(series) + steps).reshape(-1, 1)
    return model.predict(future_idx)

# 🧩 Opravená ensemble funkce
def ensemble_score_for_point(window: pd.Series, indicators: pd.DataFrame):
    if window is None or len(window) < 3 or indicators.empty:
        return 0.0
    try:
        X = np.arange(len(window)).reshape(-1,1)
        lr = LinearRegression().fit(X, window.values)
        slope = lr.coef_[0] / (float(window.iat[-1]) + 1e-12)
    except Exception:
        slope = 0.0
    last = indicators.iloc[-1]
    def safe_num(val):
        try:
            v = pd.to_numeric(val, errors='coerce')
            if pd.isna(v): return None
            return float(v)
        except Exception:
            return None
    ema9 = safe_num(last.get('EMA9'))
    ema21 = safe_num(last.get('EMA21'))
    rsi = safe_num(last.get('RSI'))
    slope_proxy = safe_num(last.get('slope_6')) or 0.0
    ema_signal = 1 if (ema9 is not None and ema21 is not None and ema9 > ema21) else -1
    rsi_signal = 1 if (rsi is not None and rsi < 35) else -1 if (rsi is not None and rsi > 65) else 0
    slope_proxy_signal = 1 if slope_proxy > 0 else -1 if slope_proxy < 0 else 0
    score = (0.45 * np.tanh(slope * 1000)) + (0.25 * ema_signal) + (0.2 * rsi_signal) + (0.1 * slope_proxy_signal)
    return float(max(-1.0, min(1.0, score)))

def rolling_backtest_ensemble(series, indicators, train_window, horizon):
    acc = []
    n = len(series)
    for start in range(0, n - train_window - horizon):
        train = series.iloc[start:start+train_window]
        test_target = series.iloc[start+train_window:start+train_window+horizon]
        slice_ = indicators.iloc[start:start+train_window]
        if slice_.isna().any().any(): continue
        score = ensemble_score_for_point(train, slice_)
        pred_dir = 1 if score > 0 else 0
        actual_dir = 1 if test_target.iloc[-1] > train.iloc[-1] else 0
        acc.append(pred_dir == actual_dir)
        if len(acc) >= 200:
            break
    return (sum(acc)/len(acc)) if acc else None, len(acc)

def calibrate_probability(raw_score: float, hist_acc: float):
    base = hist_acc if hist_acc is not None else 0.5
    conf = abs(raw_score)
    p = 0.65*base + 0.35*(0.5 + 0.5*conf*(1 if raw_score>0 else -1))
    return float(max(0.01, min(0.99, p)))

# -------------------------
# 🧭 UI
# -------------------------
col1, col2, col3 = st.columns([3,2,2])
with col1:
    symbol = st.text_input("Symbol (např. BTC-USD, ETH-USD, AAPL):", value="BTC-USD")
with col2:
    period = st.selectbox("Období dat:", ["7d","30d","90d","1y"])
with col3:
    interval = st.selectbox("Interval:", ["15m","30m","1h","4h"])

horizon_hours = st.slider("Predikce dopředu (hodin):", 1, 24, 6)
train_window = st.slider("Velikost trénovacího okna:", 8, 200, 48)
run = st.button("🔄 Aktualizovat / Spočítat predikci")

# -------------------------
# 🔍 Hlavní logika
# -------------------------
if run:
    df = fetch_data(symbol, period, interval)
    if df.empty:
        st.error("Nepodařilo se stáhnout data.")
        st.stop()
    df = add_indicators(df)
    if len(df) < train_window + 2:
        st.error("Málo dat pro zvolený train_window.")
        st.stop()

    last_window = df['Close'].iloc[-train_window:]
    indicators_df = df[['EMA9','EMA21','RSI','slope_6']]
    raw_score = ensemble_score_for_point(last_window, indicators_df)
    hist_acc, windows = rolling_backtest_ensemble(df['Close'], indicators_df, train_window, horizon=1)

    preds = linreg_predict(df['Close'], horizon_hours)
    delta = df.index[-1] - df.index[-2] if len(df.index) > 1 else pd.Timedelta(hours=1)
    future_index = [df.index[-1] + (i+1)*delta for i in range(horizon_hours)]
    pred_price = preds[-1]
    last_price = df['Close'].iloc[-1]
    rel_change = (pred_price - last_price) / (last_price + 1e-12) * 100
    prob = calibrate_probability(raw_score, hist_acc)

    st.markdown("---")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.markdown(f"<div class='metric-card'><h3>📈 Poslední cena</h3><h2>{last_price:.2f}</h2></div>", unsafe_allow_html=True)
    mcol2.markdown(f"<div class='metric-card'><h3>🔮 Predikce (+{horizon_hours}h)</h3><h2>{pred_price:.2f}</h2><p>{rel_change:.2f}%</p></div>", unsafe_allow_html=True)
    mcol3.markdown(f"<div class='metric-card'><h3>🎯 Odhad směru</h3><h2>{prob*100:.1f}%</h2></div>", unsafe_allow_html=True)

    if hist_acc is not None:
        st.write(f"📊 Historická přesnost: **{hist_acc*100:.1f}%** (na {windows} oknech)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Cena', line=dict(color='#00FFD1')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA9', line=dict(color='#FFD166', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21', line=dict(color='#6A4C93', dash='dot')))
    fig.add_trace(go.Scatter(x=future_index, y=preds, mode='lines', name='Predikce', line=dict(color='#FF6B6B', width=2, dash='dash')))
    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified", title=f"{symbol} — Predikce + indikátory")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Detail dat"):
        st.dataframe(df.tail(10))

    st.info("Tento AI asistent je pouze doplňkový analytický nástroj — nikoliv finanční poradenství.")
