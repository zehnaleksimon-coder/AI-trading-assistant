# app.py (opraveno)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.set_page_config(page_title="Trading AI Assistant v3.0", layout="wide")
st.title("🚀 Trading AI Assistant — Beta 3.0 (oprava)")
st.markdown("Interaktivní graf + ensemble predikce (EMA, RSI, slope, linreg). "
            "Predikce je asistent — vždy dělej vlastní rozhodnutí.")

# -------------------------
# Helpers / indikátory
# -------------------------
@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str, interval: str):
    """Stáhne data přes yfinance a vrátí DataFrame s indexem datetime."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[['Close']].dropna()
    return df

def add_indicators(df: pd.DataFrame):
    df = df.copy()
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    # RSI 14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df['RSI'] = 100 - (100 / (1 + rs))
    # slope: jednoduchý proxy (průměr pct_change)
    df['slope_6'] = df['Close'].pct_change().rolling(6).mean()
    return df

def linreg_predict(series: pd.Series, steps: int):
    """Simple linear regression extrapolation (returns numpy array)."""
    if len(series) < 3:
        return np.array([float(series.iat[-1])] * steps)
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression().fit(X, y)
    future_idx = np.arange(len(series), len(series) + steps).reshape(-1, 1)
    preds = model.predict(future_idx)
    return preds

# Ensemble signal for one point (returns score in [-1,1])
def ensemble_score_for_point(window: pd.Series, indicators: pd.DataFrame):
    """
    window: last train_window closing prices (pd.Series)
    indicators: same index dataframe with EMA9, EMA21, RSI, slope_6
    returns: score (positive -> up, negative -> down)
    """
    # safety checks
    if window is None or len(window) < 3:
        return 0.0
    if indicators is None or indicators.shape[0] < 1:
        return 0.0

    # linear slope from linear regression on window (normalized)
    try:
        X = np.arange(len(window)).reshape(-1,1)
        lr = LinearRegression().fit(X, window.values)
        slope = lr.coef_[0] / (float(window.iat[-1]) + 1e-12)  # relative slope
    except Exception:
        slope = 0.0

    # use last row of indicators safely
    last_ind = indicators.iloc[-1]
    ema9 = float(last_ind.get('EMA9', np.nan)) if not pd.isna(last_ind.get('EMA9', np.nan)) else None
    ema21 = float(last_ind.get('EMA21', np.nan)) if not pd.isna(last_ind.get('EMA21', np.nan)) else None
    rsi = float(last_ind.get('RSI', np.nan)) if not pd.isna(last_ind.get('RSI', np.nan)) else None
    slope_proxy = float(last_ind.get('slope_6', 0.0)) if not pd.isna(last_ind.get('slope_6', np.nan)) else 0.0

    # EMA crossover
    ema_signal = 0
    if ema9 is not None and ema21 is not None:
        ema_signal = 1 if ema9 > ema21 else -1

    # RSI signal
    rsi_signal = 0
    if rsi is not None:
        if rsi < 35:
            rsi_signal = 1
        elif rsi > 65:
            rsi_signal = -1
        else:
            rsi_signal = 0

    # slope proxy
    slope_proxy_signal = 1 if slope_proxy > 0 else -1 if slope_proxy < 0 else 0

    # weighted sum (weights chosen heuristically)
    score = (0.45 * np.tanh(slope * 1000)) + (0.25 * ema_signal) + (0.2 * rsi_signal) + (0.1 * slope_proxy_signal)
    # clip to [-1,1]
    return float(max(-1.0, min(1.0, score)))

def rolling_backtest_ensemble(series: pd.Series, indicators_df: pd.DataFrame, train_window:int, horizon:int):
    """
    Evaluate directional accuracy of the ensemble:
    For each window, compute ensemble score at train end -> predict direction -> compare with actual
    Return accuracy (0..1) and number of evaluated windows.
    """
    acc = []
    n = len(series)
    # iterate windows
    for start in range(0, n - train_window - horizon):
        train = series.iloc[start:start+train_window]
        test_target = series.iloc[start+train_window:start+train_window+horizon]
        ind_slice = indicators_df.iloc[start:start+train_window]

        # safety checks
        if len(train) < train_window:
            continue
        if ind_slice.isna().any().any():
            continue
        if len(test_target) < 1:
            continue

        score = ensemble_score_for_point(train, ind_slice)
        pred_dir = 1 if score > 0 else 0

        # make sure to compare scalars
        try:
            test_val = float(test_target.iat[-1])
            train_val = float(train.iat[-1])
        except Exception:
            continue

        actual_dir = 1 if test_val > train_val else 0
        acc.append(1 if pred_dir == actual_dir else 0)

        # limit windows to speed
        if len(acc) >= 200:
            break

    if acc:
        return (sum(acc)/len(acc)), len(acc)
    else:
        return None, 0

def calibrate_probability(raw_score: float, hist_acc: float):
    """
    Convert raw ensemble score and historical accuracy to a probability [0,1].
    We use hist_acc (if available) as base and add confidence from score magnitude.
    """
    base = hist_acc if hist_acc is not None else 0.5
    # confidence from score magnitude
    conf = (abs(raw_score)) if raw_score is not None else 0.0  # 0..1
    # combine: weighted average giving more weight to historical accuracy
    # we also bias small amount by the sign of raw_score
    sign = 1 if (raw_score if raw_score is not None else 0.0) > 0 else -1
    # compute a small directional boost
    directional = 0.5 + 0.25 * conf * sign
    p = 0.7*base + 0.3*directional
    # map to [0.01,0.99]
    return float(max(0.01, min(0.99, p)))

# -------------------------
# UI
# -------------------------
col1, col2, col3 = st.columns([3,2,2])
with col1:
    symbol = st.text_input("Symbol (např. BTC-USD, ETH-USD, AAPL):", value="BTC-USD")
with col2:
    period = st.selectbox("Období dat:", ["7d","30d","90d","1y"])
with col3:
    interval = st.selectbox("Interval:", ["15m","30m","1h","4h"])

horizon_hours = st.slider("Predikce dopředu (hodin):", min_value=1, max_value=24, value=6)
train_window = st.slider("Velikost trénovacího okna (počet bodů):", min_value=8, max_value=200, value=48)

st.markdown("---")
run = st.button("🔄 Aktualizovat / Spočítat predikci")

# -------------------------
# Main logic
# -------------------------
if run:
    df = fetch_data(symbol, period=period, interval=interval)
    if df.empty:
        st.error("Nepodařilo se stáhnout data. Zkontroluj symbol/interval/period.")
        st.stop()

    df = add_indicators(df)

    # ensure enough points
    if len(df) < train_window + 2:
        st.error("Málo dat pro zvolený train_window. Zvol kratší okno nebo delší periodu.")
        st.stop()

    # Ensure indicators are not all NaN
    if df[['EMA9','EMA21','RSI','slope_6']].isna().all().all():
        st.error("Indikátory obsahují pouze NaN — vyčkej nebo změň periodu/interval.")
        st.stop()

    # ensemble score on last window
    last_window = df['Close'].iloc[-train_window:]
    last_ind = df[['EMA9','EMA21','RSI','slope_6']].iloc[-train_window:]
    raw_score = ensemble_score_for_point(last_window, last_ind)

    # historical backtest accuracy
    hist_acc, windows = rolling_backtest_ensemble(df['Close'], df[['EMA9','EMA21','RSI','slope_6']], train_window, horizon=1)

    # predicted price via linreg
    future_steps = horizon_hours
    preds = linreg_predict(df['Close'], future_steps)
    # build future index (assume same spacing)
    if len(df.index) >= 2:
        delta = df.index[-1] - df.index[-2]
    else:
        delta = pd.Timedelta(hours=1)
    future_index = [df.index[-1] + (i+1)*delta for i in range(future_steps)]
    pred_price = float(preds[-1])
    last_price = float(df['Close'].iat[-1])
    rel_change = (pred_price - last_price) / (last_price + 1e-12) * 100

    # calibrated probability
    prob = calibrate_probability(raw_score, hist_acc)

    # -------------------------
    # Display metrics + chart
    # -------------------------
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Poslední cena", f"{last_price:.6f}")
    mcol2.metric(f"Predikce za {horizon_hours}h", f"{pred_price:.6f}", delta=f"{rel_change:.2f}%")
    mcol3.metric("Odhad šance (direkce)", f"{prob*100:.1f} %")

    # show also historical accuracy
    if hist_acc is not None:
        st.write(f"Historická směrová přesnost (rolling, windows={windows}): **{hist_acc*100:.1f}%**")
    else:
        st.write("Historická přesnost: N/A (nedostatek oken)")

    # Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Cena', line=dict(color='#00CC96')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA9', line=dict(color='#FFD166', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21', line=dict(color='#6A4C93', dash='dot')))

    fig.add_trace(go.Scatter(x=future_index, y=preds, mode='lines', name='Predikce', line=dict(color='#FF6B6B', width=2, dash='dash')))
    # uncertainty band (light)
    low = preds * 0.998
    high = preds * 1.002
    fig.add_trace(go.Scatter(x=future_index, y=low, mode='lines', name='Uncertainty low', line=dict(color='rgba(255,107,107,0.2)'), showlegend=False))
    fig.add_trace(go.Scatter(x=future_index, y=high, mode='lines', name='Uncertainty high', line=dict(color='rgba(255,107,107,0.2)'), fill='tonexty', fillcolor='rgba(255,107,107,0.12)', showlegend=False))

    fig.update_layout(title=f"{symbol} — cena + predikce ({horizon_hours}h)", xaxis_title="Čas", yaxis_title="Cena", hovermode="x unified", template="plotly_dark", height=600)
    fig.update_traces(hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Cena: %{y:.6f}<extra></extra>' )

    st.plotly_chart(fig, use_container_width=True)

    # extra: show last few rows for debugging
    with st.expander("Detail dat (poslední řádky)"):
        st.dataframe(df.tail(10))

    st.info("Tato predikce je asistent. Nejedná se o investiční poradenství — uživatel nese odpovědnost za obchodní rozhodnutí.")
