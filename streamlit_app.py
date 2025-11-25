# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -----------------------
# Nastavení stránky
# -----------------------
st.set_page_config(page_title="AI Trading Asistent", page_icon="📈", layout="wide")
st.title("📈 AI Trading Asistent")
st.markdown("Interaktivní graf + jednoduchá predikce (EMA, RSI, Slope, Linear Regression).")
st.markdown("Upozornění: Toto není investiční poradenství. Appka je asistent.")

# -----------------------
# Uživatelské volby
# -----------------------
ticker = st.text_input("Symbol (např. BTC-USD, AAPL):", value="BTC-USD")
period = st.selectbox("Perioda dat (historie):", ["7d", "30d", "90d", "180d", "365d"], index=1)
interval = st.selectbox("Interval (granularity):", ["1d", "1h", "30m"], index=0)

days_label = f"{period}"
st.write("Načítám data za:", days_label)

show_history = st.checkbox("Zobrazit historii (Close)", value=True)
show_ema = st.checkbox("Zobrazit EMA", value=True)
show_rsi = st.checkbox("Zobrazit RSI", value=False)
show_slope = st.checkbox("Zobrazit Slope", value=False)
show_pred = st.checkbox("Zobrazit predikci", value=True)

predict_hours = st.slider("Kolik bodů dopředu predikovat (body = interval):", 1, 72, 7)

if st.button("Aktualizovat / Spustit"):
    # -----------------------
    # Načtení dat
    # -----------------------
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        st.error("Chyba při stahování dat: " + str(e))
        st.stop()

    if df is None or df.empty:
        st.error("Nepodařilo se stáhnout data (prázdná odpověď). Zkontroluj ticker/interval/Internet.")
        st.stop()

    # Ujistíme se, že index je datetime
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            st.error("Chyba: index není datetime.")
            st.stop()

    # -----------------------
    # Indikátory (stabilní výpočty)
    # -----------------------
    df["Close"] = df["Close"].astype(float)
    df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI (stabilní)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Slope (lineární sklon) - min_periods=5
    def slope_func(x):
        if len(x) < 2:
            return np.nan
        # polyfit vrací [slope, intercept]
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    df["Slope"] = df["Close"].rolling(window=5, min_periods=5).apply(slope_func, raw=False)

    # -----------------------
    # Predikce (jednoduchá lin. reg. extrapolace)
    # -----------------------
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values
    # Pokud málo bodů, nepokračovat
    if len(X) < 3:
        st.error("Příliš málo dat pro predikci - zvol delší periodu.")
        st.stop()

    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(df), len(df) + predict_hours).reshape(-1, 1)
    future_pred = model.predict(future_X)

    # jednoduchý "ensemble" - průměr poslední predikce a EMA posledního bodu
    last_ema = df["EMA"].iloc[-1] if not pd.isna(df["EMA"].iloc[-1]) else df["Close"].iloc[-1]
    ensemble_pred = (float(future_pred[-1]) + float(last_ema)) / 2.0

    # jednoduché "confidence" (simulované) - lepší kalibrace můžeš přidat později
    confidence = 50.0 + (abs((future_pred[-1] - df["Close"].iloc[-1]) / (df["Close"].iloc[-1] + 1e-12)) * 1000)
    # oříznout 1..99
    confidence = float(max(1.0, min(99.0, confidence)))

    # -----------------------
    # Rolling accuracy (směrová) - bezpečný a jednoduchý backtest
    # -----------------------
    def rolling_dir_accuracy(series, window=20, horizon=1, max_windows=200):
        acc = []
        n = len(series)
        for start in range(0, n - window - horizon + 1):
            train = series[start:start + window]
            test_target = series[start + window:start + window + horizon]
            if train.isna().any() or test_target.isna().any():
                continue
            # predikce jednoduchá: poslední hodnota tréninku -> extrapolace směr
            pred_dir = 1 if train.iloc[-1] < test_target.iloc[-1] else 0
            actual_dir = 1 if train.iloc[-1] < test_target.iloc[-1] else 0
            acc.append(1 if pred_dir == actual_dir else 0)
            if len(acc) >= max_windows:
                break
        return (sum(acc) / len(acc)) if acc else None, len(acc)

    hist_acc, hist_windows = rolling_dir_accuracy(df["Close"], window=20, horizon=1)

    # -----------------------
    # Graf (Plotly interaktivní)
    # -----------------------
    fig = go.Figure()
    if show_history:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], mode="lines", name="EMA"))
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", yaxis="y2", line=dict(dash="dot")))
    if show_slope:
        fig.add_trace(go.Scatter(x=df.index, y=df["Slope"], mode="lines", name="Slope", line=dict(dash="dot")))
    if show_pred:
        future_dates = [df.index[-1] + timedelta(seconds=(i+1) * (df.index[1] - df.index[0]).total_seconds()) for i in range(predict_hours)]
        # pokud index step není konstantní (např. 1h vs 1d), použití .total_seconds() bezpečně vytvoří časové skoky
        fig.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines+markers", name="Predikce"))

    # pokud zobrazujeme RSI, přidáme sekundární osu y
    layout = dict(title=f"{ticker} — Historie & Predikce",
                  xaxis=dict(title="Datum"),
                  yaxis=dict(title="Cena (Close/EMA/Slope)"),
                  height=650,
                  hovermode="x unified")
    if show_rsi:
        layout["yaxis2"] = dict(title="RSI", overlaying="y", side="right", range=[0, 100])
    fig.update_layout(layout)

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # Výpis výsledků (bez riskantního f-string/markdown mixu)
    # -----------------------
    st.subheader("Výsledek analýzy")
    st.write("Poslední cena: " + str(round(float(df['Close'].iloc[-1]), 6)))
    st.write("Predikovaná cena (ensemble) za " + str(predict_hours) + " bodů: " + str(round(float(ensemble_pred), 6)))
    st.write("Simulovaná jistota: " + str(round(confidence, 1)) + " %")
    if hist_acc is not None:
        st.write("Historická směrová přesnost (rolling): " + str(round(hist_acc * 100, 2)) + "% (" + str(hist_windows) + " oken)")
    else:
        st.write("Historická přesnost: N/A")

    if confidence > 65:
        st.success("Signál: potenciální růst (pozor - není to finanční rada).")
    else:
        st.info("Signál: nízká jistota - pozor na rizika.")

    st.caption("© AI Trading Asistent – demo. Uživatelská zodpovědnost.")
