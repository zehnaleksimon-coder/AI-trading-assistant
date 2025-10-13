import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="Trading AI Beta", layout="wide")

st.title("📊 Trading AI – Live Beta v1.4")

symbol = st.text_input("Zadej symbol (např. AAPL, TSLA, BTC-USD):", "BTC-USD")
period = st.selectbox("Vyber časový rámec", ["1d", "5d", "7d", "1mo"])
interval = st.selectbox("Vyber interval", ["15m", "30m", "1h", "4h"])
future_hours = st.slider("Kolik hodin dopředu predikovat", 1, 24, 6)
refresh_rate = st.slider("Jak často obnovovat data (v sekundách)", 10, 60, 15)

placeholder = st.empty()

def load_data(symbol, period="7d", interval="1h"):
    data = yf.download(symbol, period=period, interval=interval)
    return data["Close"]

def make_prediction(series, future_steps):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression().fit(X, y)
    X_future = np.arange(len(series), len(series) + future_steps).reshape(-1, 1)
    y_future = model.predict(X_future)
    return y_future

while True:
    with placeholder.container():
        series = load_data(symbol, period=period, interval=interval)
        if len(series) > 5:
            y_future = make_prediction(series, future_hours)
            future_index = pd.date_range(series.index[-1], periods=future_hours+1, freq=interval)[1:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name='Historie'))
            fig.add_trace(go.Scatter(x=future_index, y=y_future, mode='lines',
                                     name='Predikce', line=dict(dash='dot', color='orange')))
            fig.update_layout(title=f"Predikce pro {symbol}",
                              xaxis_title="Čas",
                              yaxis_title="Cena",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"🔄 Poslední aktualizace: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.warning("Málo dat pro výpočet.")
        
        time.sleep(refresh_rate)
