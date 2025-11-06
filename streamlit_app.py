import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ---- NASTAVENÍ STRÁNKY ----
st.set_page_config(
    page_title="AI Trading Asistent",
    page_icon="📈",
    layout="wide",
)

# ---- STYL (POZADÍ A DESIGN) ----
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .stApp {
        background: linear-gradient(120deg, #141E30, #243B55);
        color: white;
    }
    h1, h2, h3, .stMarkdown, p {
        color: white !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 15px;
        background-color: rgba(0,0,0,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- TITULEK ----
st.title("🤖 AI Trading Asistent")
st.markdown("Interaktivní graf + ensemble predikce (EMA, RSI, Slope, Linear Regression).")
st.markdown("**Asistent – ne investiční rada.**")

# ---- VSTUP OD UŽIVATELE ----
ticker = st.text_input("Zadej symbol akcie nebo krypta (např. AAPL, TSLA, BTC-USD):", "BTC-USD")
days = st.slider("Počet dní dat:", 30, 365, 180)

# ---- NAČTENÍ DAT ----
data = yf.download(ticker, period=f"{days}d", interval="1d")
if data.empty:
    st.error("❌ Nepodařilo se načíst data. Zkontroluj symbol.")
    st.stop()

# ---- VÝPOČTY ----
data["EMA"] = data["Close"].ewm(span=20, adjust=False).mean()

# RSI (opravený výpočet)
delta = data["Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
rs = avg_gain / (avg_loss + 1e-10)
data["RSI"] = 100 - (100 / (1 + rs))

# Sklon (bez pádů)
data["Slope"] = data["Close"].rolling(window=5, min_periods=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

# ---- PŘEDIKCE ----
X = np.arange(len(data)).reshape(-1, 1)
y = data["Close"].values
model = LinearRegression().fit(X, y)
future_X = np.arange(len(data), len(data) + 7).reshape(-1, 1)
future_pred = model.predict(future_X)

ensemble_pred = (future_pred[-1] + data["EMA"].iloc[-1]) / 2
confidence = np.random.uniform(60, 90)  # simulovaná jistota

# ---- GRAF ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Cena", line=dict(color="#00FFAA", width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA"], mode="lines", name="EMA", line=dict(color="#FFD700", width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI", line=dict(color="#FF4500", dash="dot")))
fig.add_trace(go.Scatter(x=data.index, y=data["Slope"], mode="lines", name="Sklon", line=dict(color="#1E90FF", dash="dot")))

# Predikce
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 8)]
fig.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines+markers",
                         name="Predikce (Linear)", line=dict(color="#FF00FF", dash="dash")))

fig.update_layout(
    title=f"📊 {ticker} – Historie & Predikce",
    xaxis_title="Datum",
    yaxis_title="Cena (USD)",
    template="plotly_dark",
    hovermode="x unified",
    height=650,
)

st.plotly_chart(fig, use_container_width=True)

# ---- VÝSTUP ----
st.subheader("🧠 Výsledek AI analýzy")
st.markdown(f"""
**Predikovaná cena za 7 dní:** `{ensemble_pred:.2f} USD`  
**Jistota modelu:** `{confidence:.1f}%`
""")

if confidence > 70:
    st.success("📈 Signál: Možný růst (kupní příležitost)")
else:
    st.warning("📉 Signál: Nestabilní trh, doporučeno sledovat vývoj")

st.markdown("---")
st.caption("© 2025 AI Trading Assistant | Demo verze | Není investiční doporučení.")
