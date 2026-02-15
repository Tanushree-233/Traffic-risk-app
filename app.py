limport streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Traffic Risk Intelligence",
    page_icon="🚦",
    layout="wide"
)

# -------------------------------------------------
# PREMIUM CSS
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("traffic.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

df = load_data()

threshold = df['Vehicles'].mean()
df['High_Traffic'] = df['Vehicles'].apply(lambda x: 1 if x > threshold else 0)

# -------------------------------------------------
# WEATHER FUNCTION
# -------------------------------------------------
@st.cache_data(ttl=600)
def get_weather():
    try:
        API_KEY = "96aa0d8b1be301aae6e4951317f6eddc"  # optional
        city = "Mumbai"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()
        return res["main"]["temp"], res["weather"][0]["main"]
    except:
        return None, "Unavailable"

# -------------------------------------------------
# MODEL
# -------------------------------------------------
X = df[['Junction','Hour','DayOfWeek','Weekend']]
y = df['High_Traffic']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("🚦 Traffic Intelligence")

if st.sidebar.button("🔄 Retrain Model"):
    model.fit(X, y)
    st.sidebar.success("Model retrained!")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🤖 Prediction", "📈 Forecast", "🔍 Data"]
)

# =================================================
# DASHBOARD
# =================================================
if page == "📊 Dashboard":

    st.markdown("<h1 style='text-align:center;'>🚦 AI Traffic Intelligence Dashboard</h1>", unsafe_allow_html=True)

    # Weather
    temp, weather = get_weather()
    st.caption(f"🌦️ Mumbai Weather: {weather} | 🌡️ {temp if temp else '--'}°C")

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Average Vehicles", round(df['Vehicles'].mean(),2))
    c3.metric("Peak Vehicles", df['Vehicles'].max())

    st.divider()

    # Charts
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### ⏰ Hourly Pattern")
        hourly_avg = df.groupby('Hour')['Vehicles'].mean()
        st.line_chart(hourly_avg)

    with colB:
        st.markdown("#### 📍 Junction Comparison")
        junction_avg = df.groupby('Junction')['Vehicles'].mean()
        st.bar_chart(junction_avg)

    # Multi-junction comparison
    st.subheader("📍 Multi-Junction Analysis")
    selected = st.multiselect(
        "Select Junctions",
        sorted(df['Junction'].unique()),
        default=[sorted(df['Junction'].unique())[0]]
    )
    multi = df[df['Junction'].isin(selected)].groupby('Hour')['Vehicles'].mean()
    st.line_chart(multi)

    # Map
    st.subheader("🗺️ Traffic Hotspot Map")
    map_df = pd.DataFrame({
        "lat": [19.0760, 19.0820, 19.0900],
        "lon": [72.8777, 72.8850, 72.8900],
    })

    m = folium.Map(location=[19.0760, 72.8777], zoom_start=11)
    for _, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color="red",
            fill=True,
        ).add_to(m)

    st_folium(m, width=700)

# =================================================
# PREDICTION
# =================================================
elif page == "🤖 Prediction":

    st.markdown("<h1 style='text-align:center;'>🤖 Smart Risk Predictor</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        junction = st.selectbox("📍 Junction", sorted(df['Junction'].unique()))
        hour = st.slider("⏰ Hour", 0, 23, 12)

    with col2:
        day = st.slider("📅 Day of Week", 0, 6, 2)
        weekend = 1 if day >= 5 else 0

    if st.button("🚀 Analyze Traffic Risk", use_container_width=True):

        input_data = np.array([[junction, hour, day, weekend]])
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("## 🔍 Risk Assessment")

        if prob < 0.4:
            st.success(f"🟢 LOW RISK — {prob*100:.1f}%")
            recommendation = "Traffic flow expected to be smooth."
        elif prob < 0.7:
            st.warning(f"🟡 MEDIUM RISK — {prob*100:.1f}%")
            recommendation = "Moderate congestion possible."
        else:
            st.error(f"🔴 HIGH RISK — {prob*100:.1f}%")
            recommendation = "Deploy traffic control measures."

        # ALERT SYSTEM
        if prob > 0.75:
            st.error("🚨 LIVE ALERT: Heavy congestion expected!")
        elif prob > 0.5:
            st.warning("⚠️ Traffic advisory issued.")
        else:
            st.info("✅ Traffic normal.")

        st.progress(int(prob*100))

        st.markdown(f"""
        <div class="glass">
        💡 <b>Recommendation:</b><br>
        {recommendation}
        </div>
        """, unsafe_allow_html=True)

# =================================================
# FORECAST
# =================================================
elif page == "📈 Forecast":

    st.markdown("<h1 style='text-align:center;'>📈 Traffic Forecast</h1>", unsafe_allow_html=True)

    df_sorted = df.sort_values("DateTime").set_index("DateTime")
    ts = df_sorted['Vehicles'].resample('H').mean()

    try:
        model_arima = ARIMA(ts, order=(2,1,2))
        model_fit = model_arima.fit()
        forecast = model_fit.forecast(steps=24)

        st.line_chart(pd.concat([ts.tail(48), forecast]))

        # Forecast vs actual
        st.subheader("📊 Actual vs Forecast")
        comparison = pd.DataFrame({
            "Actual": ts.tail(24).values,
            "Forecast": forecast.values[:24]
        })
        st.line_chart(comparison)

        st.success("✅ Forecast generated.")

    except:
        st.warning("⚠ Forecast model warming up.")

# =================================================
# DATA
# =================================================
elif page == "🔍 Data":

    st.markdown("<h1 style='text-align:center;'>🔍 Data Explorer</h1>", unsafe_allow_html=True)

    st.dataframe(df.head(500))

    st.download_button(
        "📥 Download Dataset",
        df.to_csv(index=False),
        "traffic_data.csv",
        "text/csv"
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("✨ Built by Tanushree Rathod | Ultra Premium AI Traffic Intelligence System")
