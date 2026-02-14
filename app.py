import streamlit as st
import pandas as pd
import numpy as np
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
# PREMIUM CSS (GLASS UI)
# -------------------------------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Glass cards */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Metric styling */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
}

/* Sidebar */
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

# Train model
X = df[['Junction','Hour','DayOfWeek','Weekend']]
y = df['High_Traffic']

model = RandomForestClassifier(random_state=42)
model.fit(X,y)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("🚦 Traffic Intelligence")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🤖 Prediction", "📈 Forecast", "🔍 Data"]
)

# =================================================
# DASHBOARD
# =================================================
if page == "📊 Dashboard":

    st.markdown("<h1 style='text-align:center;'>🚦 AI Traffic Intelligence Dashboard</h1>", unsafe_allow_html=True)

    # KPIs
    st.markdown("### 📊 Key Metrics")

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

    # Insights
    peak_hour = hourly_avg.idxmax()
    peak_junction = junction_avg.idxmax()

    st.markdown(f"""
    <div class="glass">
    🚨 <b>Peak Hour:</b> {peak_hour} <br>
    📍 <b>Most Congested Junction:</b> {peak_junction}
    </div>
    """, unsafe_allow_html=True)

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

        # Multi-level risk
        if prob < 0.4:
            st.success(f"🟢 LOW RISK — {prob*100:.1f}%")
            recommendation = "Traffic flow expected to be smooth."
        elif prob < 0.7:
            st.warning(f"🟡 MEDIUM RISK — {prob*100:.1f}%")
            recommendation = "Moderate congestion possible."
        else:
            st.error(f"🔴 HIGH RISK — {prob*100:.1f}%")
            recommendation = "Deploy traffic control measures."

        # Progress bar
        st.progress(int(prob*100))

        # Recommendation card
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
        st.success("✅ Forecast generated for next 24 hours.")

    except:
        st.warning("⚠ Forecast model warming up. Try again.")

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
