import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="AI Traffic Risk System",
    page_icon="🚦",
    layout="wide"
)

# ---------------------------------
# Load Data
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("traffic.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

df = load_data()

# Create target
threshold = df['Vehicles'].mean()
df['High_Traffic'] = df['Vehicles'].apply(lambda x: 1 if x > threshold else 0)

# Train model
X = df[['Junction', 'Hour', 'DayOfWeek', 'Weekend']]
y = df['High_Traffic']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "🤖 Prediction", "📈 Forecast", "🔍 Data Explorer"]
)

# =================================
# 📊 DASHBOARD
# =================================
if page == "📊 Dashboard":

    st.title("🚦 AI Traffic Risk Analytics Dashboard")

    # KPI Cards
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Average Vehicles", round(df['Vehicles'].mean(), 2))
    col3.metric("Peak Vehicles", df['Vehicles'].max())

    st.divider()

    # Charts
    st.subheader("📈 Traffic Analysis")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Hourly Traffic Pattern**")
        hourly_avg = df.groupby('Hour')['Vehicles'].mean()
        st.line_chart(hourly_avg)

    with colB:
        st.markdown("**Junction Comparison**")
        junction_avg = df.groupby('Junction')['Vehicles'].mean()
        st.bar_chart(junction_avg)

    st.divider()

    # Peak Insights
    peak_hour = hourly_avg.idxmax()
    peak_junction = junction_avg.idxmax()

    st.subheader("🚨 Key Insights")
    st.info(f"Peak traffic occurs around **Hour {peak_hour}**.")
    st.info(f"Most congested location: **Junction {peak_junction}**.")

# =================================
# 🤖 PREDICTION PAGE
# =================================
elif page == "🤖 Prediction":

    st.title("🤖 Traffic Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        junction = st.selectbox("Select Junction", sorted(df['Junction'].unique()))
        hour = st.slider("Select Hour", 0, 23, 12)

    with col2:
        day = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        weekend = 1 if day >= 5 else 0

    if st.button("🚀 Predict Risk", use_container_width=True):

        input_data = np.array([[junction, hour, day, weekend]])
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("🔍 Risk Assessment")

        # Multi-level risk
        if prob < 0.4:
            st.success(f"🟢 LOW RISK ({prob*100:.1f}% probability)")
            st.info("Traffic is expected to be smooth. Normal monitoring is sufficient.")
        elif prob < 0.7:
            st.warning(f"🟡 MEDIUM RISK ({prob*100:.1f}% probability)")
            st.warning("Moderate congestion possible. Consider traffic monitoring.")
        else:
            st.error(f"🔴 HIGH RISK ({prob*100:.1f}% probability)")
            st.error("Heavy congestion expected. Traffic control recommended.")

        # Confidence bar
        st.markdown("### 📊 Model Confidence")
        st.progress(int(prob * 100))

# =================================
# 📈 FORECAST PAGE
# =================================
elif page == "📈 Forecast":

    st.title("📈 Traffic Forecast (Next 24 Hours)")

    df_sorted = df.sort_values("DateTime").set_index("DateTime")
    ts = df_sorted['Vehicles'].resample('H').mean()

    try:
        model_arima = ARIMA(ts, order=(2,1,2))
        model_fit = model_arima.fit()
        forecast = model_fit.forecast(steps=24)

        st.line_chart(pd.concat([ts.tail(48), forecast]))

        st.success("Forecast generated successfully.")

    except Exception as e:
        st.warning("Forecast model is stabilizing. Try again.")

# =================================
# 🔍 DATA EXPLORER
# =================================
elif page == "🔍 Data Explorer":

    st.title("🔍 Traffic Data Explorer")

    st.dataframe(df.head(500))

    st.download_button(
        label="📥 Download Dataset",
        data=df.to_csv(index=False),
        file_name="traffic_data.csv",
        mime="text/csv"
    )

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption("Built by Tanushree Rathod | AI/ML Traffic Intelligence System 🚀")
