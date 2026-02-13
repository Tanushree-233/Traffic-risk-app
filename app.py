import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Traffic Risk Dashboard",
    page_icon="🚦",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🚦 AI-Based Traffic Risk Prediction System")
st.markdown("### Smart Traffic Analytics & Risk Detection")

# -----------------------------
# Load Data
# -----------------------------
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

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("📊 Traffic Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Average Vehicles", round(df['Vehicles'].mean(), 2))
col3.metric("Peak Vehicles", df['Vehicles'].max())

st.divider()

# -----------------------------
# Charts Section
# -----------------------------
st.subheader("📈 Traffic Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Hourly Traffic Pattern**")
    hourly_avg = df.groupby('Hour')['Vehicles'].mean()
    st.line_chart(hourly_avg)

with col2:
    st.markdown("**Junction Comparison**")
    junction_avg = df.groupby('Junction')['Vehicles'].mean()
    st.bar_chart(junction_avg)

st.divider()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Prediction Controls")

junction = st.sidebar.selectbox(
    "Select Junction",
    sorted(df['Junction'].unique())
)

hour = st.sidebar.slider("Select Hour", 0, 23, 12)
day = st.sidebar.slider("Day of Week (0=Mon,6=Sun)", 0, 6, 2)
weekend = 1 if day >= 5 else 0

# -----------------------------
# Model Training
# -----------------------------
X = df[['Junction','Hour','DayOfWeek','Weekend']]
y = df['High_Traffic']

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🤖 Traffic Risk Prediction")

if st.button("🚀 Predict Traffic Risk"):

    input_data = np.array([[junction, hour, day, weekend]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Risk display
    if prediction == 1:
        st.error(f"🔴 High Traffic Risk ({probability*100:.1f}% confidence)")
    else:
        st.success(f"🟢 Low Traffic Risk ({(1-probability)*100:.1f}% confidence)")

    # Confidence bar
    st.progress(float(probability))

st.markdown("---")
st.caption("Built with ❤️ by Tanushree | AI Traffic Risk Project")
