import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🚦 Traffic Risk Dashboard")

df = pd.read_csv("traffic.csv")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

threshold = df['Vehicles'].mean()
df['High_Traffic'] = df['Vehicles'].apply(lambda x: 1 if x > threshold else 0)

X = df[['Junction','Hour','DayOfWeek','Weekend']]
y = df['High_Traffic']

model = RandomForestClassifier()
model.fit(X,y)

junction = st.selectbox("Junction", sorted(df['Junction'].unique()))
hour = st.slider("Hour", 0, 23)
day = st.slider("Day of Week", 0, 6)
weekend = 1 if day >= 5 else 0

if st.button("Predict"):
    pred = model.predict([[junction,hour,day,weekend]])[0]
    if pred == 1:
        st.error("High Traffic Risk")
    else:
        st.success("Low Traffic Risk")
