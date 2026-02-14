# 🚦 AI-Based Traffic Risk Intelligence System

An end-to-end AI/ML project that predicts traffic congestion risk using historical traffic data, machine learning, and time-series forecasting.  
Built with **Streamlit**, **Random Forest**, and **ARIMA** to provide real-time insights and intelligent traffic recommendations.

---

## 🌟 Live Demo
Deployed on Streamlit Community Cloud (https://traffic-risk-app-vbjd8rojfissbdxc2zymdu.streamlit.app/#key-insights).

---

## 📌 Project Overview

Urban traffic congestion is a major challenge in smart city planning.  
This project uses machine learning and analytics to:

- Predict high traffic risk at junctions
- Analyze hourly traffic patterns
- Forecast future traffic volume
- Provide actionable traffic management recommendations

---

## 🚀 Key Features

✅ Interactive AI-powered dashboard  
✅ Multi-level risk prediction (Low / Medium / High)  
✅ Model confidence visualization  
✅ Traffic pattern analytics  
✅ ARIMA-based 24-hour forecast  
✅ Smart traffic recommendations  
✅ Data explorer with download option  
✅ Premium glass-morphism UI  

---

## 🧠 Machine Learning Approach

### 🔹 Classification Model
- **Algorithm:** Random Forest Classifier  
- **Target:** High Traffic (binary)  
- **Features Used:**
  - Junction
  - Hour
  - Day of Week
  - Weekend flag

### 🔹 Time Series Forecasting
- **Model:** ARIMA (2,1,2)  
- **Purpose:** Predict next 24 hours traffic volume

---

## 📊 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Statsmodels (ARIMA)
- Matplotlib

---

## 📂 Project Structure

```
traffic-risk-app/
│
├── app.py              # Main Streamlit application
├── traffic.csv         # Dataset
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation (Local Run)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 How It Works

1. Load and preprocess traffic dataset  
2. Train Random Forest model  
3. User selects junction, hour, and day  
4. Model predicts traffic risk probability  
5. Dashboard visualizes trends and forecast  

---

## 🎯 Business Impact

- Helps traffic authorities plan congestion control  
- Identifies peak traffic hours  
- Supports smart city traffic management  
- Enables proactive traffic monitoring  

---

## 🔮 Future Enhancements

- Real-time traffic API integration  
- Geospatial traffic heatmaps  
- Deep learning (LSTM) forecasting  
- Multi-city scalability  
- Model explainability (SHAP dashboard)

---

## 👩‍💻 Author

**Tanushree Rathod**  
AI/ML & Data Analytics Enthusiast

---

⭐ If you like this project, consider giving it a star!
