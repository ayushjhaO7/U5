import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="DevOps AI: Crime Analysis Dashboard", layout="wide")

st.title("🛡️ Crime Analysis Distributed AI Dashboard (Unit V)")
st.markdown("""
This dashboard acts as the **Production Front-end** for your deployed AI Model. 
It communicates via a REST API (FastAPI) to pull real-time risk assessments from your Spark cluster.
""")

# Configuration
API_URL = "http://127.0.0.1:8000"

st.sidebar.header("Deployment Controls")
status_check = st.sidebar.button("Check API Status")

if status_check:
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            st.sidebar.success("✅ AI Service is Online")
        else:
            st.sidebar.error("❌ API Error")
    except Exception as e:
        st.sidebar.error(f"❌ Connection Failed: Have you run app.py?")

# Main Dashboard UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Compute Risk Level")
    state_input = st.text_input("Enter State/UT Name (e.g. BIHAR, GOA, MAHARASHTRA):", "BIHAR")
    if st.button("Predict Risk"):
        try:
            response = requests.get(f"{API_URL}/predict/{state_input}")
            if response.status_code == 200:
                data = response.json()
                st.info(f"AI assigned **{data['state']}** to **Cluster {data['cluster']}**.")
                
                # Visual Risk Mapping logic
                if data['cluster'] == 1:
                    st.error("🚨 HIGH RISK ZONE")
                elif data['cluster'] == 2:
                    st.warning("⚠️ MEDIUM RISK ZONE")
                else:
                    st.success("✅ LOW RISK ZONE")
            else:
                st.error("State not found!")
        except Exception:
            st.error("Connection Failed. Start app.py first.")

with col2:
    st.subheader("Historical Metadata (Source: clean_crime_data.csv)")
    try:
        df = pd.read_csv("../crime---analysis/data/cleaned_crime_data.csv")
        st.write("Preview of current active data being analyzed:")
        st.dataframe(df.head(10))
        
        # Plotting logic for a professional dashboard feel
        st.subheader("Analytics (Matplotlib/Seaborn)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df.groupby('YEAR')['crime_count'].sum().reset_index(), x='YEAR', y='crime_count', ax=ax, color='red')
        plt.title("Year-wise Crime Trend (Aggregated)")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Could not load historical data: {e}")

st.divider()
st.markdown("**Capstone Project 5: Distributed AI Implementation**")
st.text("Created as part of the DevOps for AI Unit - Utilizes PySpark, FastAPI, and Streamlit.")
