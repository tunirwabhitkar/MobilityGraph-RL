import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="MobilityGraph-RL Dashboard", layout="wide")

st.title("MobilityGraph-RL: Urban Mobility & Fleet Rebalancing")

st.sidebar.header("Controls")
zone_id = st.sidebar.selectbox("Select Zone", ["0", "1", "2", "3", "4"])
forecast_horizon = st.sidebar.selectbox("Forecast Horizon", ["15m", "30m", "1h", "3h"])

if st.sidebar.button("Get Forecast"):
    try:
        res = requests.post("http://localhost:8000/forecast", json={
            "zone_id": zone_id,
            "forecast_horizon": forecast_horizon
        })
        if res.status_code == 200:
            data = res.json()
            st.metric(label=f"Predicted Demand (Zone {zone_id})", value=int(data['predicted_demand']))
            st.write(f"95% Confidence Interval: [{data['lower_bound']:.1f}, {data['upper_bound']:.1f}]")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

st.header("City Overview")
st.write("Visualizations will appear here based on real-time simulated data.")

# Mock data for demonstration
if st.button("Simulate Rebalancing"):
    num_zones = 5
    demand = np.random.randint(50, 150, num_zones).tolist()
    supply = np.random.randint(50, 150, num_zones).tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current State")
        df = pd.DataFrame({'Zone': range(num_zones), 'Demand': demand, 'Supply': supply})
        df['Gap'] = df['Demand'] - df['Supply']
        st.dataframe(df)
        
    with col2:
        try:
            res = requests.post("http://localhost:8000/optimize-rebalancing", json={
                "demand": demand,
                "supply": supply
            })
            if res.status_code == 200:
                data = res.json()
                st.subheader("Optimization Results")
                st.metric("Total Cost", f"${data['total_cost']:.2f}")
                st.metric("Unmet Demand", int(data['unmet_demand']))
                st.write("Relocation Matrix:")
                st.write(np.array(data['relocations']))
        except Exception as e:
            st.error("Optimization failed.")
