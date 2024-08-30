import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import io

# Set page config
st.set_page_config(page_title="Oil Spill Detection System", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #e6f3ff, #f0f8ff);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1e3f66;
        font-family: 'Arial', sans-serif;
    }
    .stAlert {
        background-color: #e6f3ff;
        border: 1px solid #1e3f66;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #e6f3ff;
        border-left: 5px solid #1e3f66;
        padding: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #1e3f66;
        color: white;
    }
    .stSelectbox {
        color: #1e3f66;
    }
</style>
""", unsafe_allow_html=True)

# Function to load AIS data
@st.cache_data
def load_ais_data():
    # For demonstration, we'll use synthetic data
    # In a real application, you would load data from the AIS APIs mentioned in the problem statement
    n_vessels = 100
    data = {
        'mmsi': np.arange(1, n_vessels + 1),
        'latitude': np.random.uniform(18.5, 19.5, n_vessels),
        'longitude': np.random.uniform(72.5, 73.5, n_vessels),
        'sog': np.random.uniform(0, 20, n_vessels),
        'cog': np.random.uniform(0, 360, n_vessels),
        'heading': np.random.uniform(0, 360, n_vessels),
        'vessel_type': np.random.choice(['Cargo', 'Tanker', 'Passenger', 'Fishing'], n_vessels),
        'timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 60)) for _ in range(n_vessels)]
    }
    return pd.DataFrame(data)

# Function to detect anomalies in AIS data
def detect_anomalies(df):
    # For demonstration, we'll use simple thresholds
    # In a real application, you would use more sophisticated anomaly detection algorithms
    anomalies = df[
        (df['sog'] > 15) |  # Unusually high speed
        (df['sog'] < 0.1) |  # Stopped vessel
        (abs(df['cog'] - df['heading']) > 45)  # Large discrepancy between course and heading
    ]
    return anomalies

# Function to simulate satellite data for oil spill detection
def simulate_satellite_data(lat, lon):
    # For demonstration, we'll randomly generate "oil spill" data
    # In a real application, you would process actual satellite imagery
    grid_size = 50
    lat_range = np.linspace(lat - 0.1, lat + 0.1, grid_size)
    lon_range = np.linspace(lon - 0.1, lon + 0.1, grid_size)
    xx, yy = np.meshgrid(lon_range, lat_range)
    
    # Simulate an oil spill as a Gaussian blob
    zz = np.exp(-((xx - lon)**2 + (yy - lat)**2) / 0.001)
    zz += np.random.normal(0, 0.1, zz.shape)  # Add some noise
    
    return xx, yy, zz

# Main application
def main():
    st.title("Oil Spill Detection System")

    # Load AIS data
    ais_data = load_ais_data()

    # Sidebar for settings
    st.sidebar.title("Settings")
    selected_area = st.sidebar.selectbox("Select Area", ["Option 1: Off Mumbai", "Option 2: Gulf of Mexico"])

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "AIS Data Analysis", "Anomaly Detection", "Oil Spill Detection"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("Detecting oil spills at marine environment using Automatic Identification System (AIS) and satellite datasets")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1655<br>
        <strong>Organization:</strong> Ministry of Earth Sciences<br>
        <strong>Department:</strong> Indian National Center for Ocean Information Services (INCOIS)<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Smart Automation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the critical need for automatic oil spill detection in marine environments. The system aims to:

        1. Integrate AIS data with satellite datasets for comprehensive monitoring.
        2. Detect anomalies in vessel behavior that may indicate distress or potential oil spills.
        3. Analyze satellite imagery to identify possible oil spills in areas of interest.
        4. Provide early warning for potential environmental hazards.
        5. Assist regulatory authorities in quick and efficient response to oil spills.

        By leveraging AIS data and satellite imagery, this system enhances our ability to protect marine environments and respond rapidly to potential oil spill incidents.
        """)

    with tab2:
        st.header("AIS Data Analysis")

        # Display AIS data
        st.subheader("AIS Data Overview")
        st.dataframe(ais_data)

        # Visualize vessel positions
        st.subheader("Vessel Positions")
        fig = px.scatter_mapbox(ais_data, lat="latitude", lon="longitude", color="vessel_type", hover_name="mmsi",
                                zoom=8, height=600)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This map shows the current positions of vessels based on AIS data.
        Different colors represent various vessel types, allowing for quick identification of vessel distribution and patterns.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("Anomaly Detection")

        # Detect anomalies
        anomalies = detect_anomalies(ais_data)

        st.subheader("Detected Anomalies")
        st.dataframe(anomalies)

        # Visualize anomalies
        st.subheader("Anomaly Map")
        fig = px.scatter_mapbox(anomalies, lat="latitude", lon="longitude", color="vessel_type", hover_name="mmsi",
                                zoom=8, height=600)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This map highlights vessels with anomalous behavior, such as unusual speed, sudden stops, or significant course deviations.
        These anomalies may indicate vessels in distress or potential oil spill risks.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("Oil Spill Detection")

        # Simulate oil spill detection for a selected anomaly
        if not anomalies.empty:
            selected_anomaly = st.selectbox("Select Anomaly for Oil Spill Analysis", anomalies['mmsi'])
            anomaly_data = anomalies[anomalies['mmsi'] == selected_anomaly].iloc[0]
            
            st.subheader(f"Simulated Satellite Data Analysis for MMSI {selected_anomaly}")
            
            xx, yy, zz = simulate_satellite_data(anomaly_data['latitude'], anomaly_data['longitude'])
            
            fig = go.Figure(data=go.Contour(x=xx[0], y=yy[:, 0], z=zz))
            fig.update_layout(title="Simulated Oil Spill Detection", xaxis_title="Longitude", yaxis_title="Latitude")
            st.plotly_chart(fig, use_container_width=True)

            # Determine if there's a significant oil spill
            if np.max(zz) > 0.7:  # Arbitrary threshold for demonstration
                st.warning("Potential oil spill detected! Alerting authorities...")
            else:
                st.success("No significant oil spill detected in this area.")

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> This contour plot simulates satellite imagery analysis for oil spill detection.
            Darker areas indicate higher likelihood of oil presence. The system automatically alerts authorities if a potential spill is detected.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No anomalies detected for oil spill analysis.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This oil spill detection system integrates AIS data with simulated satellite imagery for demonstration purposes.
    In a real-world application, it would use actual AIS feeds and process real satellite data for more accurate and timely detection of potential oil spills.
    Always consult with environmental experts and maritime authorities for verification and response planning.
    """)

if __name__ == "__main__":
    main()