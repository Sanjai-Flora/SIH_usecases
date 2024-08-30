import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set page config
st.set_page_config(page_title="Smart AI Traffic Management System", layout="wide")

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
        color: #0066cc;
        font-family: 'Arial', sans-serif;
    }
    .stAlert {
        background-color: #cce6ff;
        border: 1px solid #0066cc;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #e6f3ff;
        border-left: 5px solid #0066cc;
        padding: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }
    .stSelectbox {
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data
@st.cache_data
def generate_data(days=30):
    np.random.seed(42)
    date_range = pd.date_range(end=datetime.now(), periods=days*24*60, freq='T')
    
    directions = ['North', 'South', 'East', 'West']
    data = []
    
    for date in date_range:
        for direction in directions:
            hour = date.hour
            weekday = date.weekday()
            
            # Simulate traffic patterns
            base_traffic = np.random.poisson(20)  # Base traffic level
            time_factor = 1 + 0.5 * np.sin(np.pi * hour / 12)  # Time of day factor
            weekday_factor = 1 if weekday < 5 else 0.7  # Weekday vs weekend factor
            
            traffic_volume = int(base_traffic * time_factor * weekday_factor)
            
            data.append({
                'timestamp': date,
                'direction': direction,
                'traffic_volume': traffic_volume,
                'current_green_time': np.random.randint(30, 120)
            })
    
    return pd.DataFrame(data)

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('traffic_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('traffic_data.csv', index=False)
    return df

# Train AI model for green time prediction
@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df[['direction', 'traffic_volume']], columns=['direction'])
    X['hour'] = df['timestamp'].dt.hour
    X['weekday'] = df['timestamp'].dt.weekday
    y = df['current_green_time']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))

    return model, mae, X.columns

# Main application
def main():
    st.title("Smart AI-based Traffic Management System")

    # Load data and train model
    df = load_data()
    model, model_mae, feature_names = train_model(df)

    # Sidebar for date range selection
    st.sidebar.title("Settings")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )

    # Filter data based on date range
    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
    filtered_df = df.loc[mask]

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Traffic Analysis", "AI Predictions", "Real-time Simulation"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("Smart AI-based Traffic Management System")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1607<br>
        <strong>Organization:</strong> Department of Science and Technology<br>
        <strong>Department:</strong> Department of Science and Technology<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Smart Automation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the need for an intelligent traffic management system that can adapt to real-time
        traffic conditions. The system aims to:

        1. Monitor traffic volumes from multiple directions in real-time.
        2. Analyze traffic patterns and identify congestion points.
        3. Use AI to predict optimal traffic light timings based on current conditions.
        4. Dynamically adjust traffic light timings to improve overall traffic flow.
        5. Provide insights and visualizations to traffic management personnel.

        By leveraging this AI-based system, urban areas can significantly reduce traffic congestion, minimize wait times,
        and improve overall traffic efficiency.
        """)

    with tab2:
        st.header("Traffic Analysis")

        # Overall traffic volume
        st.subheader("Overall Traffic Volume")
        daily_traffic = filtered_df.groupby(filtered_df['timestamp'].dt.date)['traffic_volume'].sum().reset_index()
        fig_daily = px.line(daily_traffic, x='timestamp', y='traffic_volume', title="Daily Traffic Volume")
        st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The daily traffic volume chart shows overall trends and patterns in traffic flow.
        This can help identify busy days and long-term trends in traffic volume.
        </div>
        """, unsafe_allow_html=True)

        # Traffic by direction
        st.subheader("Traffic by Direction")
        direction_traffic = filtered_df.groupby(['timestamp', 'direction'])['traffic_volume'].sum().reset_index()
        fig_direction = px.line(direction_traffic, x='timestamp', y='traffic_volume', color='direction',
                                title="Traffic Volume by Direction")
        st.plotly_chart(fig_direction, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Analyzing traffic by direction helps identify which routes experience the heaviest traffic.
        This information can be used to prioritize traffic light timing adjustments.
        </div>
        """, unsafe_allow_html=True)

        # Heatmap of traffic patterns
        st.subheader("Traffic Patterns Heatmap")
        heatmap_data = filtered_df.groupby([filtered_df['timestamp'].dt.hour, 'direction'])['traffic_volume'].mean().unstack()
        fig_heatmap = px.imshow(heatmap_data, title="Average Traffic Volume by Hour and Direction",
                                labels=dict(x="Direction", y="Hour of Day", color="Traffic Volume"))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The heatmap visualizes traffic patterns throughout the day for each direction.
        This helps in identifying peak hours and planning traffic light timings accordingly.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("AI Predictions")

        # Model performance
        st.subheader("Model Performance")
        st.metric("Mean Absolute Error", f"{model_mae:.2f} seconds")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Feature Importance for Green Time Prediction")
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The feature importance chart shows which factors most influence the AI's decision
        on traffic light timings. This can help in understanding the model's behavior and improving its accuracy.
        </div>
        """, unsafe_allow_html=True)

        # Predicted vs Actual Green Time
        st.subheader("Predicted vs Actual Green Time")
        X_test = pd.get_dummies(filtered_df[['direction', 'traffic_volume']], columns=['direction'])
        X_test['hour'] = filtered_df['timestamp'].dt.hour
        X_test['weekday'] = filtered_df['timestamp'].dt.weekday
        y_test = filtered_df['current_green_time']
        y_pred = model.predict(X_test)

        fig_pred_vs_actual = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Green Time', 'y': 'Predicted Green Time'},
                                        title="Predicted vs Actual Green Time")
        fig_pred_vs_actual.add_trace(go.Scatter(x=[0, max(y_test)], y=[0, max(y_test)], mode='lines', name='Ideal Prediction'))
        st.plotly_chart(fig_pred_vs_actual, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This scatter plot compares the AI's predicted green light timings with the actual timings.
        Points closer to the diagonal line indicate more accurate predictions.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("Real-time Traffic Simulation")

        # Simulated real-time traffic management
        st.subheader("Traffic Light Control Simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_direction = st.selectbox("Select Direction", ['North', 'South', 'East', 'West'])
        with col2:
            current_traffic = st.slider("Current Traffic Volume", 0, 100, 50)

        if st.button("Simulate Traffic Conditions"):
            # Prepare input for prediction
            input_data = pd.DataFrame({
                'traffic_volume': [current_traffic],
                'hour': [datetime.now().hour],
                'weekday': [datetime.now().weekday()]
            })
            
            # Add direction columns
            for direction in ['North', 'South', 'East', 'West']:
                input_data[f'direction_{direction}'] = 1 if direction == selected_direction else 0

            # Reorder columns to match the training data
            input_data = input_data.reindex(columns=feature_names, fill_value=0)

            # Make prediction
            predicted_green_time = model.predict(input_data)[0]

            # Display results
            st.success(f"Recommended Green Light Time: {predicted_green_time:.0f} seconds")

            # Visualization of traffic flow
            fig_traffic_flow = go.Figure(data=[
                go.Bar(name='Current Traffic', x=[selected_direction], y=[current_traffic]),
                go.Bar(name='Green Time', x=[selected_direction], y=[predicted_green_time])
            ])
            fig_traffic_flow.update_layout(title="Traffic Volume vs Green Time", barmode='group')
            st.plotly_chart(fig_traffic_flow, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> This simulation demonstrates how the AI adjusts green light timings based on
            current traffic conditions. The system aims to balance traffic flow across all directions while minimizing wait times.
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This AI-based traffic management system provides real-time analysis and recommendations for
    optimizing traffic flow. The system continuously learns from new data to improve its predictions and adapt
    to changing traffic patterns. Regular updates with actual traffic data will enhance the accuracy and
    effectiveness of the AI model.
    """)

if __name__ == "__main__":
    main()