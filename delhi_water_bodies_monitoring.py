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
st.set_page_config(page_title="Delhi Water Bodies Monitoring System", layout="wide")

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
def generate_data(days=365):
    water_bodies = {
        'Yamuna River': (28.6139, 77.2090),
        'Najafgarh Lake': (28.5084, 76.9850),
        'Bhalswa Lake': (28.7459, 77.1609),
        'Sanjay Lake': (28.6139, 77.3031),
        'Hauz Khas Lake': (28.5494, 77.2001)
    }
    pollutants = ['Dissolved Oxygen', 'pH', 'Turbidity', 'Phosphates', 'Nitrates', 'E. coli', 'Heavy Metals']
    
    data = []
    for _ in range(20000):  # Generate 20000 records
        date = datetime.now() - timedelta(days=np.random.randint(0, days))
        water_body = np.random.choice(list(water_bodies.keys()))
        lat, lon = water_bodies[water_body]
        pollutant = np.random.choice(pollutants)
        level = np.random.uniform(0, 10)
        rainfall = np.random.uniform(0, 50)
        temperature = np.random.uniform(10, 40)
        
        data.append({
            'date': date,
            'water_body': water_body,
            'latitude': lat,
            'longitude': lon,
            'pollutant': pollutant,
            'level': level,
            'rainfall': rainfall,
            'temperature': temperature
        })
    
    return pd.DataFrame(data)

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('water_bodies_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Check if latitude and longitude columns exist, if not, add them
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            water_bodies = {
                'Yamuna River': (28.6139, 77.2090),
                'Najafgarh Lake': (28.5084, 76.9850),
                'Bhalswa Lake': (28.7459, 77.1609),
                'Sanjay Lake': (28.6139, 77.3031),
                'Hauz Khas Lake': (28.5494, 77.2001)
            }
            df['latitude'] = df['water_body'].map(lambda x: water_bodies[x][0])
            df['longitude'] = df['water_body'].map(lambda x: water_bodies[x][1])
            
            # Save the updated DataFrame
            df.to_csv('water_bodies_data.csv', index=False)
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('water_bodies_data.csv', index=False)
    return df

# Train predictive model
@st.cache_resource
def train_model(df):
    # Create dummy variables for categorical features
    X = pd.get_dummies(df[['water_body', 'pollutant']], columns=['water_body', 'pollutant'])
    X['rainfall'] = df['rainfall']
    X['temperature'] = df['temperature']
    y = df['level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))

    return model, mae

# Calculate Water Quality Index
def calculate_wqi(row):
    weights = {'Dissolved Oxygen': 0.3, 'pH': 0.2, 'Turbidity': 0.2, 'Phosphates': 0.1, 'Nitrates': 0.1, 'E. coli': 0.05, 'Heavy Metals': 0.05}
    return sum(row[pollutant] * weights[pollutant] for pollutant in weights if pollutant in row)

# Main application
def main():
    st.title("Delhi Water Bodies Monitoring System")

    # Load data and train model
    df = load_data()
    model, model_mae = train_model(df)

    # Sidebar for date range selection
    st.sidebar.title("Settings")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['date'].min().date(), df['date'].max().date()),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )

    # Filter data based on date range
    mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
    filtered_df = df.loc[mask]

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Problem Statement", "Overview", "Water Body Analysis", "Environmental Factors", "Predictive Analytics"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("Online real-time survey and monitoring of water bodies in Delhi")
        
        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1619<br>
        <strong>Organization:</strong> Government of NCT of Delhi<br>
        <strong>Department:</strong> IT Department, GNCTD<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Miscellaneous
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        This application addresses the critical need for a technological solution to conduct real-time surveys
        and monitoring of water bodies in Delhi. The system aims to:
        
        1. Provide comprehensive real-time data on water quality across Delhi's major water bodies.
        2. Track multiple pollutants and environmental factors affecting water quality.
        3. Analyze trends and patterns in water pollution over time and across different locations.
        4. Predict potential pollution events using advanced analytics.
        5. Support decision-making for water body rejuvenation efforts.
        
        By leveraging this system, the Government of NCT of Delhi can effectively monitor, analyze, and
        improve the health of Delhi's water bodies, contributing to overall environmental sustainability.
        """)

    with tab2:
        st.header("Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Measurements", len(filtered_df))
        with col2:
            st.metric("Avg. Pollutant Level", f"{filtered_df['level'].mean():.2f}")
        with col3:
            st.metric("Water Bodies Monitored", filtered_df['water_body'].nunique())
        with col4:
            st.metric("Pollutants Tracked", filtered_df['pollutant'].nunique())

        # Pollutant levels over time
        fig_levels = px.line(filtered_df.groupby(['date', 'pollutant'])['level'].mean().reset_index(),
                             x='date', y='level', color='pollutant', title="Average Pollutant Levels Over Time")
        fig_levels.update_layout(colorway=px.colors.qualitative.Bold)
        st.plotly_chart(fig_levels, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Monitor trends in pollutant levels over time to identify seasonal patterns
        and long-term changes in water quality across Delhi's water bodies.
        </div>
        """, unsafe_allow_html=True)

        # Map of water bodies
        st.subheader("Water Bodies in Delhi")
        water_body_locations = filtered_df.groupby('water_body').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()

        fig_map = go.Figure(go.Scattermapbox(
            lat=water_body_locations['latitude'],
            lon=water_body_locations['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(size=9),
            text=water_body_locations['water_body']
        ))

        fig_map.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=28.6139, lon=77.2090),
                zoom=10
            ),
            showlegend=False,
            height=600
        )

        st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        st.header("Water Body Analysis")

        # Pollutant levels by water body
        fig_levels_by_body = px.box(filtered_df, x='water_body', y='level', color='pollutant',
                                    title="Pollutant Levels by Water Body")
        fig_levels_by_body.update_layout(colorway=px.colors.qualitative.Set2)
        st.plotly_chart(fig_levels_by_body, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Compare pollutant levels across different water bodies to identify
        those that may require immediate intervention or targeted cleanup efforts.
        </div>
        """, unsafe_allow_html=True)

        # Water quality index
        wqi_df = filtered_df.pivot(index=['date', 'water_body'], columns='pollutant', values='level').reset_index()
        wqi_df['WQI'] = wqi_df.apply(calculate_wqi, axis=1)

        fig_wqi = px.line(wqi_df, x='date', y='WQI', color='water_body', title="Water Quality Index Over Time")
        fig_wqi.update_layout(colorway=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_wqi, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The Water Quality Index provides a comprehensive measure of overall water quality.
        Track this index to assess the effectiveness of pollution control measures and prioritize interventions.
        </div>
        """, unsafe_allow_html=True)

        # Pollution hotspots
        st.subheader("Pollution Hotspots")
        hotspots = filtered_df.groupby('water_body')['level'].mean().sort_values(ascending=False)
        fig_hotspots = px.bar(hotspots, x=hotspots.index, y='level', title="Average Pollution Levels by Water Body")
        fig_hotspots.update_layout(colorway=[px.colors.sequential.Reds[-1]])
        st.plotly_chart(fig_hotspots, use_container_width=True)

    with tab4:
        st.header("Environmental Factors")

        # Correlation between environmental factors and pollutant levels
        corr_df = filtered_df.groupby('date').agg({
            'level': 'mean',
            'rainfall': 'mean',
            'temperature': 'mean'
        }).reset_index()

        fig_corr = px.scatter_matrix(corr_df, dimensions=['level', 'rainfall', 'temperature'],
                                     title="Correlation Between Pollutant Levels and Environmental Factors")
        fig_corr.update_layout(colorway=[px.colors.sequential.Viridis[4]])
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Analyze the relationships between environmental factors and pollutant levels
        to understand how rainfall and temperature impact water quality in Delhi's water bodies.
        </div>
        """, unsafe_allow_html=True)

        # Seasonal patterns
        filtered_df['month'] = filtered_df['date'].dt.month
        monthly_avg = filtered_df.groupby(['month', 'pollutant'])['level'].mean().reset_index()

        fig_seasonal = px.line(monthly_avg, x='month', y='level', color='pollutant',
                               title="Seasonal Patterns in Pollutant Levels")
        fig_seasonal.update_xaxes(tickmode='array', tickvals=list(range(1, 13)),
                                  ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        fig_seasonal.update_layout(colorway=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_seasonal, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Identifying seasonal patterns in pollutant levels can help in planning
        targeted interventions and allocating resources more effectively throughout the year.
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.header("Predictive Analytics")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Feature Importance for Pollutant Level Prediction")
        fig_importance.update_layout(colorway=[px.colors.sequential.Viridis[-1]])
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Understanding which factors most influence pollutant levels can guide
        pollution control strategies and help prioritize monitoring efforts.
        </div>
        """, unsafe_allow_html=True)


        # Predictive model performance
        st.metric("Model Performance (Mean Absolute Error)", f"{model_mae:.2f}")

        # Pollutant level prediction
        st.subheader("Pollutant Level Prediction")
        col1, col2 = st.columns(2)
        with col1:
            pred_water_body = st.selectbox("Select Water Body", df['water_body'].unique())
            pred_pollutant = st.selectbox("Select Pollutant", df['pollutant'].unique())
        with col2:
            pred_rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 25.0)
            pred_temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)

        if st.button("Predict Pollutant Level"):
            input_data = pd.DataFrame({
                'water_body': [pred_water_body],
                'pollutant': [pred_pollutant],
                'rainfall': [pred_rainfall],
                'temperature': [pred_temperature]
            })
            
            # Create dummy variables for water_body and pollutant
            input_encoded = pd.get_dummies(input_data, columns=['water_body', 'pollutant'])
            
            # Ensure all columns from the training data are present
            for column in model.feature_names_in_:
                if column not in input_encoded.columns:
                    input_encoded[column] = 0
            
            # Select only the columns used during training
            input_encoded = input_encoded[model.feature_names_in_]

            predicted_level = model.predict(input_encoded)[0]
            st.success(f"Predicted pollutant level: {predicted_level:.2f}")

            # Visualization of prediction
            fig_prediction = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = predicted_level,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Predicted {pred_pollutant} Level"},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 3.33], 'color': "lightgreen"},
                        {'range': [3.33, 6.66], 'color': "yellow"},
                        {'range': [6.66, 10], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 8}}))
            st.plotly_chart(fig_prediction)

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> Use these predictions to anticipate pollutant levels under various
            environmental conditions. This can help in proactive management of water quality and early
            warning systems for potential pollution events.
            </div>
            """, unsafe_allow_html=True)

        # Additional feature: Trend Analysis
        st.header("Trend Analysis")
        trend_pollutant = st.selectbox("Select Pollutant for Trend Analysis", df['pollutant'].unique())
        trend_df = filtered_df[filtered_df['pollutant'] == trend_pollutant].groupby('date')['level'].mean().reset_index()
        trend_df['MA'] = trend_df['level'].rolling(window=7).mean()

        fig_trend = px.line(trend_df, x='date', y=['level', 'MA'], 
                            title=f"Trend Analysis for {trend_pollutant}")
        fig_trend.update_layout(yaxis_title="Pollutant Level", xaxis_title="Date")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Trend analysis helps identify long-term patterns and potential improvements or
        deteriorations in water quality. Use this information to assess the effectiveness of pollution control measures.
        </div>
        """, unsafe_allow_html=True)

        # Additional feature: Comparative Analysis
        st.header("Comparative Analysis")
        col1, col2 = st.columns(2)
        with col1:
            comp_water_body1 = st.selectbox("Select First Water Body", df['water_body'].unique(), key='wb1')
        with col2:
            comp_water_body2 = st.selectbox("Select Second Water Body", df['water_body'].unique(), key='wb2')

        comp_df = filtered_df[filtered_df['water_body'].isin([comp_water_body1, comp_water_body2])]
        comp_df = comp_df.groupby(['date', 'water_body'])['level'].mean().reset_index()

        fig_comp = px.line(comp_df, x='date', y='level', color='water_body',
                        title=f"Comparative Analysis: {comp_water_body1} vs {comp_water_body2}")
        fig_comp.update_layout(yaxis_title="Average Pollutant Level", xaxis_title="Date")
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Comparative analysis allows for direct comparison of water quality between different
        water bodies. Use this to identify which water bodies may require more attention or to assess the relative
        success of different pollution control strategies.
        </div>
        """, unsafe_allow_html=True)

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Note:** This system provides real-time monitoring and analysis of water bodies in Delhi.
        Use these insights to optimize resource allocation, identify pollution hotspots, and improve
        overall water quality. Regular updates with actual field data will enhance the accuracy and
        usefulness of this tool.
        """)

        # Recommendations based on analysis
        st.header("Recommendations")
        st.markdown("""
        Based on the analysis provided by this monitoring system, consider the following recommendations:

        1. **Prioritize Intervention:** Focus immediate cleanup efforts on water bodies with consistently high pollutant levels.
        2. **Seasonal Planning:** Adjust pollution control measures based on observed seasonal patterns in pollutant levels.
        3. **Environmental Management:** Implement strategies to mitigate the impact of rainfall and temperature on water quality.
        4. **Predictive Action:** Use the predictive model to anticipate and prevent potential pollution events.
        5. **Comparative Improvements:** Learn from water bodies with better water quality indices and apply successful strategies to others.
        6. **Trend-based Goals:** Set water quality improvement goals based on observed trends and work towards consistent long-term improvements.
        7. **Public Awareness:** Use the visualizations and insights from this system to educate the public about water quality issues and improvement efforts.
        8. **Continuous Monitoring:** Regularly update the system with real-time data to ensure ongoing accuracy and relevance of insights.
        """)

if __name__ == "__main__":
    main()