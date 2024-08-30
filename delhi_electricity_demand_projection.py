import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set page config
st.set_page_config(page_title="Delhi Power Demand Forecast", layout="wide")

# Custom CSS to make it colorful with a white theme
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 24px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        color: #31333F;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #ffffff;
        border-top: 2px solid #ff4b4b;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #ffffff;
        border-radius: 0px 0px 4px 4px;
        padding: 16px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .problem-statement {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .problem-statement h3 {
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    # Using Yahoo Finance to get temperature data for Delhi (you might want to replace this with actual power demand data)
    data = yf.download("BHEL.NS", start="2010-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    df["y"] = df["y"] * 100  # Scaling up the values to simulate MW
    return df

data = load_data()

# Forecasting functions
def prophet_forecast(data, horizon):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return forecast

def arima_forecast(data, horizon):
    model = ARIMA(data['y'], order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=horizon)
    return pd.DataFrame({'ds': pd.date_range(start=data['ds'].iloc[-1] + pd.Timedelta(days=1), periods=horizon),
                         'yhat': forecast})

def holt_winters_forecast(data, horizon):
    model = ExponentialSmoothing(data['y'], seasonal_periods=7, trend='add', seasonal='add')
    results = model.fit()
    forecast = results.forecast(horizon)
    return pd.DataFrame({'ds': pd.date_range(start=data['ds'].iloc[-1] + pd.Timedelta(days=1), periods=horizon),
                         'yhat': forecast})

# Streamlit app
st.title("🔌 Delhi Electricity Demand Projection")

# Sidebar for algorithm selection and parameters
st.sidebar.header("⚙️ Forecast Settings")
algorithm = st.sidebar.selectbox("Select Algorithm", ["Prophet", "ARIMA", "Holt-Winters"])
window = st.sidebar.slider("Training Window (days)", 30, 365, 180)
horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 90, 30)

# Filter data based on window
data_window = data.iloc[-window:]

# Generate forecast based on selected algorithm
if algorithm == "Prophet":
    forecast = prophet_forecast(data_window, horizon)
elif algorithm == "ARIMA":
    forecast = arima_forecast(data_window, horizon)
else:  # Holt-Winters
    forecast = holt_winters_forecast(data_window, horizon)

# Tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs(["📋 Problem Statement", "📈 Forecast", "🌡️ Weather Impact", "🏙️ Urban Growth", "💡 Recommendations"])

with tab0:
    st.header("Problem Statement")
    
    st.markdown("""
    <div class="problem-statement">
        <h3>Problem Statement ID: 1624</h3>
        <h3>Title: To develop an Artificial Intelligence (AI) based model for electricity demand projection including peak demand projection for Delhi Power system</h3>
        
        <h4>Background:</h4>
        <p>The load profile of power requirement in NCT of Delhi is highly peculiar. We are witnessing huge load variations during the winter and summer months and also during day and night during the same 24-hour window. This causes imbalance in matching the requisite power purchase with the electricity demand.</p>
        
        <h4>Key Points:</h4>
        <ul>
            <li>Peak load in Delhi touched 8300 MW this summer while the minimum load during winters goes as low as 2000 MW.</li>
            <li>Peak occurs twice during summer: first during daytime (around 15:30 hrs) and second at night (after 23:00 hrs).</li>
            <li>Solar generation creates a Duck-curve effect, with +/- 15% variation allowed by CERC.</li>
            <li>Uneven load growth across the city, with higher growth in upcoming areas.</li>
            <li>Load curve is highly peaky due to predominantly domestic and commercial loads, with minimal industrial load.</li>
            <li>Minimal agricultural load, unlike other states where it provides stability to the load curve.</li>
        </ul>
        
        <h4>Expected Solution:</h4>
        <p>An Artificial Intelligence based model with suitable compensation methodology to factor in:</p>
        <ul>
            <li>Weather effects (temperature, humidity, wind speed, rains/showers)</li>
            <li>Public holidays / weekly holidays</li>
            <li>Natural load growth</li>
            <li>Real estate development</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="problem-statement">
        <h3>Organization: Government of NCT of Delhi</h3>
        <h3>Department: IT Department, GNCTD</h3>
        <h3>Category: Software</h3>
        <h3>Theme: Smart Automation</h3>
    </div>
    """, unsafe_allow_html=True)

with tab1:
    st.header("Demand Forecast")
    
    # Plot forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_window['ds'], y=data_window['y'], mode='lines', name='Historical', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#ff7f0e')))
    fig.update_layout(title=f"Delhi Power Demand Forecast ({algorithm})", xaxis_title="Date", yaxis_title="Demand (MW)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show forecast statistics
    st.subheader("Forecast Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Forecasted Demand", f"{forecast['yhat'].mean():.0f} MW")
    col2.metric("Peak Forecasted Demand", f"{forecast['yhat'].max():.0f} MW")
    col3.metric("Minimum Forecasted Demand", f"{forecast['yhat'].min():.0f} MW")

with tab2:
    st.header("Weather Impact on Demand")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", 0, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    
    with col2:
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
        is_holiday = st.checkbox("Is it a holiday?")
    
    # Simulate weather impact
    base_demand = forecast['yhat'].iloc[-1]
    temp_factor = 50 * (temperature - 25)
    humidity_factor = 10 * (humidity - 50)
    wind_factor = -5 * wind_speed
    holiday_factor = -500 if is_holiday else 0
    
    weather_impact = base_demand + temp_factor + humidity_factor + wind_factor + holiday_factor
    
    # Display impact
    st.metric("Estimated Demand", f"{weather_impact:.0f} MW", f"{weather_impact - base_demand:.0f} MW")
    
    # Weather impact chart
    impact_data = pd.DataFrame({
        "Factor": ["Temperature", "Humidity", "Wind", "Holiday"],
        "Impact (MW)": [temp_factor, humidity_factor, wind_factor, holiday_factor]
    })
    fig_impact = px.bar(impact_data, x="Factor", y="Impact (MW)", title="Weather Impact on Demand",
                        color="Impact (MW)", color_continuous_scale="RdYlBu")
    st.plotly_chart(fig_impact)

with tab3:
    st.header("Urban Growth Scenarios")
    
    growth_rate = st.slider("Annual Growth Rate (%)", 0.0, 10.0, 5.0)
    years = st.slider("Projection Years", 1, 20, 5)
    
    # Calculate growth scenarios
    current_demand = forecast['yhat'].iloc[-1]
    conservative_growth = current_demand * (1 + growth_rate/200) ** years
    moderate_growth = current_demand * (1 + growth_rate/100) ** years
    aggressive_growth = current_demand * (1 + growth_rate/50) ** years
    
    # Display growth scenarios
    col1, col2, col3 = st.columns(3)
    col1.metric("Conservative Growth", f"{conservative_growth:.0f} MW", f"{conservative_growth - current_demand:.0f} MW")
    col2.metric("Moderate Growth", f"{moderate_growth:.0f} MW", f"{moderate_growth - current_demand:.0f} MW")
    col3.metric("Aggressive Growth", f"{aggressive_growth:.0f} MW", f"{aggressive_growth - current_demand:.0f} MW")
    
    # Growth scenario chart
    growth_data = pd.DataFrame({
        "Year": range(years + 1),
        "Conservative": [current_demand * (1 + growth_rate/200) ** i for i in range(years + 1)],
        "Moderate": [current_demand * (1 + growth_rate/100) ** i for i in range(years + 1)],
        "Aggressive": [current_demand * (1 + growth_rate/50) ** i for i in range(years + 1)]
    })
    fig_growth = px.line(growth_data, x="Year", y=["Conservative", "Moderate", "Aggressive"],
                         title="Urban Growth Scenarios", color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"])
    st.plotly_chart(fig_growth)

with tab4:
    st.header("Recommendations")
    
    recommendations = [
        ("Demand Response Programs", "Implement smart metering and time-of-use pricing to encourage off-peak consumption.", 300),
        ("Renewable Energy Integration", "Increase solar and wind capacity to match daytime peak demands and reduce reliance on conventional sources.", 500),
        ("Energy Storage Solutions", "Invest in battery storage systems to manage the duck curve effect and improve grid stability.", 400),
        ("Grid Modernization", "Upgrade transmission and distribution infrastructure to handle increasing peak demands and integrate smart grid technologies.", 200),
        ("Energy Efficiency Initiatives", "Promote energy-efficient appliances and building standards to reduce overall demand.", 350)
    ]
    
    for title, description, savings in recommendations:
        expander = st.expander(title)
        with expander:
            st.write(description)
            st.metric("Potential Savings", f"{savings} MW")
    
    # Potential savings chart
    savings_data = pd.DataFrame(recommendations, columns=["Initiative", "Description", "Potential Savings (MW)"])
    fig_savings = px.bar(savings_data, x="Initiative", y="Potential Savings (MW)", 
                         title="Potential Savings from Recommendations",
                         color="Potential Savings (MW)", color_continuous_scale="Viridis")
    st.plotly_chart(fig_savings)

# Display filtered data statistics
st.sidebar.subheader("Data Statistics")
st.sidebar.metric("Average Demand", f"{data_window['y'].mean():.0f} MW")
st.sidebar.metric("Peak Demand", f"{data_window['y'].max():.0f} MW")
st.sidebar.metric("Minimum Demand", f"{data_window['y'].min():.0f} MW")

# Add a footer
st.markdown("---")
st.markdown("Developed for the Government of NCT of Delhi | IT Department, GNCTD")