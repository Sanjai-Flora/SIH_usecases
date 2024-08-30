import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page config
st.set_page_config(page_title="AI-ML based Agri-Horticultural Commodity Price Prediction", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #f0f8ff, #e6f3ff);
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

# Generate synthetic data
@st.cache_data
def generate_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    commodities = ['Gram', 'Tur', 'Urad', 'Moong', 'Masur', 'Onion', 'Potato', 'Tomato']
    
    data = []
    for commodity in commodities:
        base_price = np.random.uniform(30, 100)
        trend = np.random.uniform(-0.01, 0.02)
        seasonality = np.random.uniform(0.1, 0.3)
        noise = np.random.normal(0, 0.05, len(date_range))
        
        prices = base_price + trend * np.arange(len(date_range)) + seasonality * np.sin(2 * np.pi * np.arange(len(date_range)) / 365) + noise
        prices = np.maximum(prices, 1)  # Ensure prices are positive
        
        for date, price in zip(date_range, prices):
            data.append({
                'date': date,
                'commodity': commodity,
                'price': price
            })
    
    return pd.DataFrame(data)

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('commodity_price_data.csv')
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        start_date = datetime(2018, 1, 1)
        end_date = datetime.now()
        df = generate_data(start_date, end_date)
        df.to_csv('commodity_price_data.csv', index=False)
    return df

# Train and forecast using ARIMA
def arima_forecast(data, window, horizon):
    model = ARIMA(data[-window:], order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=horizon)
    return forecast

# Train and forecast using Holt-Winters
def holt_winters_forecast(data, window, horizon):
    if len(data) < 730:  # Less than 2 years of data
        # Use additive model without seasonal component
        model = ExponentialSmoothing(data[-window:], trend='add', seasonal=None)
    else:
        model = ExponentialSmoothing(data[-window:], seasonal_periods=365, trend='add', seasonal='add')
    results = model.fit()
    forecast = results.forecast(horizon)
    return forecast

# Train and forecast using Prophet
def prophet_forecast(data, window, horizon):
    df = pd.DataFrame({'ds': data.index[-window:], 'y': data[-window:]})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return forecast['yhat'][-horizon:]

# Main application
def main():
    st.title("AI-ML based Agri-Horticultural Commodity Price Prediction")

    # Load data
    df = load_data()

    # Sidebar for settings
    st.sidebar.title("Settings")
    selected_commodity = st.sidebar.selectbox("Select Commodity", df['commodity'].unique())
    selected_algorithm = st.sidebar.selectbox("Select Algorithm", ["ARIMA", "Holt-Winters", "Prophet"])
    
    # Calculate max window based on available data
    max_window = (df['date'].max() - df['date'].min()).days
    window = st.sidebar.slider("Training Window (days)", 30, max_window, min(180, max_window))
    horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

    # Filter data for selected commodity
    commodity_df = df[df['commodity'] == selected_commodity].sort_values('date')

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Price Analysis", "Price Forecasting", "Market Intervention"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("Development of AI-ML based models for predicting prices of agri-horticultural commodities")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1647<br>
        <strong>Organization:</strong> Ministry of Consumer Affairs, Food and Public Distribution<br>
        <strong>Department:</strong> Department of Consumer Affairs<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Agriculture, FoodTech & Rural Development
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the need for advanced price prediction models for essential food commodities. The system aims to:

        1. Monitor daily prices of 22 essential food commodities across 550 price reporting centres.
        2. Analyze price trends based on seasonality, historical data, and market intelligence.
        3. Implement multiple forecasting models including ARIMA, Holt-Winters, and Prophet.
        4. Assist in decision-making for market interventions and buffer stock management.
        5. Provide insights to stabilize price volatility in the market.

        By leveraging AI-ML models, the Department of Consumer Affairs can make more informed decisions on market interventions and buffer stock management.
        """)

    with tab2:
        st.header("Price Analysis")

        # Historical price trend
        st.subheader(f"Historical Price Trend for {selected_commodity}")
        fig_trend = px.line(commodity_df, x='date', y='price', title=f"{selected_commodity} Price Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This chart shows the historical price trend of the selected commodity.
        It helps in identifying long-term trends, seasonality, and potential anomalies in pricing.
        </div>
        """, unsafe_allow_html=True)

        # Price distribution
        st.subheader(f"Price Distribution for {selected_commodity}")
        fig_dist = px.histogram(commodity_df, x='price', nbins=30, title=f"{selected_commodity} Price Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The price distribution histogram provides an overview of the price ranges
        and their frequency. This can help in understanding the typical price ranges and identifying outliers.
        </div>
        """, unsafe_allow_html=True)

        # Seasonality analysis
        st.subheader("Seasonality Analysis")
        commodity_df['month'] = commodity_df['date'].dt.month
        monthly_avg = commodity_df.groupby('month')['price'].mean().reset_index()
        fig_seasonality = px.line(monthly_avg, x='month', y='price', title=f"Monthly Average Price for {selected_commodity}")
        fig_seasonality.update_xaxes(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        st.plotly_chart(fig_seasonality, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The seasonality analysis shows how prices typically vary across months.
        This information is crucial for understanding seasonal patterns and planning market interventions.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("Price Forecasting")

        # Prepare data for forecasting
        prices = commodity_df.set_index('date')['price']

        # Perform forecasting based on selected algorithm
        try:
            if selected_algorithm == "ARIMA":
                forecast = arima_forecast(prices, window, horizon)
            elif selected_algorithm == "Holt-Winters":
                forecast = holt_winters_forecast(prices, window, horizon)
            else:  # Prophet
                forecast = prophet_forecast(prices, window, horizon)

            # Prepare data for plotting
            actual_data = prices[-window:]
            forecast_dates = pd.date_range(start=actual_data.index[-1] + timedelta(days=1), periods=horizon)
            forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast})

            # Plotting results
            st.subheader(f"{selected_algorithm} Forecast for {selected_commodity}")
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=actual_data.index, y=actual_data, name='Historical Data'))
            fig_forecast.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast'], name='Forecast', line=dict(dash='dash')))
            fig_forecast.update_layout(title=f"{selected_algorithm} Forecast for {selected_commodity}", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Display forecast values
            st.subheader("Forecast Values")
            st.dataframe(forecast_df)

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> The forecast provides a prediction of future prices based on historical trends and patterns.
            Use this information to anticipate price movements and plan market interventions accordingly.
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during forecasting: {str(e)}")
            st.info("Please try adjusting the training window or selecting a different algorithm.")

    with tab4:
        st.header("Market Intervention Recommendations")

        # Simulated market intervention thresholds
        lower_threshold = np.percentile(commodity_df['price'], 25)
        upper_threshold = np.percentile(commodity_df['price'], 75)

        st.subheader("Price Thresholds for Market Intervention")
        col1, col2 = st.columns(2)
        col1.metric("Lower Threshold (25th percentile)", f"₹{lower_threshold:.2f}")
        col2.metric("Upper Threshold (75th percentile)", f"₹{upper_threshold:.2f}")

        # Current price and recommendation
        current_price = commodity_df['price'].iloc[-1]
        if current_price < lower_threshold:
            recommendation = "Consider releasing buffer stock to stabilize prices."
            color = "red"
        elif current_price > upper_threshold:
            recommendation = "Consider procuring more stock to build buffer and stabilize prices."
            color = "orange"
        else:
            recommendation = "Prices are within normal range. Continue monitoring."
            color = "green"

        st.markdown(f"""
        <div class="insight-box" style="border-left-color: {color};">
        <strong>Current Price:</strong> ₹{current_price:.2f}<br>
        <strong>Recommendation:</strong> {recommendation}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        These recommendations are based on historical price data and current market trends. 
        Always consider additional factors such as:
        - Crop production estimates
        - Weather forecasts
        - Global market trends
        - Government policies and regulations
        before making final decisions on market interventions.
        """)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This AI-ML based price prediction system provides insights based on
    historical data and multiple forecasting models. For the most accurate results, regularly update the
    system with the latest price data from all 550 price reporting centres. Always consult with
    market experts and consider additional factors before making final decisions on market interventions.
    """)

if __name__ == "__main__":
    main()