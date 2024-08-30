import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Set page config
st.set_page_config(page_title="Certificate Issuance Monitoring System", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stAlert {
        background-color: #e6f3ff;
        border: 1px solid #1f77b4;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data
@st.cache_data
def generate_data(days=30):
    subdivisions = ['North', 'South', 'East', 'West', 'Central']
    certificate_types = ['Caste', 'Income', 'Domicile', 'Marriage', 'Birth', 'Death']
    
    data = []
    for _ in range(5000):  # Generate 5000 records
        date = datetime.now() - timedelta(days=np.random.randint(0, days))
        subdivision = np.random.choice(subdivisions)
        cert_type = np.random.choice(certificate_types)
        processing_time = np.random.randint(1, 30)
        status = np.random.choice(['Pending', 'Issued', 'Rejected'], p=[0.3, 0.6, 0.1])
        
        data.append({
            'date': date,
            'subdivision': subdivision,
            'certificate_type': cert_type,
            'processing_time': processing_time,
            'status': status
        })
    
    return pd.DataFrame(data)

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('certificate_data.csv')
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('certificate_data.csv', index=False)
    return df

# Train predictive model
@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df[['subdivision', 'certificate_type']], drop_first=True)
    y = df['processing_time']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    return model, mae

# Main application
def main():
    st.title("Certificate Issuance Monitoring System")

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
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Subdivision Analysis", "Resource Optimization", "Predictive Analytics"])

    with tab1:
        st.header("Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Applications", len(filtered_df))
        with col2:
            st.metric("Avg. Processing Time", f"{filtered_df['processing_time'].mean():.1f} days")
        with col3:
            st.metric("Pending Applications", len(filtered_df[filtered_df['status'] == 'Pending']))
        with col4:
            st.metric("Issuance Rate", f"{len(filtered_df[filtered_df['status'] == 'Issued']) / len(filtered_df):.1%}")

        # Application volume over time
        fig_volume = px.line(filtered_df.groupby('date').size().reset_index(name='count'),
                             x='date', y='count', title="Daily Application Volume")
        st.plotly_chart(fig_volume, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Monitor daily application volumes to identify trends and potential spikes in demand.
        This can help in proactive resource allocation and managing workload efficiently.
        </div>
        """, unsafe_allow_html=True)

        # Certificate type distribution
        fig_cert_types = px.pie(filtered_df, names='certificate_type', title="Certificate Type Distribution")
        st.plotly_chart(fig_cert_types, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Understanding the distribution of certificate types can help in specialized
        resource allocation and targeted process improvements for high-demand certificate types.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.header("Subdivision Analysis")

        # Workload by subdivision
        fig_workload = px.bar(filtered_df.groupby('subdivision').size().reset_index(name='count'),
                              x='subdivision', y='count', title="Workload by Subdivision")
        st.plotly_chart(fig_workload, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Identify subdivisions with high workloads that may require additional resources
        or process improvements to manage the demand effectively.
        </div>
        """, unsafe_allow_html=True)

        # Processing time by subdivision
        fig_proc_time = px.box(filtered_df, x='subdivision', y='processing_time',
                               title="Processing Time by Subdivision")
        st.plotly_chart(fig_proc_time, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Analyze processing time variations across subdivisions to identify
        areas that may need efficiency improvements or additional resources.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("Resource Optimization")

        # Simulated current resource allocation
        current_resources = pd.DataFrame({
            'subdivision': filtered_df['subdivision'].unique(),
            'staff': np.random.randint(5, 20, size=len(filtered_df['subdivision'].unique())),
            'workstations': np.random.randint(3, 15, size=len(filtered_df['subdivision'].unique()))
        })

        # Calculate efficiency metrics
        efficiency_metrics = filtered_df.groupby('subdivision').agg({
            'processing_time': 'mean',
            'date': 'count'
        }).reset_index()
        efficiency_metrics = efficiency_metrics.merge(current_resources, on='subdivision')
        efficiency_metrics['applications_per_staff'] = efficiency_metrics['date'] / efficiency_metrics['staff']

        # Visualize current resource allocation and efficiency
        fig_resources = px.scatter(efficiency_metrics, x='staff', y='date', size='applications_per_staff',
                                   color='subdivision', hover_name='subdivision',
                                   labels={'staff': 'Number of Staff', 'date': 'Number of Applications'},
                                   title="Resource Allocation and Efficiency by Subdivision")
        st.plotly_chart(fig_resources, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This chart helps identify imbalances in resource allocation. Subdivisions with high
        application counts but low staff numbers may need additional resources. Conversely, subdivisions with low
        application counts but high staff numbers might have excess capacity that could be reallocated.
        </div>
        """, unsafe_allow_html=True)

        # Recommend resource reallocation
        st.subheader("Resource Reallocation Recommendations")
        avg_apps_per_staff = efficiency_metrics['applications_per_staff'].mean()
        for _, row in efficiency_metrics.iterrows():
            if row['applications_per_staff'] > avg_apps_per_staff * 1.2:
                st.warning(f"{row['subdivision']} is overloaded. Consider adding {int(row['applications_per_staff'] / avg_apps_per_staff) - row['staff']} more staff.")
            elif row['applications_per_staff'] < avg_apps_per_staff * 0.8:
                st.info(f"{row['subdivision']} has excess capacity. Consider reallocating {row['staff'] - int(row['date'] / avg_apps_per_staff)} staff to busier subdivisions.")

    with tab4:
        st.header("Predictive Analytics")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Feature Importance for Processing Time Prediction")
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This chart shows which factors have the most influence on processing times.
        Focus on improving processes related to high-importance features to reduce overall processing times.
        </div>
        """, unsafe_allow_html=True)

        # Predictive model performance
        st.metric("Model Performance (Mean Absolute Error)", f"{model_mae:.2f} days")

        # Workload prediction
        st.subheader("Workload Prediction")
        pred_subdivision = st.selectbox("Select Subdivision", df['subdivision'].unique())
        pred_cert_type = st.selectbox("Select Certificate Type", df['certificate_type'].unique())

        if st.button("Predict Processing Time"):
            input_data = pd.DataFrame({
                'subdivision': [pred_subdivision],
                'certificate_type': [pred_cert_type]
            })
            input_encoded = pd.get_dummies(input_data, drop_first=True)
            for col in model.feature_names_in_:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model.feature_names_in_]
            
            predicted_time = model.predict(input_encoded)[0]
            st.success(f"Predicted processing time: {predicted_time:.2f} days")

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> Use these predictions to anticipate processing times for different types of
            applications in various subdivisions. This can help in managing citizen expectations and identifying
            areas where process improvements may be needed.
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This system provides real-time monitoring and analysis of certificate issuance processes.
    Use these insights to optimize resource allocation, identify bottlenecks, and improve overall efficiency.
    Regular updates and feedback from ground-level staff can further enhance the accuracy and usefulness of this tool.
    """)

if __name__ == "__main__":
    main()