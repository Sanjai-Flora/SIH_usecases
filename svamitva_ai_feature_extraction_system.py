import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import io
from PIL import Image

# Set page config
st.set_page_config(page_title="SVAMITVA AI Feature Extraction System", layout="wide")

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
def generate_data(n_samples=1000):
    np.random.seed(42)

    # Generate features
    building_area = np.random.uniform(50, 500, n_samples)
    roof_type = np.random.choice(['RCC', 'Tiled', 'Tin', 'Others'], n_samples)
    road_length = np.random.uniform(10, 100, n_samples)
    waterbody_area = np.random.uniform(0, 1000, n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'building_area': building_area,
        'roof_type': roof_type,
        'road_length': road_length,
        'waterbody_area': waterbody_area,
        'timestamp': [datetime.now() - timedelta(days=i) for i in range(n_samples)]
    })

    return df

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('svamitva_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('svamitva_data.csv', index=False)
    return df

# Train classification model for roof type
@st.cache_resource
def train_model(df):
    X = df[['building_area', 'road_length', 'waterbody_area']]
    y = df['roof_type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

# Main application
def main():
    st.title("SVAMITVA AI Feature Extraction System")

    # Load data and train model
    df = load_data()
    model, model_accuracy = train_model(df)

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
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Data Analysis", "Feature Extraction", "Model Performance"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("Development and Optimization of AI model for Feature identification/Extraction from drone orthophotos")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1705<br>
        <strong>Organization:</strong> Ministry of Panchayati Raj<br>
        <strong>Department:</strong> Ministry of Panchayati Raj<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Robotics and Drones
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the need for an AI-powered system to extract features from drone orthophotos
        as part of the SVAMITVA Scheme. The system aims to:

        1. Develop an AI model for identifying key features in orthophotos with high precision.
        2. Extract building footprints and classify roof types (RCC, Tiled, Tin, Others).
        3. Extract road features and waterbodies.
        4. Achieve a target accuracy of 95% in feature identification.
        5. Optimize the model for efficient processing and deployment.

        By leveraging this system, the Ministry of Panchayati Raj can enhance the SVAMITVA Scheme's efficiency
        in creating accurate land records and facilitating rural development.
        """)

    with tab2:
        st.header("Data Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Observations", len(filtered_df))
        with col2:
            st.metric("Average Building Area", f"{filtered_df['building_area'].mean():.2f} sq m")
        with col3:
            st.metric("Total Road Length", f"{filtered_df['road_length'].sum():.2f} km")

        # Feature distribution visualization
        st.subheader("Feature Distribution")
        feature_to_plot = st.selectbox("Select Feature", ['building_area', 'road_length', 'waterbody_area'])
        
        fig = px.histogram(filtered_df, x=feature_to_plot, color='roof_type',
                           title=f"Distribution of {feature_to_plot.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The histogram shows the distribution of selected features across different roof types.
        This can help identify patterns in building characteristics and land use.
        </div>
        """, unsafe_allow_html=True)

        # Roof type distribution
        st.subheader("Roof Type Distribution")
        roof_type_counts = filtered_df['roof_type'].value_counts()
        fig_pie = px.pie(values=roof_type_counts.values, names=roof_type_counts.index,
                         title="Distribution of Roof Types")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The pie chart shows the proportion of different roof types in the dataset.
        This information can be useful for urban planning and development strategies.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("Feature Extraction")

        # Simulated feature extraction
        st.subheader("Simulated Feature Extraction")
        if st.button("Extract Features from New Image"):
            new_sample = generate_data(1).iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Building Area", f"{new_sample['building_area']:.2f} sq m")
            with col2:
                st.metric("Road Length", f"{new_sample['road_length']:.2f} m")
            with col3:
                st.metric("Waterbody Area", f"{new_sample['waterbody_area']:.2f} sq m")

            # Predict roof type
            features = new_sample[['building_area', 'road_length', 'waterbody_area']].values.reshape(1, -1)
            predicted_roof_type = model.predict(features)[0]
            st.metric("Predicted Roof Type", predicted_roof_type)

            # Radar chart for feature visualization
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[new_sample['building_area']/500, new_sample['road_length']/100, new_sample['waterbody_area']/1000],
                theta=['Building Area', 'Road Length', 'Waterbody Area'],
                fill='toself'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> The radar chart visualizes the extracted features from the image.
            This representation helps in quickly assessing the characteristics of the analyzed area.
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.header("Model Performance")

        # Model accuracy
        st.subheader("Model Accuracy")
        st.metric("Roof Type Classification Accuracy", f"{model_accuracy:.2f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': ['building_area', 'road_length', 'waterbody_area'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Feature Importance for Roof Type Classification")
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The feature importance plot shows which characteristics are most crucial
        for classifying roof types. This information can guide the development of more targeted feature
        extraction strategies.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This system provides AI-powered feature extraction from drone orthophotos for the SVAMITVA Scheme.
    Use these insights to enhance land record creation and rural development planning. Regular updates with
    actual field data will improve the accuracy and reliability of the feature extraction model.
    """)

if __name__ == "__main__":
    main()