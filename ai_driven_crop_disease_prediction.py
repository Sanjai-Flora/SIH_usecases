import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# Set page config
st.set_page_config(page_title="AI-Driven Crop Disease Prediction and Management System", layout="wide")

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
        color: #2e7d32;
        font-family: 'Arial', sans-serif;
    }
    .stAlert {
        background-color: #e8f5e9;
        border: 1px solid #2e7d32;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
    }
    .stSelectbox {
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data
@st.cache_data
def generate_data(n_samples=1000):
    np.random.seed(42)
    crops = ['Wheat', 'Rice', 'Maize', 'Potato', 'Tomato']
    diseases = ['Healthy', 'Rust', 'Blight', 'Leaf Spot', 'Powdery Mildew']
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        disease = np.random.choice(diseases)
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 90)
        rainfall = np.random.uniform(0, 50)
        soil_moisture = np.random.uniform(20, 80)
        
        data.append({
            'crop': crop,
            'disease': disease,
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'soil_moisture': soil_moisture,
            'date': datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
    
    return pd.DataFrame(data)

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('crop_disease_data.csv')
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('crop_disease_data.csv', index=False)
    return df

# Train AI model for disease prediction
@st.cache_resource
def train_model(df):
    X = df[['temperature', 'humidity', 'rainfall', 'soil_moisture']]
    y = df['disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test))

    return model, accuracy, report

# Main application
def main():
    st.title("AI-Driven Crop Disease Prediction and Management System")

    # Load data and train model
    df = load_data()
    model, model_accuracy, classification_report = train_model(df)

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
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Crop Health Analysis", "Disease Prediction", "Management Recommendations"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("AI-Driven Crop Disease Prediction and Management System")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1638<br>
        <strong>Organization:</strong> Ministry of Agriculture and Farmers Welfare<br>
        <strong>Department:</strong> University of Agricultural Sciences, Dharwad (UASD), The Indian Council of Agricultural Research<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Agriculture, FoodTech & Rural Development
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the critical need for early detection and management of crop diseases. The system aims to:

        1. Analyze crop images and environmental data to predict potential disease outbreaks.
        2. Provide farmers with actionable insights and treatment recommendations.
        3. Utilize machine learning algorithms to identify crop diseases.
        4. Suggest preventive measures and treatments based on real-time data.
        5. Offer a user-friendly interface for both mobile and web platforms.

        By leveraging this AI-driven system, farmers can mitigate risks, reduce crop losses, and improve overall agricultural productivity.
        """)

    with tab2:
        st.header("Crop Health Analysis")

        # Overall disease prevalence
        st.subheader("Disease Prevalence")
        disease_counts = filtered_df['disease'].value_counts()
        fig_disease = px.pie(disease_counts, values=disease_counts.values, names=disease_counts.index,
                             title="Overall Disease Prevalence")
        st.plotly_chart(fig_disease, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This chart shows the distribution of various crop diseases in the selected time range.
        It helps identify the most common diseases affecting crops.
        </div>
        """, unsafe_allow_html=True)

        # Disease prevalence by crop
        st.subheader("Disease Prevalence by Crop")
        crop_disease = filtered_df.groupby(['crop', 'disease']).size().unstack(fill_value=0)
        fig_crop_disease = px.bar(crop_disease, x=crop_disease.index, y=crop_disease.columns,
                                  title="Disease Prevalence by Crop", barmode='stack')
        st.plotly_chart(fig_crop_disease, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This stacked bar chart illustrates how different diseases affect various crops.
        It can help in identifying which crops are more susceptible to specific diseases.
        </div>
        """, unsafe_allow_html=True)

        # Environmental factors correlation
        st.subheader("Environmental Factors Correlation")
        corr_matrix = filtered_df[['temperature', 'humidity', 'rainfall', 'soil_moisture']].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                             title="Correlation between Environmental Factors")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This heatmap shows the correlation between different environmental factors.
        Understanding these relationships can help in predicting disease-favorable conditions.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("Disease Prediction")

        # Model performance
        st.subheader("Model Performance")
        st.metric("Model Accuracy", f"{model_accuracy:.2f}")

        st.text("Classification Report:")
        st.text(classification_report)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': ['temperature', 'humidity', 'rainfall', 'soil_moisture'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Feature Importance for Disease Prediction")
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The feature importance chart shows which environmental factors have the most significant
        impact on disease prediction. This information can guide farmers in prioritizing which conditions to monitor closely.
        </div>
        """, unsafe_allow_html=True)

        # Disease prediction tool
        st.subheader("Disease Prediction Tool")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature (°C)", 15.0, 35.0, 25.0)
            humidity = st.slider("Humidity (%)", 30.0, 90.0, 60.0)
        with col2:
            rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 25.0)
            soil_moisture = st.slider("Soil Moisture (%)", 20.0, 80.0, 50.0)

        if st.button("Predict Disease"):
            input_data = pd.DataFrame({
                'temperature': [temperature],
                'humidity': [humidity],
                'rainfall': [rainfall],
                'soil_moisture': [soil_moisture]
            })
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data).max()
            
            st.success(f"Predicted Disease: {prediction}")
            st.info(f"Prediction Confidence: {probability:.2f}")

    with tab4:
        st.header("Management Recommendations")

        # Simulated disease management recommendations
        st.subheader("Disease Management Guide")
        selected_disease = st.selectbox("Select Disease", df['disease'].unique())

        recommendations = {
            'Healthy': "No treatment needed. Continue regular monitoring and maintenance.",
            'Rust': "Apply fungicide treatment. Improve air circulation around plants. Remove infected leaves.",
            'Blight': "Use copper-based fungicides. Ensure proper drainage. Rotate crops in the next season.",
            'Leaf Spot': "Apply appropriate fungicide. Avoid overhead watering. Remove and destroy infected plant debris.",
            'Powdery Mildew': "Apply sulfur-based fungicide. Increase plant spacing for better air circulation. Use resistant varieties in future plantings."
        }

        st.markdown(f"""
        <div class="insight-box">
        <strong>Recommendation for {selected_disease}:</strong><br>
        {recommendations.get(selected_disease, "No specific recommendation available.")}
        </div>
        """, unsafe_allow_html=True)

        # Preventive measures
        st.subheader("General Preventive Measures")
        st.markdown("""
        1. Practice crop rotation to break disease cycles.
        2. Use disease-resistant crop varieties when available.
        3. Maintain proper plant spacing for good air circulation.
        4. Regularly monitor crops for early signs of disease.
        5. Implement proper irrigation management to avoid excess moisture.
        6. Keep fields clean and free of plant debris.
        7. Use balanced fertilization to promote plant health.
        8. Consider using biological control agents as part of an integrated pest management strategy.
        """)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This AI-driven crop disease prediction and management system provides insights based on
    environmental data and machine learning models. For the most accurate results, regularly update the
    system with local crop and weather data. Always consult with agricultural experts for final decisions
    on crop management strategies.
    """)

if __name__ == "__main__":
    main()