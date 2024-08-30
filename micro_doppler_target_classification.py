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
st.set_page_config(page_title="Micro-Doppler Target Classification System", layout="wide")

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
    frequency = np.random.uniform(1, 10, n_samples)
    amplitude = np.random.uniform(0, 1, n_samples)
    duration = np.random.uniform(0.1, 2, n_samples)
    
    # Generate labels (0 for bird, 1 for drone)
    labels = np.random.choice([0, 1], n_samples)
    
    # Adjust features based on labels
    frequency += labels * np.random.uniform(0, 5, n_samples)  # Drones tend to have higher frequency
    amplitude += labels * np.random.uniform(0, 0.5, n_samples)  # Drones tend to have higher amplitude
    duration -= labels * np.random.uniform(0, 0.5, n_samples)  # Birds tend to have longer duration
    
    # Create DataFrame
    df = pd.DataFrame({
        'frequency': frequency,
        'amplitude': amplitude,
        'duration': duration,
        'label': labels
    })
    
    df['class'] = df['label'].map({0: 'Bird', 1: 'Drone'})
    df['timestamp'] = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]
    
    return df

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('micro_doppler_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('micro_doppler_data.csv', index=False)
    return df

# Train classification model
@st.cache_resource
def train_model(df):
    X = df[['frequency', 'amplitude', 'duration']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

# Main application
def main():
    st.title("Micro-Doppler Target Classification System")

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
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Data Analysis", "Classification", "Real-time Detection"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("Micro-Doppler based Target Classification")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1606<br>
        <strong>Organization:</strong> Bharat Electronics Limited (BEL)<br>
        <strong>Department:</strong> Bharat Electronics Limited (BEL)<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Robotics and Drones
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the critical need for efficient technology to detect and classify drones in airspace,
        distinguishing them from birds using micro-Doppler signatures captured from radar sensors. The system aims to:

        1. Analyze micro-Doppler signatures of flying objects.
        2. Classify targets as either drones or birds based on their micro-Doppler characteristics.
        3. Provide real-time detection and classification capabilities.
        4. Visualize and interpret micro-Doppler data for better understanding.
        5. Support security and surveillance operations by enhancing situational awareness.

        By leveraging this system, Bharat Electronics Limited can contribute to improved airspace monitoring,
        drone detection, and overall security in various applications.
        """)

    with tab2:
        st.header("Data Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Observations", len(filtered_df))
        with col2:
            st.metric("Drones Detected", len(filtered_df[filtered_df['class'] == 'Drone']))
        with col3:
            st.metric("Birds Detected", len(filtered_df[filtered_df['class'] == 'Bird']))

        # Micro-Doppler signature visualization
        st.subheader("Micro-Doppler Signature Visualization")
        selected_class = st.selectbox("Select Target Class", ['All', 'Drone', 'Bird'])
        
        if selected_class != 'All':
            plot_df = filtered_df[filtered_df['class'] == selected_class]
        else:
            plot_df = filtered_df

        fig = px.scatter_3d(plot_df, x='frequency', y='amplitude', z='duration', color='class',
                            title="3D Micro-Doppler Signature Plot")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The 3D plot shows the distribution of micro-Doppler signatures for drones and birds.
        Drones tend to have higher frequency and amplitude, while birds often have longer duration signatures.
        </div>
        """, unsafe_allow_html=True)

        # Time series of detections
        st.subheader("Detection Timeline")
        timeline_df = filtered_df.groupby(['timestamp', 'class']).size().unstack(fill_value=0).reset_index()
        fig_timeline = px.line(timeline_df, x='timestamp', y=['Drone', 'Bird'],
                               title="Target Detections Over Time")
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The timeline shows the frequency of drone and bird detections over time.
        This can help identify patterns or unusual activities in the monitored airspace.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("Classification")

        # Model performance
        st.subheader("Model Performance")
        st.metric("Model Accuracy", f"{model_accuracy:.2f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': ['frequency', 'amplitude', 'duration'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Feature Importance for Target Classification")
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The feature importance plot shows which micro-Doppler characteristics
        are most crucial for distinguishing between drones and birds. This information can guide the
        development of more targeted detection strategies.
        </div>
        """, unsafe_allow_html=True)

        # Confusion Matrix
        y_true = df['label']
        y_pred = model.predict(df[['frequency', 'amplitude', 'duration']])
        cm = confusion_matrix(y_true, y_pred)
        
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Bird', 'Drone'], y=['Bird', 'Drone'],
                           title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The confusion matrix provides a detailed view of the model's performance,
        showing how well it distinguishes between drones and birds. This helps in understanding any
        misclassifications and potential areas for improvement.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("Real-time Detection Simulation")

        # Simulated real-time detection
        st.subheader("Simulated Target Detection")
        if st.button("Detect New Target"):
            new_target = generate_data(1).iloc[0]
            
            features = new_target[['frequency', 'amplitude', 'duration']].values.reshape(1, -1)
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Class", "Drone" if prediction == 1 else "Bird")
            with col2:
                st.metric("Confidence", f"{max(prediction_proba):.2f}")

            # Radar chart for feature visualization
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[new_target['frequency'], new_target['amplitude'], new_target['duration']],
                theta=['Frequency', 'Amplitude', 'Duration'],
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
            <strong>Insight:</strong> The radar chart visualizes the micro-Doppler features of the detected target.
            This representation helps in quickly assessing the characteristics that led to the classification decision.
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This system provides real-time classification of targets as drones or birds based on
    micro-Doppler signatures. Use these insights to enhance airspace monitoring and security operations.
    Regular updates with actual field data will improve the accuracy and reliability of the classification model.
    """)

if __name__ == "__main__":
    main()