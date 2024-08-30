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
st.set_page_config(page_title="AI-Powered Rail Madad Complaint Management", layout="wide")

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

    complaint_categories = ['Cleanliness', 'Damage', 'Staff Behavior', 'Food Quality', 'Delay']
    urgency_levels = ['Low', 'Medium', 'High']
    departments = ['Housekeeping', 'Maintenance', 'Customer Service', 'Catering', 'Operations']
    resolution_status = ['Pending', 'In Progress', 'Resolved']

    df = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(n_samples)],
        'category': np.random.choice(complaint_categories, n_samples),
        'urgency': np.random.choice(urgency_levels, n_samples),
        'department': np.random.choice(departments, n_samples),
        'resolution_time': np.random.randint(1, 72, n_samples),
        'status': np.random.choice(resolution_status, n_samples),
        'sentiment_score': np.random.uniform(-1, 1, n_samples)
    })

    return df

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('rail_madad_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('rail_madad_data.csv', index=False)
    return df

# Train classification model
@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df[['category', 'urgency', 'department']], drop_first=True)
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

# Main application
def main():
    st.title("AI-Powered Rail Madad Complaint Management")

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
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Complaint Analysis", "AI-Powered Features", "Performance Metrics"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("Enhancing Rail Madad with AI-powered Complaint Management")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1711<br>
        <strong>Organization:</strong> Ministry of Railway<br>
        <strong>Department:</strong> Ministry of Railway<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Smart Automation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the need for an AI-powered system to enhance the Rail Madad complaint management process. The system aims to:

        1. Automate categorization and prioritization of complaints.
        2. Enhance data extraction from images, videos, and audio.
        3. Implement automated response and intelligent routing.
        4. Enable predictive maintenance through issue prediction.
        5. Provide continuous improvement through feedback analysis.
        6. Offer AI-assisted training and resource allocation.

        By leveraging this system, the Ministry of Railway can significantly improve the efficiency and effectiveness of the Rail Madad platform.
        """)

    with tab2:
        st.header("Complaint Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Complaints", len(filtered_df))
        with col2:
            st.metric("Average Resolution Time", f"{filtered_df['resolution_time'].mean():.2f} hours")
        with col3:
            st.metric("Resolved Complaints", len(filtered_df[filtered_df['status'] == 'Resolved']))

        # Complaint category distribution
        st.subheader("Complaint Category Distribution")
        fig_category = px.pie(filtered_df, names='category', title="Distribution of Complaint Categories")
        st.plotly_chart(fig_category, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The pie chart shows the distribution of complaint categories.
        This information can help in allocating resources and identifying areas that need improvement.
        </div>
        """, unsafe_allow_html=True)

        # Complaint urgency over time
        st.subheader("Complaint Urgency Over Time")
        urgency_df = filtered_df.groupby(['timestamp', 'urgency']).size().unstack(fill_value=0).reset_index()
        fig_urgency = px.area(urgency_df, x='timestamp', y=['Low', 'Medium', 'High'],
                              title="Complaint Urgency Trends")
        st.plotly_chart(fig_urgency, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The area chart displays urgency trends over time.
        This can help in identifying peak periods and allocating resources accordingly.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("AI-Powered Features")

        # Simulated AI categorization
        st.subheader("AI-Powered Complaint Categorization")
        if st.button("Simulate New Complaint"):
            new_complaint = generate_data(1).iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Category", new_complaint['category'])
            with col2:
                st.metric("Assigned Urgency", new_complaint['urgency'])
            with col3:
                st.metric("Routed to Department", new_complaint['department'])

            # Sentiment analysis
            sentiment = "Positive" if new_complaint['sentiment_score'] > 0 else "Negative"
            st.metric("Sentiment Analysis", sentiment)

            st.markdown("""
            <div class="insight-box">
            <strong>AI Features:</strong><br>
            - Automated categorization based on complaint content<br>
            - Urgency detection for prioritization<br>
            - Smart routing to appropriate department<br>
            - Sentiment analysis for feedback processing
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.header("Performance Metrics")

        # Model accuracy
        st.subheader("AI Model Accuracy")
        st.metric("Complaint Status Prediction Accuracy", f"{model_accuracy:.2f}")

        # Resolution time distribution
        st.subheader("Resolution Time Distribution")
        fig_resolution = px.histogram(filtered_df, x='resolution_time', nbins=20,
                                      title="Distribution of Complaint Resolution Times")
        st.plotly_chart(fig_resolution, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The histogram shows the distribution of resolution times.
        This can help in setting realistic SLAs and identifying areas for process improvement.
        </div>
        """, unsafe_allow_html=True)

        # Department performance
        st.subheader("Department Performance")
        dept_performance = filtered_df.groupby('department')['resolution_time'].mean().sort_values(ascending=True)
        fig_dept = px.bar(dept_performance, x=dept_performance.index, y='resolution_time',
                          title="Average Resolution Time by Department")
        st.plotly_chart(fig_dept, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The bar chart compares the average resolution time across departments.
        This can help in identifying high-performing departments and areas needing improvement.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This AI-powered system enhances the Rail Madad complaint management process.
    It provides automated categorization, prioritization, and routing of complaints,
    along with advanced analytics for continuous improvement. Regular updates with
    real complaint data will further improve the accuracy and effectiveness of the system.
    """)

if __name__ == "__main__":
    main()