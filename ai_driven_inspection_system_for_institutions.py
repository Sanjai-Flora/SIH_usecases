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
st.set_page_config(page_title="AI-Driven Inspection System for Institutions", layout="wide")

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

    inspection_status = ['Completed', 'In Progress', 'Scheduled']
    institution_types = ['Engineering', 'Management', 'Pharmacy', 'Architecture', 'Applied Arts and Crafts']
    compliance_levels = ['Fully Compliant', 'Partially Compliant', 'Non-Compliant']

    df = pd.DataFrame({
        'institution_id': range(1, n_samples + 1),
        'inspection_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
        'institution_type': np.random.choice(institution_types, n_samples),
        'status': np.random.choice(inspection_status, n_samples),
        'infrastructure_score': np.random.uniform(0.5, 1.0, n_samples),
        'faculty_score': np.random.uniform(0.6, 1.0, n_samples),
        'student_performance_score': np.random.uniform(0.4, 1.0, n_samples),
        'compliance_level': np.random.choice(compliance_levels, n_samples),
        'overall_score': np.random.uniform(0.5, 1.0, n_samples)
    })

    return df

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('institution_inspection_data.csv')
        df['inspection_date'] = pd.to_datetime(df['inspection_date'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('institution_inspection_data.csv', index=False)
    return df

# Train classification model
@st.cache_resource
def train_model(df):
    X = df[['infrastructure_score', 'faculty_score', 'student_performance_score', 'overall_score']]
    y = df['compliance_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

# Main application
def main():
    st.title("AI-Driven Inspection System for Institutions")

    # Load data and train model
    df = load_data()
    model, model_accuracy = train_model(df)

    # Sidebar for date range selection
    st.sidebar.title("Settings")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['inspection_date'].min().date(), df['inspection_date'].max().date()),
        min_value=df['inspection_date'].min().date(),
        max_value=df['inspection_date'].max().date()
    )

    # Filter data based on date range
    mask = (df['inspection_date'].dt.date >= date_range[0]) & (df['inspection_date'].dt.date <= date_range[1])
    filtered_df = df.loc[mask]

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Inspection Analysis", "AI-Powered Features", "Performance Metrics"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("AI-driven Inspection of Institutions")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1730<br>
        <strong>Organization:</strong> AICTE<br>
        <strong>Department:</strong> All India Council for Technical Education (Regulation Bureau)<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Smart Automation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the need for an AI-driven Inspection System for Institutions. The system aims to:

        1. Automate facility inspections using image recognition.
        2. Analyze documents and reports using natural language processing.
        3. Collect and analyze real-time data from various sources.
        4. Identify trends and potential issues using machine learning algorithms.
        5. Generate comprehensive reports with actionable insights.

        By leveraging this AI-driven system, AICTE can significantly improve the efficiency, consistency, and effectiveness of institutional inspections across India.
        """)

    with tab2:
        st.header("Inspection Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Inspections", len(filtered_df))
        with col2:
            st.metric("Average Overall Score", f"{filtered_df['overall_score'].mean():.2f}")
        with col3:
            st.metric("Fully Compliant Institutions", f"{(filtered_df['compliance_level'] == 'Fully Compliant').mean():.2%}")

        # Compliance level distribution
        st.subheader("Compliance Level Distribution")
        fig_compliance = px.pie(filtered_df, names='compliance_level', title="Distribution of Compliance Levels")
        st.plotly_chart(fig_compliance, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The pie chart shows the distribution of compliance levels among inspected institutions.
        This information helps in understanding the overall compliance landscape and identifying areas for improvement.
        </div>
        """, unsafe_allow_html=True)

        # Scores by institution type
        st.subheader("Inspection Scores by Institution Type")
        fig_scores = px.box(filtered_df, x='institution_type', y=['infrastructure_score', 'faculty_score', 'student_performance_score'],
                            title="Score Distribution by Institution Type")
        st.plotly_chart(fig_scores, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The box plot displays score distributions across different institution types.
        This can help in identifying which types of institutions excel in certain areas or may require more support.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("AI-Powered Features")

        # Simulated AI facility inspection
        st.subheader("AI-Powered Facility Inspection")
        if st.button("Simulate Facility Inspection"):
            infrastructure_score = np.random.uniform(0.5, 1.0)
            st.metric("Infrastructure Score", f"{infrastructure_score:.2f}")

            if infrastructure_score >= 0.9:
                st.success("Facilities meet or exceed standards. No major issues detected.")
            elif infrastructure_score >= 0.7:
                st.warning("Minor improvements needed in some areas of infrastructure.")
            else:
                st.error("Significant infrastructure improvements required to meet standards.")

        # Simulated AI document analysis
        st.subheader("AI-Powered Document Analysis")
        if st.button("Simulate Document Analysis"):
            document_score = np.random.uniform(0.6, 1.0)
            st.metric("Document Compliance Score", f"{document_score:.2f}")

            if document_score >= 0.9:
                st.success("Documents are fully compliant with regulations.")
            elif document_score >= 0.75:
                st.warning("Minor issues detected in documentation. Please review and update.")
            else:
                st.error("Significant compliance issues found in documentation. Immediate attention required.")

        st.markdown("""
        <div class="insight-box">
        <strong>AI Features:</strong><br>
        - Automated facility inspection using image recognition<br>
        - Document analysis using natural language processing<br>
        - Real-time data collection and analysis<br>
        - Pattern recognition for identifying trends and potential issues<br>
        - Automated report generation with actionable insights
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("Performance Metrics")

        # Model accuracy
        st.subheader("AI Model Accuracy")
        st.metric("Compliance Level Prediction Accuracy", f"{model_accuracy:.2f}")

        # Inspection trends
        st.subheader("Inspection Score Trends")
        timeline_df = filtered_df.groupby('inspection_date')[['infrastructure_score', 'faculty_score', 'student_performance_score']].mean().reset_index()
        fig_timeline = px.line(timeline_df, x='inspection_date', y=['infrastructure_score', 'faculty_score', 'student_performance_score'],
                               title="Average Inspection Score Trends")
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The line chart shows trends in average inspection scores over time.
        This can help in identifying improvements or declines in different areas and guide policy decisions.
        </div>
        """, unsafe_allow_html=True)

        # Compliance level by institution type
        st.subheader("Compliance Level by Institution Type")
        compliance_by_type = filtered_df.groupby('institution_type')['compliance_level'].value_counts(normalize=True).unstack()
        fig_compliance_type = px.bar(compliance_by_type, x=compliance_by_type.index, y=compliance_by_type.columns,
                                     title="Compliance Level Distribution by Institution Type")
        st.plotly_chart(fig_compliance_type, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The stacked bar chart shows the distribution of compliance levels across different institution types.
        This information can be used to tailor inspection and support strategies for different types of institutions.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This AI-driven system enhances the institutional inspection process by automating facility assessments,
    improving document analysis, and providing data-driven insights. Regular updates with real inspection data will
    further improve the accuracy and effectiveness of the system.
    """)

if __name__ == "__main__":
    main()