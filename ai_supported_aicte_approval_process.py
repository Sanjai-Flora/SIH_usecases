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
st.set_page_config(page_title="AI-Supported AICTE Approval Process Portal", layout="wide")

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

    application_status = ['Submitted', 'Document Verification', 'Under Evaluation', 'Approved', 'Rejected']
    institution_types = ['Engineering', 'Management', 'Pharmacy', 'Architecture', 'Applied Arts and Crafts']
    evaluator_expertise = ['Academic', 'Industry', 'Research', 'Administration']

    df = pd.DataFrame({
        'application_id': range(1, n_samples + 1),
        'submission_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
        'institution_type': np.random.choice(institution_types, n_samples),
        'status': np.random.choice(application_status, n_samples),
        'processing_time': np.random.randint(1, 100, n_samples),
        'document_compliance_score': np.random.uniform(0.5, 1.0, n_samples),
        'evaluator_expertise': np.random.choice(evaluator_expertise, n_samples),
        'infrastructure_score': np.random.uniform(0.6, 1.0, n_samples)
    })

    return df

# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('aicte_approval_data.csv')
        df['submission_date'] = pd.to_datetime(df['submission_date'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('aicte_approval_data.csv', index=False)
    return df

# Train classification model
@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df[['institution_type', 'evaluator_expertise']], drop_first=True)
    X['document_compliance_score'] = df['document_compliance_score']
    X['infrastructure_score'] = df['infrastructure_score']
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

# Main application
def main():
    st.title("AI-Supported AICTE Approval Process Portal")

    # Load data and train model
    df = load_data()
    model, model_accuracy = train_model(df)

    # Sidebar for date range selection
    st.sidebar.title("Settings")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['submission_date'].min().date(), df['submission_date'].max().date()),
        min_value=df['submission_date'].min().date(),
        max_value=df['submission_date'].max().date()
    )

    # Filter data based on date range
    mask = (df['submission_date'].dt.date >= date_range[0]) & (df['submission_date'].dt.date <= date_range[1])
    filtered_df = df.loc[mask]

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Application Analysis", "AI-Powered Features", "Performance Metrics"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("AI-supported AICTE Approval Process Portal")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1729<br>
        <strong>Organization:</strong> AICTE<br>
        <strong>Department:</strong> All India Council for Technical Education (Regulation Bureau)<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Smart Automation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the need for an AI-supported portal to modernize and streamline the AICTE approval process. The system aims to:

        1. Automate and optimize the approval workflow.
        2. Enhance document verification using AI technologies.
        3. Improve transparency and communication between institutions and evaluators.
        4. Efficiently allocate resources and balance workload.
        5. Strengthen security and compliance measures.
        6. Implement digital dimension tracking of infrastructure.

        By leveraging this AI-supported system, AICTE can significantly improve the efficiency, accuracy, and transparency of the approval process for technical education institutions across India.
        """)

    with tab2:
        st.header("Application Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Applications", len(filtered_df))
        with col2:
            st.metric("Average Processing Time", f"{filtered_df['processing_time'].mean():.2f} days")
        with col3:
            st.metric("Approval Rate", f"{(filtered_df['status'] == 'Approved').mean():.2%}")

        # Application status distribution
        st.subheader("Application Status Distribution")
        fig_status = px.pie(filtered_df, names='status', title="Distribution of Application Statuses")
        st.plotly_chart(fig_status, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The pie chart shows the distribution of application statuses.
        This information helps in understanding the current state of applications and identifying bottlenecks.
        </div>
        """, unsafe_allow_html=True)

        # Processing time by institution type
        st.subheader("Processing Time by Institution Type")
        fig_processing = px.box(filtered_df, x='institution_type', y='processing_time',
                                title="Processing Time Distribution by Institution Type")
        st.plotly_chart(fig_processing, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The box plot displays processing time distributions across different institution types.
        This can help in identifying which types of institutions may require more attention or resources.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("AI-Powered Features")

        # Simulated AI document verification
        st.subheader("AI-Powered Document Verification")
        if st.button("Simulate Document Verification"):
            compliance_score = np.random.uniform(0.5, 1.0)
            st.metric("Document Compliance Score", f"{compliance_score:.2f}")

            if compliance_score >= 0.9:
                st.success("Documents are fully compliant with AICTE standards.")
            elif compliance_score >= 0.7:
                st.warning("Minor issues detected. Please review and resubmit the highlighted documents.")
            else:
                st.error("Significant compliance issues found. Please carefully review all documents.")

        # Simulated AI infrastructure verification
        st.subheader("AI-Powered Infrastructure Verification")
        if st.button("Simulate Infrastructure Verification"):
            infrastructure_score = np.random.uniform(0.6, 1.0)
            st.metric("Infrastructure Score", f"{infrastructure_score:.2f}")

            if infrastructure_score >= 0.9:
                st.success("Infrastructure meets or exceeds AICTE standards.")
            elif infrastructure_score >= 0.75:
                st.warning("Minor improvements needed in some areas of infrastructure.")
            else:
                st.error("Significant infrastructure improvements required to meet AICTE standards.")

        st.markdown("""
        <div class="insight-box">
        <strong>AI Features:</strong><br>
        - Automated document verification and compliance checking<br>
        - AI-powered infrastructure assessment using computer vision<br>
        - Intelligent evaluator matching based on expertise and workload<br>
        - Predictive analytics for application outcomes and processing times
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("Performance Metrics")

        # Model accuracy
        st.subheader("AI Model Accuracy")
        st.metric("Application Status Prediction Accuracy", f"{model_accuracy:.2f}")

        # Processing time trend
        st.subheader("Processing Time Trend")
        timeline_df = filtered_df.groupby('submission_date')['processing_time'].mean().reset_index()
        fig_timeline = px.line(timeline_df, x='submission_date', y='processing_time',
                               title="Average Processing Time Trend")
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The line chart shows the trend in average processing times.
        This can help in identifying improvements in efficiency over time or periods that may require additional resources.
        </div>
        """, unsafe_allow_html=True)

        # Evaluator workload distribution
        st.subheader("Evaluator Workload Distribution")
        evaluator_workload = filtered_df['evaluator_expertise'].value_counts()
        fig_workload = px.bar(evaluator_workload, x=evaluator_workload.index, y=evaluator_workload.values,
                              title="Application Distribution by Evaluator Expertise")
        st.plotly_chart(fig_workload, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The bar chart shows the distribution of applications across different evaluator expertise areas.
        This information can be used to ensure balanced workload distribution and identify any skill gaps.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This AI-supported system enhances the AICTE approval process by automating workflows,
    improving document and infrastructure verification, and providing data-driven insights.
    Regular updates with real application data will further improve the accuracy and effectiveness of the system.
    """)

if __name__ == "__main__":
    main()