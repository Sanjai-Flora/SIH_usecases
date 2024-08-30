import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Student Dropout Reduction System", layout="wide")

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
def generate_student_data(n_students=1000):
    np.random.seed(42)
    data = {
        'student_id': range(1, n_students + 1),
        'attendance': np.random.uniform(60, 100, n_students),
        'grades': np.random.uniform(40, 100, n_students),
        'behavior_score': np.random.uniform(50, 100, n_students),
        'socioeconomic_score': np.random.uniform(20, 100, n_students),
        'parental_engagement': np.random.uniform(0, 10, n_students),
        'dropout_risk': np.random.uniform(0, 1, n_students)
    }
    return pd.DataFrame(data)

# Load or generate data
@st.cache_resource
def load_data():
    return generate_student_data()

# AI-Driven Early Warning System
def predict_dropout_risk(student_data):
    # This is a simplified model. In a real application, you would use a more sophisticated ML model.
    risk = (100 - student_data['attendance']) * 0.3 + \
           (100 - student_data['grades']) * 0.3 + \
           (100 - student_data['behavior_score']) * 0.2 + \
           (100 - student_data['socioeconomic_score']) * 0.1 + \
           (10 - student_data['parental_engagement']) * 10 * 0.1
    return risk / 100

# Main application
def main():
    st.title("Student Dropout Reduction System")

    # Load data
    df = load_data()

    # Sidebar for settings
    st.sidebar.title("Settings")
    risk_threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.7)

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Problem Statement", "Early Warning System", "Community Learning Hub", "Financial Support", "Parental Engagement"])

    with tab1:
        st.header("Problem Statement")
        st.subheader("Implement Software Solutions to Reduce Student Dropout Rates at Various Educational Stages")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1661<br>
        <strong>Organization:</strong> Ministry of Education<br>
        <strong>Department:</strong> Samagra Shiksha Abhiyan, Department of School Education & Literacy (DoSEL)<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Smart Education
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application addresses the critical need for reducing student dropout rates in India. The system aims to:

        1. Implement AI-driven early warning systems to identify at-risk students.
        2. Provide community learning hub platforms for additional educational support.
        3. Manage financial support systems to reduce economic barriers.
        4. Engage parents in their child's education through dedicated portals.
        5. Offer flexible schooling options and personalized support for at-risk students.

        By leveraging these software solutions, we aim to significantly reduce dropout rates and align with the National Education Policy (NEP) 2020's objectives for universal access to quality education.
        """)

    with tab2:
        st.header("AI-Driven Early Warning System")

        # Calculate dropout risk
        df['dropout_risk'] = predict_dropout_risk(df)

        # Display high-risk students
        high_risk_students = df[df['dropout_risk'] > risk_threshold].sort_values('dropout_risk', ascending=False)
        st.subheader("High-Risk Students")
        st.dataframe(high_risk_students)

        # Visualize risk distribution
        st.subheader("Dropout Risk Distribution")
        fig = px.histogram(df, x='dropout_risk', nbins=30, title="Distribution of Dropout Risk")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The AI-driven early warning system identifies students at high risk of dropping out.
        Educators should focus on students above the risk threshold for immediate intervention.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("Community Learning Hub Platform")

        # Simulate community learning hub data
        hub_data = pd.DataFrame({
            'Hub Name': ['Hub A', 'Hub B', 'Hub C', 'Hub D'],
            'Active Students': np.random.randint(50, 200, 4),
            'Online Classes': np.random.randint(10, 50, 4),
            'Tutoring Sessions': np.random.randint(20, 100, 4),
            'Resource Downloads': np.random.randint(100, 1000, 4)
        })

        st.subheader("Community Learning Hub Statistics")
        st.dataframe(hub_data)

        # Visualize hub activity
        fig = px.bar(hub_data, x='Hub Name', y=['Active Students', 'Online Classes', 'Tutoring Sessions'],
                     title="Community Learning Hub Activity")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Community learning hubs provide crucial support to students in underserved areas.
        The data shows active engagement across various hubs, indicating their effectiveness in supplementing formal education.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("Financial Support Management System")

        # Simulate financial support data
        financial_data = pd.DataFrame({
            'Support Type': ['Scholarships', 'Grants', 'Loans', 'Stipends'],
            'Applications': np.random.randint(100, 1000, 4),
            'Approved': np.random.randint(50, 500, 4),
            'Total Amount (₹)': np.random.randint(100000, 1000000, 4)
        })

        st.subheader("Financial Support Statistics")
        st.dataframe(financial_data)

        # Visualize financial support distribution
        fig = px.pie(financial_data, values='Total Amount (₹)', names='Support Type',
                     title="Distribution of Financial Support")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The financial support system is crucial in reducing economic barriers to education.
        The pie chart shows the distribution of different types of financial aid, helping identify areas that may need more funding or promotion.
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.header("Parental Engagement Portal")

        # Simulate parental engagement data
        engagement_data = pd.DataFrame({
            'Engagement Type': ['Parent-Teacher Meetings', 'Progress Report Views', 'Resource Downloads', 'Event Participation'],
            'Engagement Count': np.random.randint(100, 1000, 4)
        })

        st.subheader("Parental Engagement Statistics")
        st.dataframe(engagement_data)

        # Visualize parental engagement
        fig = px.bar(engagement_data, x='Engagement Type', y='Engagement Count',
                     title="Parental Engagement Activities")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> Parental engagement is a key factor in student success. The data shows various ways parents are
        involved in their child's education. Encouraging more participation in areas with lower engagement could help reduce dropout rates.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This Student Dropout Reduction System demonstrates various software solutions to address dropout rates.
    In a real-world application, it would integrate with actual school databases, learning management systems, and 
    financial aid platforms for more accurate and comprehensive analysis and support.
    """)

if __name__ == "__main__":
    main()