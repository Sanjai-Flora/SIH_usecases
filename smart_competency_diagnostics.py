import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Smart Competency Diagnostic", layout="wide")

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

# Simulated data
@st.cache_data
def load_data():
    jobs = pd.DataFrame({
        'title': ['Software Developer', 'Data Analyst', 'Project Manager', 'UX Designer', 'Marketing Specialist'],
        'skills': [
            'Python, Java, SQL, Git',
            'Python, SQL, Excel, Statistics',
            'Project Management, Agile, Communication, Leadership',
            'UI/UX Design, Figma, Adobe XD, User Research',
            'Digital Marketing, SEO, Social Media, Content Creation'
        ],
        'description': [
            'Develop and maintain software applications',
            'Analyze data and create reports',
            'Lead projects and manage teams',
            'Design user interfaces and improve user experience',
            'Plan and execute marketing campaigns'
        ]
    })
    return jobs

jobs = load_data()

# Competency questions
questions = [
    "Rate your proficiency in Python programming (1-5)",
    "Rate your proficiency in SQL (1-5)",
    "Rate your proficiency in Project Management (1-5)",
    "Rate your proficiency in UI/UX Design (1-5)",
    "Rate your proficiency in Digital Marketing (1-5)"
]

# Function to calculate similarity
def calculate_similarity(candidate_skills, job_skills):
    vectorizer = CountVectorizer().fit_transform([candidate_skills, job_skills])
    return cosine_similarity(vectorizer)[0][1]

# Streamlit app
st.title("🧠 Smart Competency Diagnostic and Job Recommender")

# Tabs
tab0, tab1, tab2, tab3 = st.tabs(["📋 Problem Statement", "📊 Competency Diagnostic", "💼 Job Recommendations", "📈 Skill Gap Analysis"])

with tab0:
    st.header("Problem Statement")
    
    st.markdown("""
    <div class="problem-statement">
        <h3>Problem Statement ID: 1628</h3>
        <h3>Title: Smart Competency Diagnostic and Candidate Profile Score Calculator</h3>
        
        <h4>Key Requirements:</h4>
        <ul>
            <li>AI-Powered Job/Training Recommendation System</li>
            <li>Skill Gap Analysis and Recommendations</li>
            <li>Adaptive Learning Pathways</li>
            <li>Real-Time Job Market Insights</li>
            <li>Skills Verification and Certification</li>
            <li>Resume Wizard</li>
            <li>Community and Peer Support</li>
        </ul>
        
        <h4>Expected Outcome:</h4>
        <p>The system will facilitate a personalized job matching process and offer targeted skill development recommendations, helping job seekers become more competitive in the job market.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="problem-statement">
        <h3>Organization: Government of Punjab</h3>
        <h3>Department: Punjab Skill Development Mission (PSDM)</h3>
        <h3>Category: Software</h3>
        <h3>Theme: Smart Education</h3>
    </div>
    """, unsafe_allow_html=True)

with tab1:
    st.header("Competency Diagnostic")
    
    st.write("Please rate your proficiency in the following skills:")
    
    user_scores = {}
    for question in questions:
        user_scores[question] = st.slider(question, 1, 5, 3)
    
    if st.button("Calculate Competency Score"):
        total_score = sum(user_scores.values())
        max_score = len(questions) * 5
        competency_percentage = (total_score / max_score) * 100
        
        st.subheader("Your Competency Score")
        st.metric("Overall Competency", f"{competency_percentage:.1f}%")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = competency_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Competency Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig)
        
        # Radar chart for individual skills
        categories = [q.split("in ")[1].split(" (")[0] for q in questions]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[user_scores[q] for q in questions],
            theta=categories,
            fill='toself',
            name='Your Skills'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            showlegend=False
        )
        st.subheader("Your Skill Profile")
        st.plotly_chart(fig_radar)

with tab2:
    st.header("Job Recommendations")
    
    if 'user_scores' in locals():
        user_skills = ", ".join([f"{q.split('in ')[1].split(' (')[0]}" for q in questions if user_scores[q] >= 3])
        
        similarities = jobs['skills'].apply(lambda x: calculate_similarity(user_skills, x))
        recommendations = jobs.loc[similarities.nlargest(3).index]
        
        st.subheader("Top Job Recommendations")
        for _, job in recommendations.iterrows():
            with st.expander(job['title']):
                st.write(f"**Description:** {job['description']}")
                st.write(f"**Required Skills:** {job['skills']}")
                match_percentage = calculate_similarity(user_skills, job['skills']) * 100
                st.metric("Match Percentage", f"{match_percentage:.1f}%")
    else:
        st.write("Please complete the Competency Diagnostic to get job recommendations.")

with tab3:
    st.header("Skill Gap Analysis")
    
    if 'user_scores' in locals() and 'recommendations' in locals():
        st.subheader("Skill Gap Analysis for Top Recommendation")
        top_job = recommendations.iloc[0]
        required_skills = set(top_job['skills'].split(', '))
        user_skills = set([q.split("in ")[1].split(" (")[0] for q in questions if user_scores[q] >= 3])
        
        missing_skills = required_skills - user_skills
        
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=list(required_skills),
            y=[1 if skill in user_skills else 0 for skill in required_skills],
            name='Your Skills'
        ))
        fig_gap.add_trace(go.Bar(
            x=list(required_skills),
            y=[1 for _ in required_skills],
            name='Required Skills'
        ))
        fig_gap.update_layout(
            title=f"Skill Gap Analysis for {top_job['title']}",
            xaxis_title="Skills",
            yaxis_title="Proficiency",
            barmode='overlay'
        )
        st.plotly_chart(fig_gap)
        
        if missing_skills:
            st.subheader("Recommended Training")
            for skill in missing_skills:
                st.write(f"- Online course in {skill}")
        else:
            st.write("Great job! You have all the required skills for this position.")
    else:
        st.write("Please complete the Competency Diagnostic to see the Skill Gap Analysis.")

# Add a footer
st.markdown("---")
st.markdown("Developed for the Government of Punjab | Punjab Skill Development Mission (PSDM)")