import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
from collections import defaultdict

# Set page config
st.set_page_config(page_title="Sansthaein Aur Samvidhan: Learn the Constitution", layout="wide")

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

# Simulated database of constitutional articles
@st.cache_data
def load_constitution_data():
    data = {
        "Legislature": {
            "Article 79": "Parliament consists of the President and two Houses - Lok Sabha and Rajya Sabha.",
            "Article 80": "Composition of the Rajya Sabha.",
            "Article 81": "Composition of the Lok Sabha.",
            # Add more articles...
        },
        "Executive": {
            "Article 52": "There shall be a President of India.",
            "Article 53": "Executive power of the Union vested in the President.",
            "Article 74": "Council of Ministers to aid and advise President.",
            # Add more articles...
        },
        "Judiciary": {
            "Article 124": "Establishment and constitution of Supreme Court.",
            "Article 214": "High Courts for States.",
            "Article 233": "Appointment of district judges.",
            # Add more articles...
        }
    }
    return data

# Game: Spin the Wheel
def spin_the_wheel_game(constitution_data):
    st.subheader("Spin the Wheel: Constitution Edition")
    
    organs = list(constitution_data.keys())
    selected_organ = st.selectbox("Select an organ of the Constitution", organs)
    
    if st.button("Spin the Wheel"):
        article = random.choice(list(constitution_data[selected_organ].keys()))
        st.success(f"The wheel landed on: {article}")
        st.info(f"Simplified explanation: {constitution_data[selected_organ][article]}")

# Game: Constitution Cards
def constitution_cards_game(constitution_data):
    st.subheader("Constitution Cards")
    
    all_articles = [(organ, article, explanation) 
                    for organ, articles in constitution_data.items() 
                    for article, explanation in articles.items()]
    
    if st.button("Draw a Card"):
        organ, article, explanation = random.choice(all_articles)
        st.success(f"You drew: {article} ({organ})")
        st.info(f"Simplified explanation: {explanation}")

# Quiz game
def quiz_game(constitution_data):
    st.subheader("Constitution Quiz")
    
    all_questions = []
    for organ, articles in constitution_data.items():
        for article, explanation in articles.items():
            all_questions.append({
                "question": f"Which article of the Constitution deals with: {explanation}",
                "correct_answer": article,
                "organ": organ
            })
    
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'current_question' not in st.session_state:
        st.session_state.current_question = random.choice(all_questions)
    
    st.write(st.session_state.current_question["question"])
    user_answer = st.text_input("Enter the article number:")
    
    if st.button("Submit Answer"):
        if user_answer.lower() == st.session_state.current_question["correct_answer"].lower():
            st.success("Correct!")
            st.session_state.quiz_score += 1
        else:
            st.error(f"Incorrect. The correct answer is {st.session_state.current_question['correct_answer']}")
        st.session_state.current_question = random.choice(all_questions)
    
    st.write(f"Current Score: {st.session_state.quiz_score}")

# Main application
def main():
    st.title("Sansthaein Aur Samvidhan: Learn the Constitution")

    # Load constitution data
    constitution_data = load_constitution_data()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Home", "Spin the Wheel", "Constitution Cards", "Quiz", "Learning Analytics"])

    if app_mode == "Home":
        st.header("Welcome to Sansthaein Aur Samvidhan")
        st.subheader("Let's Learn the Constitution in a Simpler Manner")

        st.markdown("""
        <div class="insight-box">
        <strong>Problem Statement ID:</strong> 1699<br>
        <strong>Organization:</strong> Ministry of Law & Justice<br>
        <strong>Department:</strong> Department of Justice<br>
        <strong>Category:</strong> Software<br>
        <strong>Theme:</strong> Miscellaneous
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This application is designed to spread constitutional literacy among citizens by simplifying
        the language of the Constitution of India, focusing on the three organs: Legislature, Executive, and Judiciary.
        
        Features:
        1. Simplified explanations of constitutional articles
        2. Multiple game formats for engaging learning
        3. Quiz to test your knowledge
        4. Learning analytics to track progress
        
        Explore the different modes in the sidebar to start your constitutional journey!
        """)

    elif app_mode == "Spin the Wheel":
        spin_the_wheel_game(constitution_data)

    elif app_mode == "Constitution Cards":
        constitution_cards_game(constitution_data)

    elif app_mode == "Quiz":
        quiz_game(constitution_data)

    elif app_mode == "Learning Analytics":
        st.header("Learning Analytics")

        # Simulated user progress data
        progress_data = pd.DataFrame({
            'Organ': ['Legislature', 'Executive', 'Judiciary'] * 10,
            'Articles_Learned': np.random.randint(1, 20, 30),
            'Quiz_Score': np.random.randint(0, 100, 30),
            'Time_Spent_Minutes': np.random.randint(5, 60, 30)
        })

        # Overall progress
        st.subheader("Overall Learning Progress")
        fig = px.bar(progress_data.groupby('Organ').sum().reset_index(), 
                     x='Organ', y='Articles_Learned', 
                     title="Articles Learned by Constitutional Organ")
        st.plotly_chart(fig)

        # Quiz performance
        st.subheader("Quiz Performance")
        fig = px.box(progress_data, x='Organ', y='Quiz_Score', 
                     title="Quiz Scores Distribution by Constitutional Organ")
        st.plotly_chart(fig)

        # Time spent learning
        st.subheader("Time Spent Learning")
        fig = px.pie(progress_data, values='Time_Spent_Minutes', names='Organ', 
                     title="Distribution of Time Spent Learning Each Organ")
        st.plotly_chart(fig)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The learning analytics provide valuable insights into user engagement and performance.
        This data can be used to identify areas where users might need additional support or where the content could be
        improved to enhance the learning experience.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This application is a prototype demonstrating the concept of gamified constitutional learning.
    In a production environment, it would include a more comprehensive database of constitutional articles,
    additional game formats, and integration with user accounts for personalized learning experiences.
    """)

if __name__ == "__main__":
    main()