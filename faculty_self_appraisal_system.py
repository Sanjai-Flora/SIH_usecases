import streamlit as st
import plotly.express as px
import pandas as pd
import random
from datetime import datetime, timedelta

# Mock data generation
def generate_mock_data(n=50):
    names = ["Dr. " + name for name in ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]]
    departments = ["Computer Science", "Physics", "Mathematics", "Chemistry", "Biology", "Engineering", "Psychology", "Economics", "History", "Literature"]
    
    data = []
    for _ in range(n):
        name = random.choice(names)
        dept = random.choice(departments)
        publications = random.randint(0, 20)
        seminars = random.randint(0, 10)
        projects = random.randint(0, 5)
        lectures = random.randint(20, 100)
        emp_code = f"EMP{random.randint(1000, 9999)}"
        submission_date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        data.append({
            "Name": name,
            "Department": dept,
            "Publications": publications,
            "Seminars": seminars,
            "Projects": projects,
            "Lectures": lectures,
            "Employee Code": emp_code,
            "Submission Date": submission_date
        })
    
    return pd.DataFrame(data)

# Main application
def main():
    st.set_page_config(page_title="Faculty Self-Appraisal System", layout="wide")
    st.title("Faculty Self-Appraisal System")

    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Faculty Dashboard", "Admin Panel"])

    if page == "Faculty Dashboard":
        faculty_dashboard()
    else:
        admin_panel()

# Faculty Dashboard
def faculty_dashboard():
    st.header("Faculty Dashboard")

    # Personal Information Form
    st.subheader("Personal Information")
    name = st.text_input("Full Name")
    emp_code = st.text_input("Employee Code")
    department = st.selectbox("Department", ["Computer Science", "Physics", "Mathematics", "Chemistry", "Biology", "Engineering", "Psychology", "Economics", "History", "Literature"])

    # Academic Activities Form
    st.subheader("Academic Activities")
    publications = st.number_input("Number of Publications", min_value=0)
    seminars = st.number_input("Number of Seminars Attended", min_value=0)
    projects = st.number_input("Number of Ongoing Projects", min_value=0)
    lectures = st.number_input("Number of Lectures Delivered", min_value=0)

    if st.button("Submit Self-Appraisal"):
        st.success("Self-Appraisal submitted successfully!")

# Admin Panel
def admin_panel():
    st.header("Admin Panel")

    # Generate mock data
    df = generate_mock_data()

    # Sorting options
    sort_by = st.selectbox("Sort by", ["Name", "Employee Code", "Submission Date"])
    df_sorted = df.sort_values(by=sort_by)

    # Display sorted data
    st.dataframe(df_sorted)

    # Visualizations
    st.subheader("Faculty Performance Overview")

    # Publications by Department
    fig_pub = px.bar(df.groupby("Department")["Publications"].mean().reset_index(), 
                     x="Department", y="Publications", title="Average Publications by Department")
    st.plotly_chart(fig_pub)

    # Seminars vs Projects Scatter Plot
    fig_scatter = px.scatter(df, x="Seminars", y="Projects", color="Department", 
                             title="Seminars vs Projects by Department")
    st.plotly_chart(fig_scatter)

    # Lectures Distribution
    fig_hist = px.histogram(df, x="Lectures", title="Distribution of Lectures Delivered")
    st.plotly_chart(fig_hist)

    # Download option (placeholder)
    if st.button("Download Report (PDF)"):
        st.info("PDF download functionality would be implemented here.")

if __name__ == "__main__":
    main()