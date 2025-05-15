import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Project Management Dashboard", layout="wide")

# White theme styling
st.markdown("""
    <style>
        .main, .block-container {
            background-color: #ffffff;
            color: black;
        }
        .css-18e3th9 {
            background-color: #ffffff;
        }
        .css-1d391kg, .css-1v0mbdj, .css-ffhzg2, .css-1dp5vir, .stMetric {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Project Management Dashboard")

# Load data
@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    if not csv_files:
        st.warning("No CSV files found in the repository.")
        return pd.DataFrame()

    combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    combined_df["Hours"] = combined_df["minutes"] / 60
    combined_df["started_at"] = pd.to_datetime(combined_df["started_at"], errors="coerce")
    return combined_df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")
search_term = st.sidebar.text_input("Search Tasks")
user_filter = st.sidebar.selectbox("Select User", options=["All"] + list(df["user_first_name"].unique()))

if user_filter != "All":
    df = df[df["user_first_name"] == user_filter]

if search_term:
    df = df[df["task"].str.contains(search_term, case=False, na=False)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tasks", df.shape[0])
col2.metric("Total Hours", round(df["Hours"].sum(), 2))
col3.metric("Unique Users", df["user_first_name"].nunique())
col4.metric("Average Hours/Task", round(df["Hours"].mean(), 2))

# Tabs for Visuals
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "🧑‍💼 User Insights", "📊 Analytics", "📌 Task Completion Patterns"])

# Tab 1: Raw Data
with tab1:
    st.subheader("Complete Data Table")
    st.dataframe(df, use_container_width=True)

# Tab 2: User Analysis
with tab2:
    st.subheader("🔄 User Task Distribution")
    user_task_counts = df["user_first_name"].value_counts().reset_index()
    user_task_counts.columns = ["User", "Task Count"]
    bar_fig = px.bar(user_task_counts, x="User", y="Task Count", title="Task Distribution by User", color="Task Count", color_continuous_scale='Viridis')
    st.plotly_chart(bar_fig, use_container_width=True)

# Tab 3: Analytics
with tab3:
    st.subheader("📈 Hours Over Time")
    time_df = df.groupby([df["started_at"].dt.date])['Hours'].sum().reset_index()
    line_fig = px.line(time_df, x="started_at", y="Hours", title="Total Hours Over Time", markers=True, color_discrete_sequence=['#FF5733'])
    st.plotly_chart(line_fig, use_container_width=True)

    st.subheader("📌 Task Distribution by Category")
    if "task" in df.columns:
        task_counts = df["task"].value_counts().reset_index()
        task_counts.columns = ["Task", "Count"]
        pie_fig = px.pie(task_counts, names="Task", values="Count", title="Task Distribution", hole=0.3)
        st.plotly_chart(pie_fig, use_container_width=True)

# Tab 4: Task Completion Patterns
with tab4:
    st.subheader("📊 Task Completion Trends")
    df["Weekday"] = df["started_at"].dt.day_name()
    weekday_counts = df["Weekday"].value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    bar_fig = px.bar(weekday_counts, x=weekday_counts.index, y=weekday_counts.values, title="Tasks Completed by Day of the Week", color=weekday_counts.values, color_continuous_scale='Viridis')
    st.plotly_chart(bar_fig, use_container_width=True)

    hour_df = df["started_at"].dt.hour.value_counts().sort_index()
    hour_fig = px.bar(hour_df, x=hour_df.index, y=hour_df.values, title="Tasks Completed by Hour of the Day", color=hour_df.values, color_continuous_scale='Viridis')
    st.plotly_chart(hour_fig, use_container_width=True)
