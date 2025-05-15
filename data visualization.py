import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Task Dashboard", layout="wide")

# Red theme styling
st.markdown("""
    <style>
        .main, .block-container {
            background-color: red;
            color: white;
        }
        .css-18e3th9 {
            background-color: red;
        }
        .css-1d391kg, .css-1v0mbdj, .css-ffhzg2, .css-1dp5vir, .stMetric {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Task Dashboard (Red Theme)")

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

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Tasks", df.shape[0])
col2.metric("Total Hours", round(df["Hours"].sum(), 2))
col3.metric("Unique Users", df["user_first_name"].nunique())

# Tabs for Visuals
tab1, tab2, tab3 = st.tabs(["📋 Data Table", "🧑‍💼 User Insights", "📅 Time Insights"])

# Tab 1: Raw Data
with tab1:
    st.subheader("Complete Data Table")
    st.dataframe(df, use_container_width=True)

# Tab 2: User Analysis
with tab2:
    st.subheader("🔄 Donut Chart: Tasks per User")
    user_task_counts = df["user_first_name"].value_counts().reset_index()
    user_task_counts.columns = ["User", "Task Count"]
    
    donut_fig = px.pie(user_task_counts, values="Task Count", names="User", hole=0.4, 
                       title="Task Distribution by User", color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(donut_fig, use_container_width=True)

    st.subheader("📊 Task Duration Histogram")
    hist_fig = px.histogram(df, x="Hours", nbins=20, title="Distribution of Task Durations (Hours)",
                            color_discrete_sequence=['red'])
    st.plotly_chart(hist_fig, use_container_width=True)

# Tab 3: Time Analysis
with tab3:
    st.subheader("📈 Cumulative Hours Over Time (By User)")
    if "user_first_name" in df.columns:
        df_grouped = df.groupby([df["started_at"].dt.date, "user_first_name"])["Hours"].sum().reset_index()
        df_grouped["started_at"] = pd.to_datetime(df_grouped["started_at"])
        area_fig = px.area(df_grouped, x="started_at", y="Hours", color="user_first_name", 
                           title="Hours Worked Over Time by User", 
                           color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(area_fig, use_container_width=True)
    else:
        st.warning("User data not available for time-based breakdown.")
