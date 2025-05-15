import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Task Dashboard", layout="wide")

# Grey theme styling
st.markdown("""
    <style>
        .main, .block-container {
            background-color: #2f2f2f;
            color: white;
        }
        .css-18e3th9 {
            background-color: #2f2f2f;
        }
        .css-1d391kg, .css-1v0mbdj, .css-ffhzg2, .css-1dp5vir, .stMetric {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Task Dashboard (Grey Theme)")

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
col1, col2, col3 = st.columns(3)
col1.metric("Total Tasks", df.shape[0])
col2.metric("Total Hours", round(df["Hours"].sum(), 2))
col3.metric("Unique Users", df["user_first_name"].nunique())

# Tabs for Visuals with new design
tab1, tab2, tab3 = st.tabs(["📋 Data Table", "🧑‍💼 User Insights", "📊 Analytics"])

# Tab 1: Raw Data
with tab1:
    st.subheader("Complete Data Table")
    st.dataframe(df, use_container_width=True)

# Tab 2: User Analysis
with tab2:
    st.subheader("🔄 User Task Distribution")
    user_task_counts = df["user_first_name"].value_counts().reset_index()
    user_task_counts.columns = ["User", "Task Count"]
    bar_fig = px.bar(user_task_counts, x="User", y="Task Count", title="Task Distribution by User", color="Task Count", color_continuous_scale='Blues')
    st.plotly_chart(bar_fig, use_container_width=True)

# Tab 3: Analytics
with tab3:
    st.subheader("📈 Hours Over Time")
    if "user_first_name" in df.columns:
        time_df = df.groupby([df["started_at"].dt.date])["Hours"].sum().reset_index()
        line_fig = px.line(time_df, x="started_at", y="Hours", title="Total Hours Over Time", markers=True, color_discrete_sequence=['#00BFFF'])
        st.plotly_chart(line_fig, use_container_width=True)
    else:
        st.warning("No data available for time-based breakdown.")
