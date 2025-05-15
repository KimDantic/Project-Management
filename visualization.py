import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

st.set_page_config(page_title="Task Dashboard", layout="wide")

st.markdown("""
    <style>
        .main, .block-container {
            background-color: black;
            color: lightgreen;
        }
        .css-18e3th9 {
            background-color: black;
        }
        .css-1d391kg, .css-1v0mbdj, .css-ffhzg2, .css-1dp5vir, .stMetric {
            color: lightgreen !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data

def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    if not csv_files:
        st.warning("No CSV files found.")
        return pd.DataFrame()

    dataframes = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df['started_at'] = pd.to_datetime(combined_df['started_at'], errors='coerce')
    combined_df.dropna(subset=['started_at'], inplace=True)

    combined_df['year_month'] = combined_df['started_at'].dt.to_period('M').astype(str)
    combined_df['Hours'] = combined_df['minutes'] / 60

    return combined_df

combined_df = load_data()

st.title("📊 Task Dashboard")
st.sidebar.header("Filters")

categories = st.sidebar.multiselect("Select Categories", options=["technology", "actions", "design", "writing", "meetings", "business", "errors", "time", "miscellaneous"])
date_filter = st.sidebar.date_input("Filter by Date Range", [])

filtered_data = combined_df.copy()

if categories:
    filtered_data = filtered_data[filtered_data['Categorized'].apply(lambda x: any(cat in x for cat in categories))]

if len(date_filter) == 2:
    start_date, end_date = pd.to_datetime(date_filter)
    filtered_data = filtered_data[(filtered_data['started_at'] >= start_date) & (filtered_data['started_at'] <= end_date)]

st.subheader("Overview")
col1, col2 = st.columns(2)
col1.metric("Total Tasks", filtered_data.shape[0])
col2.metric("Total Hours", round(filtered_data["Hours"].sum(), 2))

st.subheader("Task Distribution by Category")
fig_cat = px.bar(filtered_data.explode('Categorized')['Categorized'].value_counts().reset_index(), 
                 x='index', y='Categorized', color='Categorized', title='Tasks by Category')
st.plotly_chart(fig_cat, use_container_width=True)

st.subheader("Time Trends")
hours_time = filtered_data.groupby('year_month')['Hours'].sum().reset_index()
fig_hours = px.line(hours_time, x='year_month', y='Hours', title='Total Hours Over Time', markers=True)
st.plotly_chart(fig_hours, use_container_width=True)

st.subheader("Filtered Data")
st.dataframe(filtered_data)
