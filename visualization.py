import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

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
        st.warning("No CSV files found in the repository.")
        return pd.DataFrame()

    dataframes = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df['year_month'] = pd.to_datetime(combined_df["started_at"], errors="coerce").dt.to_period("M")

    return combined_df

combined_df = load_data()

st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Select Categories", options=combined_df['Categorized'].explode().unique())

overview_tab, trends_tab, data_tab = st.tabs(["📊 Overview", "📈 Time Trends", "📋 Data Table"])

with overview_tab:
    st.title("📊 Dashboard Overview")
    st.metric("Total Tasks", combined_df.shape[0])

    st.subheader("🔍 Word Cloud")
    all_words = [word for sublist in combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] for word in sublist]
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Greens').generate(' '.join(all_words))
    st.image(wordcloud.to_array(), use_column_width=True)

with trends_tab:
    st.title("📈 Time-Based Analysis")

    fig = px.line(combined_df, x='year_month', y='Hours', title='Total Hours Over Time')
    st.plotly_chart(fig)

with data_tab:
    st.title("📋 Data Table")
    st.dataframe(combined_df, use_container_width=True)
