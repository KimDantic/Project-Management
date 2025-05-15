import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

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

    lemmatizer = WordNetLemmatizer()
    combined_df['Lemmatized_Words'] = combined_df['task'].dropna().apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in str(x).split()]))

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

# Tabs for Visuals
tab1, tab2 = st.tabs(["📋 Data Table", "📌 Lemmatized Words Table"])

# Tab 1: Raw Data
with tab1:
    st.subheader("Complete Data Table")
    st.dataframe(df, use_container_width=True)

# Tab 2: Lemmatized Words
with tab2:
    st.subheader("Lemmatized Words in Tasks")
    lemmatized_df = df[['task', 'Lemmatized_Words']].dropna()
    st.dataframe(lemmatized_df, use_container_width=True)
