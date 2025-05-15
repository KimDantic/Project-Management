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
import plotly.express as px

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# Streamlit page configuration
st.set_page_config(page_title="Task Dashboard", layout="wide")

# Apply custom CSS for black background
st.markdown(
    """
    <style>
    body {background-color: #000000; color: #FFFFFF;}
    .stSidebar {background-color: #000000;}
    .stTabs > div {background-color: #000000;}
    .stMarkdown {color: #FFFFFF;}
    .css-18e3th9 {background-color: #000000;}
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the repository.")
        return pd.DataFrame()

    dataframes = []
    for filename in csv_files:
        df = pd.read_csv(filename)
        numeric_id = filename.split('-')[2] if '-' in filename else 'Unknown'
        df['ProjectID'] = numeric_id
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + "-" + combined_df['id'].astype(str)
    combined_df['Full_Name'] = combined_df['user_first_name'].astype(str) + " " + combined_df['user_last_name'].astype(str)

    return combined_df

# Load the data
combined_df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Select Categories", options=combined_df['Categorized'].explode().unique())

# Tabs for graphs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Hours", "Entries", "4", "5", "6", "7"])

# Tab 1: Overview
with tab1:
    st.subheader("Preview of Filtered Data (First 100 Rows)")
    st.dataframe(combined_df.head(100), use_container_width=True)
