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

# Apply custom CSS for black background without sidebar
st.markdown(
    """
    <style>
    body {background-color: #000000; color: #FFFFFF;}
    .stSidebar {display: none;}
    .stTabs > div {background-color: #000000; color: #FFFFFF;}
    .stMarkdown {color: #FFFFFF;}
    .css-18e3th9 {background-color: #000000; color: #FFFFFF;}
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

# Tabs for graphs
tab1, tab2, tab3 = st.tabs(["Overview", "Task Count by User", "Filtered Data View"])

# Tab 1: Overview
with tab1:
    st.subheader("Overview - Task Dashboard")
    st.write("This dashboard provides an overview of tasks and user performance.")

# Tab 2: User Task Count Visualization
with tab2:
    st.subheader("Task Count by User")
    user_task_counts = combined_df['Full_Name'].value_counts().reset_index()
    user_task_counts.columns = ['Full_Name', 'Task Count']

    fig_bar = px.bar(
        user_task_counts,
        x='Full_Name',
        y='Task Count',
        color='Task Count',
        color_continuous_scale='Blues',
        title="Task Count by User"
    )

    fig_bar.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font_color='#FFFFFF',
        xaxis_title='Users',
        yaxis_title='Task Count',
        title_font_size=18,
        title_x=0.5
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# Tab 3: Filtered Data View
with tab3:
    st.subheader("Filtered Data Preview (First 100 Rows)")
    st.dataframe(combined_df.head(100), use_container_width=True)
