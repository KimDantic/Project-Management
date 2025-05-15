import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import nltk
from nltk.stem import WordNetLemmatizer

# Streamlit page configuration
st.set_page_config(page_title="Unique Task Dashboard", layout="wide")

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# Load main data
@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

    if not csv_files:
        st.error("No CSV files found in the directory.")
        return pd.DataFrame()

    dataframes = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df['Date'] = pd.to_datetime(combined_df['started_at'], errors='coerce')

    # Text processing
    lemmatizer = WordNetLemmatizer()
    combined_df['task_processed'] = combined_df['task'].str.lower().str.replace('[^a-z ]', '', regex=True)
    combined_df['task_lemmatized'] = combined_df['task_processed'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    return combined_df

# Load data
data = load_data()

# Sidebar filters
st.sidebar.header("Filters")
project_ids = st.sidebar.multiselect("Select Project ID", options=data['ProjectID'].unique())

# Filter data
filtered_data = data.copy()
if project_ids:
    filtered_data = filtered_data[filtered_data['ProjectID'].isin(project_ids)]

# Visualization
st.title("Task Dashboard")
st.subheader("Task Distribution by Category")
category_counts = filtered_data['task_lemmatized'].str.split().explode().value_counts().nlargest(10)
fig, ax = plt.subplots()
sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax, palette="viridis")
ax.set_title("Top 10 Task Keywords")
st.pyplot(fig)

st.subheader("Task Trend Over Time")
date_trend = filtered_data.groupby(filtered_data['Date'].dt.to_period('M')).size()
st.line_chart(date_trend)

# Download filtered data
csv_data = filtered_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download Filtered Data", data=csv_data, file_name="filtered_tasks.csv")
