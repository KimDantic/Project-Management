import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from wordcloud import WordCloud  # Import wordcloud
import subprocess
import sys

# Function to install wordcloud (only needed if not already installed)
def install_wordcloud():
    try:
        import wordcloud
    except ImportError:
        st.warning("Installing wordcloud library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])
        st.success("wordcloud library installed successfully. Please rerun the app.")
        # Important:  You might need to rerun the app after installation.
        return False  # Indicate that the app should be rerun
    return True #Indicate that the library is present

# Check if wordcloud is installed, and install if necessary
if not install_wordcloud():
    st.stop()  # Stop execution to allow installation and rerun

# Load data
@st.cache_data
def load_data():
    # Replace with your GitHub data loading logic
    url = 'https://api.github.com/repos/romero220/projectmanagement/contents/'
    token = st.secrets['GITHUB_TOKEN']  # Access the token from Streamlit secrets
    headers = {'Authorization': f'token {token}'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        files = response.json()

        csv_files = [(file['name'], file['download_url']) for file in files if file['name'].endswith('.csv')]

        dataframes = []

        for filename, url in csv_files:
            df = pd.read_csv(url)
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from GitHub: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

# Streamlit App
st.title('Data Analysis Dashboard')
data = load_data()

if not data.empty:
    # Sidebar filter
    st.sidebar.title('Filter Data')
    if 'Category' in data.columns:
        selected_category = st.sidebar.multiselect(
            'Select Category',
            options=data['Category'].unique(),
            default=data['Category'].unique()
        )
        filtered_data = data[data['Category'].isin(selected_category)]

        # Tabs for visualization
        tab1, tab2, tab3 = st.tabs(['Data Overview', 'Category Distribution', 'Word Cloud'])

        # Tab 1: Data Overview
        with tab1:
            st.header('Data Overview')
            st.write(filtered_data)

        # Tab 2: Category Distribution
        with tab2:
            st.header('Category Distribution')
            if 'Category' in filtered_data.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=filtered_data, x='Category')
                st.pyplot(plt.gcf())
            else:
                st.warning("The 'Category' column is missing in the filtered data.")

        # Tab 3: Word Cloud
        with tab3:
            st.header('Word Cloud')
            if 'Issue' in filtered_data.columns:
                text = ' '.join(filtered_data['Issue'].dropna().astype(str))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())
            else:
                st.warning("The 'Issue' column is missing in the filtered data.")
    else:
        st.warning("The 'Category' column is missing in the loaded data.")
else:
    st.info("No data loaded. Please ensure your GitHub repository and token are correctly configured.")
