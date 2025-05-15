import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from wordcloud import WordCloud  # Import wordcloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from io import StringIO

# Function to install wordcloud (only needed if not already installed)
def install_wordcloud():
    try:
        import wordcloud
    except ImportError:
        import subprocess
        import sys
        st.warning("Installing wordcloud library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])
        st.success("wordcloud library installed successfully. Please rerun the app.")
        return False
    return True

# Check if wordcloud is installed, and install if necessary
if not install_wordcloud():
    st.stop()

# Setup NLTK (moved outside load_data to avoid caching issues)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    else:
        return ""  # Return empty string for non-string input


# Load data
@st.cache_data
def load_data():
    # GitHub repo details
    repo_owner = "romero220"
    repo_name = "projectmanagement"
    branch = "main"
    token = st.secrets.get("GITHUB_TOKEN")  # Use st.secrets.get()
    if not token:
        st.error("GITHUB_TOKEN is not set in Streamlit secrets.  Please add it!")
        return pd.DataFrame()

    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
    headers = {"Authorization": f"token {token}"}

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        files = response.json()

        # Filter CSV files
        csv_files = [(file['name'], file['download_url']) for file in files if file['name'].endswith('.csv')]

        if not csv_files:
            st.error("No CSV files found in the repository.")
            return pd.DataFrame()
        else:
            dataframes = []
            for filename, url in csv_files:
                try:
                    # Use a more robust way to read CSV from a URL
                    response = requests.get(url)
                    response.raise_for_status()  # Check for download errors
                    csv_content = response.text
                    df = pd.read_csv(StringIO(csv_content))

                    numeric_id = filename.split('-')[2]
                    df['FileID'] = numeric_id
                    dataframes.append(df)
                except Exception as e:
                    st.error(f"Error reading CSV file {filename}: {e}")
                    return pd.DataFrame()

            combined_df = pd.concat(dataframes, ignore_index=True)

            # Ensure columns exist
            required_columns = ['Issue', 'Category']
            if all(col in combined_df.columns for col in required_columns):
                data = combined_df[required_columns].copy()
                data.columns = ['text', 'label']
                data['text'] = data['text'].apply(preprocess_text) # Preprocess
                return data
            else:
                st.error(f"Missing columns: {required_columns}")
                return pd.DataFrame()  # Return empty DataFrame

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from GitHub: {e}")
        return pd.DataFrame()

# Streamlit App
st.title('Data Analysis Dashboard')
data = load_data()

if not data.empty:
    # Sidebar filter
    st.sidebar.title('Filter Data')
    if 'label' in data.columns:
        selected_category = st.sidebar.multiselect(
            'Select Category',
            options=data['label'].unique(),
            default=data['label'].unique()
        )
        filtered_data = data[data['label'].isin(selected_category)]

        # Tabs for visualization
        tab1, tab2, tab3 = st.tabs(['Data Overview', 'Category Distribution', 'Word Cloud'])

        # Tab 1: Data Overview
        with tab1:
            st.header('Data Overview')
            st.write(filtered_data)

        # Tab 2: Category Distribution
        with tab2:
            st.header('Category Distribution')
            if 'label' in filtered_data.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=filtered_data, x='label')
                st.pyplot(plt.gcf())
            else:
                st.warning("The 'Category' column is missing in the filtered data.")

        # Tab 3: Word Cloud
        with tab3:
            st.header('Word Cloud')
            if 'text' in filtered_data.columns:
                text = ' '.join(filtered_data['text'].dropna().astype(str))
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
