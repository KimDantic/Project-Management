import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Setup NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# GitHub repo details
repo_owner = "romero220"
repo_name = "projectmanagement"
branch = "main"
token = os.getenv("GITHUB_TOKEN")  # Ensure your token is stored in an environment variable

api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
headers = {"Authorization": f"token {token}"}

response = requests.get(api_url, headers=headers)
response.raise_for_status()
files = response.json()

# Filter CSV files
csv_files = [(file['name'], file['download_url']) for file in files if file['name'].endswith('.csv')]

if not csv_files:
    print("No CSV files found in the repository.")
else:
    print("CSV files found:")
    dataframes = []

    for filename, url in csv_files:
        print(f"Reading file: {filename}")
        numeric_id = filename.split('-')[2]
        df = pd.read_csv(url)
        df['FileID'] = numeric_id
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Ensure columns exist
    required_columns = ['Issue', 'Category']
    if all(col in combined_df.columns for col in required_columns):
        data = combined_df[required_columns].copy()
        data.columns = ['text', 'label']
        print("\nData loaded successfully.")
    else:
        print(f"Missing columns: {required_columns}")
