import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    # Replace with your GitHub data loading logic
    url = 'https://api.github.com/repos/romero220/projectmanagement/contents/'
    token = st.secrets['GITHUB_TOKEN']
    headers = {'Authorization': f'token {token}'}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    files = response.json()

    csv_files = [(file['name'], file['download_url']) for file in files if file['name'].endswith('.csv')]

    dataframes = []

    for filename, url in csv_files:
        df = pd.read_csv(url)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

# Streamlit App
st.title('Data Analysis Dashboard')
data = load_data()

# Sidebar filter
st.sidebar.title('Filter Data')
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
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_data, x='Category')
    st.pyplot(plt.gcf())

# Tab 3: Word Cloud
with tab3:
    st.header('Word Cloud')
    from wordcloud import WordCloud
    text = ' '.join(filtered_data['Issue'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())
