import streamlit as st
import pandas as pd
import seaborn as sns
import os
import warnings
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
import plotly.express as px
from wordcloud import WordCloud
from github import Github
import datetime
import matplotlib.pyplot as plt


# Token and GitHub details
token = "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"
repo_owner = "romero220"
repo_name = "projectmanagement"
branch = "main"

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# Streamlit config
st.set_page_config(page_title="Task Dashboard", layout="wide")
# --- Enhanced Dashboard Title with Subtitle and Styling ---
st.markdown("""
    <div style="display: flex; align-items: center; gap: 18px;">
        <img src="https://img.icons8.com/color/96/000000/task.png" width="60" style="margin-bottom: 0;">
        <div>
            <h1 style="margin-bottom: 0; color: #fc6c64;">🗂️ Project Management Dashboard</h1>
            <p style="margin-top: 0; font-size: 1.2rem; color: #555;">
                Visualize, analyze, and manage your team's project tasks with interactive insights.
            </p>
        </div>
    </div>
    <hr style="border: 1px solid #fc6c64; margin-top: 0.5rem; margin-bottom: 1.5rem;">
""", unsafe_allow_html=True)


# Color palette - changed to tab10 for better category distinction
# We'll dynamically set number of colors needed based on unique users
def get_color_palette(num_colors):
    return sns.color_palette("tab10", n_colors=num_colors).as_hex()

@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    if not csv_files:
        print("No CSV files found.")
        return pd.DataFrame()

    dataframes = []
    for filename in csv_files:
        if os.path.getsize(filename) == 0:
            continue  # Skip empty files
        try:
            df = pd.read_csv(filename)
            numeric_id = filename.split('-')[2] if '-' in filename else 'Unknown'
            df['ProjectID'] = numeric_id
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not dataframes:
        return pd.DataFrame()

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + "-" + combined_df['id'].astype(str)
    combined_df['Full_Name'] = combined_df['user_first_name'].astype(str) + " " + combined_df['user_last_name'].astype(str)
    combined_df['Hours'] = combined_df['minutes'] / 60

    combined_df['task_wo_punct'] = combined_df['task'].apply(lambda x: ''.join([char for char in str(x) if char not in string.punctuation]))
    combined_df['task_wo_punct_split'] = combined_df['task_wo_punct'].apply(lambda x: re.split(r'\W+', str(x).lower()))
    stopword = nltk.corpus.stopwords.words('english')
    combined_df['task_wo_punct_split_wo_stopwords'] = combined_df['task_wo_punct_split'].apply(lambda x: [word for word in x if word not in stopword])
    lemmatizer = WordNetLemmatizer()
    combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] = combined_df['task_wo_punct_split_wo_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return combined_df

# Load data
combined_df = load_data()

# Compute keyword_counts after loading data so it's available globally
if not combined_df.empty:
    keyword_counts = pd.Series(
        {kw: combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(lambda x: kw in x).sum()
         for kw in set([item for sublist in combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] for item in sublist])}
    )
    keyword_counts = keyword_counts.sort_values(ascending=False).reset_index()
    keyword_counts.columns = ['keyword', 'count']
    # Drop rows where 'keyword' contains only numbers
    keyword_counts = keyword_counts[~keyword_counts['keyword'].str.fullmatch(r'\d+')]
    keyword_counts['keyword+count'] = keyword_counts['keyword'] + " (" + keyword_counts['count'].astype(str) + ")"
else:
    keyword_counts = pd.DataFrame(columns=['keyword', 'count', 'keyword+count'])

# Show logo at the top of the sidebar
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image("POPIN.png")

st.sidebar.markdown(
    f"""
    <div style="text-align: center; padding-bottom: 10px;">
        <img src="data:image/png;base64,{img_base64}" width="240">
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar filters
st.sidebar.markdown("## 🛠️ Controls & Data Filters")
st.sidebar.header("Filters")

# --- Dynamic Sidebar Filters ---

# Helper to get filtered options based on current selections
def get_dynamic_options(df):
    project_ids = df['ProjectID'].dropna().unique()
    full_names = df['Full_Name'].dropna().unique()
    # Recompute keyword counts for the filtered data
    if not df.empty and 'task_wo_punct_split_wo_stopwords_lemmatized' in df.columns:
        keyword_counts = pd.Series(
            {kw: df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(lambda x: kw in x).sum()
             for kw in set([item for sublist in df['task_wo_punct_split_wo_stopwords_lemmatized'] for item in sublist])}
        )
        keyword_counts = keyword_counts.sort_values(ascending=False).reset_index()
        keyword_counts.columns = ['keyword', 'count']
        keyword_counts = keyword_counts[~keyword_counts['keyword'].str.fullmatch(r'\d+')]
        keyword_counts['keyword+count'] = keyword_counts['keyword'] + " (" + keyword_counts['count'].astype(str) + ")"
        keyword_options = keyword_counts['keyword+count'].tolist()
        keyword_lookup = {f"{row['keyword']} ({row['count']})": row['keyword'] for _, row in keyword_counts.iterrows()}
    else:
        keyword_options = []
        keyword_lookup = {}
    # Add "All Users" option at the top
    full_name_options = ["All Users"] + list(full_names)
    return project_ids, full_name_options, keyword_options, keyword_lookup

# Initial options (unfiltered)
project_ids, full_name_options, keyword_options, keyword_lookup = get_dynamic_options(combined_df)

# Sidebar filter widgets (step by step, updating options dynamically)
# Add an info icon with tooltip beside the Project ID selector
col_proj, col_info = st.sidebar.columns([8, 1])
ProjectID = col_proj.multiselect("Select Project ID", options=project_ids)
# Use a gray circle with a white "i" for the info icon
col_info.markdown(
    '''
    <span title="Filter the data by Project ID. You can select multiple projects. \nThe Project ID was extracted from the filename.">
        <svg width="18" height="18" style="vertical-align:middle;">
            <circle cx="9" cy="9" r="8" fill="#b0b0b0"/>
            <text x="9" y="13" text-anchor="middle" font-size="12" fill="white" font-family="Arial" font-weight="bold">i</text>
        </svg>
    </span>
    ''',
    unsafe_allow_html=True
)
# Filter data by ProjectID for next filter options
filtered_df = combined_df[combined_df['ProjectID'].isin(ProjectID)] if ProjectID else combined_df


# Update options after ProjectID filter
project_ids, full_name_options, keyword_options, keyword_lookup = get_dynamic_options(filtered_df)
col_fullname, col_fullname_info = st.sidebar.columns([8, 1])
full_name_filter = col_fullname.multiselect("Filter by Full Name", options=full_name_options, default=["All Users"])
col_fullname_info.markdown(
    '''
    <span title="Filter the data by user full name. Select 'All Users' to aggregate all users as one.">
        <svg width="18" height="18" style="vertical-align:middle;">
            <circle cx="9" cy="9" r="8" fill="#b0b0b0"/>
            <text x="9" y="13" text-anchor="middle" font-size="12" fill="white" font-family="Arial" font-weight="bold">i</text>
        </svg>
    </span>
    ''',
    unsafe_allow_html=True
)

# Filter data by Full_Name for next filter options
if full_name_filter and full_name_filter != ["All Users"]:
    filtered_df = filtered_df[filtered_df['Full_Name'].isin(full_name_filter)]

# Update options after Full_Name filter
project_ids, full_name_options, keyword_options, keyword_lookup = get_dynamic_options(filtered_df)
col_keyword, col_keyword_info = st.sidebar.columns([8, 1])
keyword_finder = col_keyword.multiselect("Select a Keyword", options=keyword_options)
col_keyword_info.markdown(
    '''
    <span title="Filter tasks by keyword (lemmatized, stopwords removed). Select one or more keywords to filter tasks containing them.">
        <svg width="18" height="18" style="vertical-align:middle;">
            <circle cx="9" cy="9" r="8" fill="#b0b0b0"/>
            <text x="9" y="13" text-anchor="middle" font-size="12" fill="white" font-family="Arial" font-weight="bold">i</text>
        </svg>
    </span>
    ''',
    unsafe_allow_html=True
)

# Filter data by selected keywords for next filter options
if keyword_finder:
    selected_keywords = [keyword_lookup[k] for k in keyword_finder]
    filtered_df = filtered_df[
        filtered_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(
            lambda x: any(word in x for word in selected_keywords)
        )
    ]

# Date and search filters (do not affect options, just values)
col_date, col_date_info = st.sidebar.columns([8, 1])
date_filter = col_date.date_input("Filter by Date", [])
col_date_info.markdown(
    '''
    <span title="Filter tasks by start date. Select a single date or a date range.">
        <svg width="18" height="18" style="vertical-align:middle;">
            <circle cx="9" cy="9" r="8" fill="#b0b0b0"/>
            <text x="9" y="13" text-anchor="middle" font-size="12" fill="white" font-family="Arial" font-weight="bold">i</text>
        </svg>
    </span>
    ''',
    unsafe_allow_html=True
)
col_search, col_search_info = st.sidebar.columns([8, 1])
search_term = col_search.text_input("Search Task", "")
col_search_info.markdown(
    '''
    <span title="Search for tasks containing a specific keyword or phrase. This filter matches any part of the task description.">
        <svg width="18" height="18" style="vertical-align:middle;">
            <circle cx="9" cy="9" r="8" fill="#b0b0b0"/>
            <text x="9" y="13" text-anchor="middle" font-size="12" fill="white" font-family="Arial" font-weight="bold">i</text>
        </svg>
    </span>
    ''',
    unsafe_allow_html=True
)
col_timegroup, col_timegroup_info = st.sidebar.columns([8, 1])
time_group = col_timegroup.selectbox("Group by Time Period", options=["Yearly", "Monthly", "Weekly", "Daily"])
col_timegroup_info.markdown(
    '''
    <span title="Choose how to group time-based charts: Yearly, Monthly, Weekly, or Daily.">
        <svg width="18" height="18" style="vertical-align:middle;">
            <circle cx="9" cy="9" r="8" fill="#b0b0b0"/>
            <text x="9" y="13" text-anchor="middle" font-size="12" fill="white" font-family="Arial" font-weight="bold">i</text>
        </svg>
    </span>
    ''',
    unsafe_allow_html=True
)

# Filter data efficiently
filtered_data = combined_df

if ProjectID:
    filtered_data = filtered_data[filtered_data['ProjectID'].isin(ProjectID)]

if keyword_finder:
    selected_keywords = [keyword_lookup[k] for k in keyword_finder]
    filtered_data = filtered_data[
        filtered_data['task_wo_punct_split_wo_stopwords_lemmatized'].apply(
            lambda x: any(word in x for word in selected_keywords)
        )
    ]

if len(date_filter) == 2:
    filtered_data = filtered_data.copy()
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce").dt.tz_localize(None)
    start_date = pd.to_datetime(date_filter[0])
    end_date = pd.to_datetime(date_filter[1])
    filtered_data = filtered_data[
        (filtered_data["started_at"] >= start_date) & (filtered_data["started_at"] <= end_date)
    ]

if search_term:
    filtered_data = filtered_data[filtered_data['task'].str.contains(search_term, case=False, na=False)]

if "Full_Name" in filtered_data.columns:
    if full_name_filter == ["All Users"]:
        # Treat all users as one entity by setting a single value for 'Full_Name'
        filtered_data = filtered_data.copy()
        filtered_data['Full_Name'] = 'All Users'
    elif full_name_filter and full_name_filter != ["All Users"]:
        # Filter for selected users
        filtered_data = filtered_data[filtered_data['Full_Name'].isin(full_name_filter)]
    else:
        # No user filter applied: treat all users as separate entities (default behavior)
        pass
else:
    st.warning("'Full_Name' column not found in the data. User filtering skipped.")

filtered_data = filtered_data.reset_index(drop=True)

# Download filtered data button
csv_data = filtered_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(label="📥 Download Filtered CSV", data=csv_data, file_name="filtered_data.csv", mime="text/csv")

# File upload and GitHub push

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
expected_columns = [
    "id", "task", "started_at", "minutes", "description", "created_at", "updated_at", "user_id",
    "user_first_name", "user_last_name", "user_biography", "user_feedbacks_count",
    "user_feedbacks_average", "user_achievements_count", "user_locale", "user_created_at", "user_updated_at"
]

def push_file_to_github(file_name, file_content, repo_owner, repo_name, branch, token):
    g = Github(token)
    repo = g.get_user(repo_owner).get_repo(repo_name)
    try:
        # Check if file exists
        contents = repo.get_contents(file_name, ref=branch)
        repo.update_file(
            contents.path,
            f"Update {file_name} via dashboard at {datetime.datetime.now()}",
            file_content,
            contents.sha,
            branch=branch
        )
    except Exception:
        # If file does not exist, create it
        repo.create_file(
            file_name,
            f"Add {file_name} via dashboard at {datetime.datetime.now()}",
            file_content,
            branch=branch
        )

if uploaded_file is not None:
    file_name = uploaded_file.name
    try:
        uploaded_df = pd.read_csv(uploaded_file, nrows=10)
        uploaded_cols = list(uploaded_df.columns)
        if uploaded_cols != expected_columns:
            st.sidebar.error(
                "Uploaded file schema does not match the expected columns.\n\n"
                f"Expected columns:\n{expected_columns}\n\n"
                f"Found columns:\n{uploaded_cols}"
            )
        else:
            if st.sidebar.button(f"Upload '{file_name}' to GitHub"):
                file_content = uploaded_file.getvalue().decode("utf-8")
                try:
                    push_file_to_github(
                        file_name=file_name,
                        file_content=file_content,
                        repo_owner=repo_owner,
                        repo_name=repo_name,
                        branch=branch,
                        token=token
                    )
                    st.sidebar.success(f"File '{file_name}' uploaded to GitHub repository '{repo_owner}/{repo_name}'!")
                except Exception as e:
                    st.sidebar.error(f"GitHub upload failed: {e}")
            else:
                st.sidebar.info("Click the button above to upload to GitHub.")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

# --- File Deletion Option (GitHub) ---
with st.sidebar.expander("🗑️ Delete CSV File(s) from GitHub Repository"):
    try:
        g = Github(token)
        repo = g.get_user(repo_owner).get_repo(repo_name)
        contents = repo.get_contents("", ref=branch)
        csv_files = [content.name for content in contents if content.name.endswith('.csv')]
    except Exception as e:
        st.info(f"Could not fetch files from GitHub: {e}")
        csv_files = []

    if not csv_files:
        st.info("No CSV files found in the GitHub repository.")
    else:
        selected_file = st.selectbox("Select a file to delete", options=csv_files, key="delete_github_file")
        confirm_delete = st.checkbox(f"Confirm deletion of '{selected_file}' from GitHub")
        if st.button("Delete Selected File from GitHub", disabled=not confirm_delete):
            try:
                file_content = repo.get_contents(selected_file, ref=branch)
                repo.delete_file(
                    file_content.path,
                    f"Delete {selected_file} via dashboard at {datetime.datetime.now()}",
                    file_content.sha,
                    branch=branch
                )
                st.success(f"File '{selected_file}' deleted from GitHub repository.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error deleting file from GitHub: {e}")


# Tabs
# --- Improved Tabs with Emojis and Custom Styling ---

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 Data Overview", 
    "📊 User Summary", 
    "📈 Main Visualizations", 
    "🏆 Top & Bottom Users", 
    "☁️ Word Cloud", 
    "👤 User Drilldown", 
    "💡 Extra Insights"
])

# Optional: Add a little CSS for tab header spacing and font
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        padding: 0.5rem 1.2rem;
        margin-right: 0;  /* No horizontal gap between tabs */
        border-radius: 8px 8px 0 0;
        background: #f7f7f7;
        color: #333;
        min-width: 120px;  /* <-- Adjust this value to set tab width */
        /* You can also use width: 200px; for fixed width */
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: #fc6c64;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce")

def get_file_details():
    files = []
    for file in [f for f in os.listdir('.') if f.endswith('.csv')]:
        if os.path.getsize(file) == 0:
            continue  # Skip empty files
        try:
            df = pd.read_csv(file)
            files.append({'Filename': file, 'Rows (Excluding Headers)': len(df)})
        except Exception as e:
            st.warning(f"Error reading {file}: {e}")
    return files

with tab1:
    st.header("Overview of Data Files")

    details = get_file_details()
    st.subheader("Uploaded CSV Files")
    st.table(pd.DataFrame(details if details else [{"Filename": "", "Rows (Excluding Headers)": 0}]))
    st.caption("If you have just uploaded a file, please refresh the page to see it listed here. Newly uploaded files are pushed to GitHub and not automatically loaded into the local directory until the app is restarted or refreshed.")

    st.subheader("Preview of Filtered Data (First 100 Rows)")
    st.dataframe(filtered_data.head(100), use_container_width=True)

    st.subheader("Missing Values by Column")
    missing_counts = filtered_data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if not missing_counts.empty:
        missing_df = pd.DataFrame({'Column': missing_counts.index, 'Missing Values': missing_counts.values})
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("No missing values in the filtered dataset.")

    st.markdown("### 🧠 Insight")
    num_files = len(details) if details else 0
    num_rows = len(filtered_data)
    num_cols = len(filtered_data.columns)
    num_missing_cols = missing_counts.shape[0]
    summary_text = (
        f"A total of **{num_files} file{'s' if num_files != 1 else ''}** have been uploaded. "
        f"The filtered dataset currently holds **{num_rows:,} rows** and **{num_cols} columns**, "
        f"providing a substantial view of your team's logged activity.\n\n"
        f"{'There are **' + str(num_missing_cols) + ' column' + ('s' if num_missing_cols != 1 else '') + '** with missing values that may need attention.' if num_missing_cols > 0 else '✅ No missing values detected—your data looks clean and complete.'}\n\n"
        f"To assist with review, the first 100 rows are displayed above as a quick preview."
    )
    st.info(summary_text)


with tab2:
    st.subheader(" 📊 User Summary")
    if not filtered_data.empty:
        user_summary = (
            filtered_data
            .groupby(['user_first_name', 'user_last_name'])
            .agg(
                Total_Minutes=('minutes', 'sum'),
                Task_Count=('minutes', 'count'),
                Avg_Minutes_Per_Task=('minutes', 'mean')
            )
            .reset_index()
        )
        user_summary['Avg_Minutes_Per_Task'] = user_summary['Avg_Minutes_Per_Task'].round(2)
        user_summary.columns = ['First Name', 'Last Name', 'Total Minutes', 'Task Count', 'Avg Minutes/Task']

        # Sort by Total Minutes, then Task Count, then Avg Minutes/Task (all descending)
        user_summary = user_summary.sort_values(
            by=['Total Minutes', 'Task Count', 'Avg Minutes/Task'],
            ascending=[False, False, False]
        )

        st.dataframe(user_summary, use_container_width=True)
        st.download_button(
            label="📥 Download User Summary",
            data=user_summary.to_csv(index=False),
            file_name="user_summary.csv"
        )
        total_minutes = filtered_data['minutes'].sum()
        avg_minutes = filtered_data['minutes'].mean()
        total_tasks = filtered_data.shape[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Time Spent (min)", total_minutes)
        col2.metric("Average Time per Task (min)", round(avg_minutes, 2))
        col3.metric("Total Tasks", total_tasks)
        st.markdown("### 🧠 Insight")
        top_user = user_summary.iloc[0]['First Name'] if not user_summary.empty else None
        top_minutes = user_summary.iloc[0]['Total Minutes'] if not user_summary.empty else None
        top_tasks = user_summary.iloc[0]['Task Count'] if not user_summary.empty else None
        summary_text = (
            f"Across the selected date range, a total of **{total_minutes} minutes** were logged across **{total_tasks} tasks**.\n\n"
            f"Notably, **{top_user}** emerged as the top contributor with **{top_minutes} minutes** spent on **{top_tasks} tasks**, "
            f"suggesting a consistently high engagement level.\n\n"
            f"These metrics offer a holistic view of team workload, individual contribution, and time investment per task."
        )
        st.info(summary_text)
    else:
        st.info("No data to show in User Summary.")

with tab3:
    st.markdown("## 📈  Main Visualizations")

    # Time grouping
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce").dt.tz_localize(None)
    filtered_data = filtered_data.sort_values("started_at", ascending=True)

    if time_group == "Yearly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.year.astype(str)
    elif time_group == "Monthly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.strftime('%b %Y')
    elif time_group == "Weekly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.to_period("W").apply(lambda p: p.start_time.strftime("%b %-d, %Y"))
    elif time_group == "Daily":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.strftime('%b %-d, %Y')

    # Group data
    grouped = filtered_data.groupby(["Full_Name", "TimeGroup"])['Hours'].sum().reset_index()
    grouped["TimeGroupSort"] = pd.to_datetime(grouped["TimeGroup"].str.split("/").str[0], errors='coerce')
    grouped = grouped.sort_values("TimeGroupSort")

    # POPIN color palette
    popin_palette = ['#fc6c64', '#fa928b', '#fbbcb6', '#fcdad7', '#f7a59d', '#d94d44', '#a13630']

    st.subheader(f"Hours Count by User - {time_group} View")
    fig_timegroup = px.bar(
        grouped,
        x="TimeGroup",
        y="Hours",
        color="Full_Name",
        barmode="group",
        title=f"Accumulated Hours per User - {time_group}",
        labels={"TimeGroup": "Time", "Hours": "Total Hours"},
        color_discrete_sequence=popin_palette,
        height=500
    )
    fig_timegroup.update_xaxes(type="category", tickangle=-45)
    st.plotly_chart(fig_timegroup, use_container_width=True)

    # Average Hours per Day of the Week (Bars per User)
    if not filtered_data.empty:
        filtered_data['Day'] = filtered_data['started_at'].dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        avg_hours_per_user = (
            filtered_data.groupby(['Full_Name', 'Day'])['Hours']
            .mean()
            .reset_index()
        )
        avg_hours_per_user['Day'] = pd.Categorical(avg_hours_per_user['Day'], categories=days_order, ordered=True)
        avg_hours_per_user = avg_hours_per_user.sort_values('Day')

        fig_avg_hours = px.bar(
            avg_hours_per_user,
            x='Day',
            y='Hours',
            color='Full_Name',
            barmode='group',
            title="Average Hours per Day of the Week by User",
            labels={'Day': 'Day of the Week', 'Hours': 'Average Hours', 'Full_Name': 'User'},
            color_discrete_sequence=popin_palette,
            height=500
        )
        fig_avg_hours.update_xaxes(categoryorder='array', categoryarray=days_order)
        st.plotly_chart(fig_avg_hours, use_container_width=True)
    else:
        st.info("No data available for the average hours per day of the week chart.")

    st.subheader(" ⏱️ Task Duration Distribution and Outliers")
    if not filtered_data.empty and 'minutes' in filtered_data.columns:
        # Histogram
        fig_hist = px.histogram(filtered_data, x='minutes', nbins=30, title="Histogram of Task Durations")
        st.plotly_chart(fig_hist, use_container_width=True)
        # Boxplot
        fig_box = px.box(filtered_data, y='minutes', title="Boxplot of Task Durations")
        st.plotly_chart(fig_box, use_container_width=True)

        # Outlier calculation
        Q1 = filtered_data['minutes'].quantile(0.25)
        Q3 = filtered_data['minutes'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = filtered_data[(filtered_data['minutes'] < lower_bound) | (filtered_data['minutes'] > upper_bound)]

        st.markdown(f"**Number of outliers:** {outliers.shape[0]}")
        cols_to_show = ['date', 'user_first_name', 'task', 'minutes']
        available_cols = [col for col in cols_to_show if col in outliers.columns]
        if not outliers.empty and available_cols:
            st.dataframe(outliers[available_cols], use_container_width=True)
        elif not outliers.empty:
            st.dataframe(outliers, use_container_width=True)
        else:
            st.info("No outlier data available for the selected columns.")

        # Insight Section
        st.markdown("### 🧠  Insight")
        shortest = round(filtered_data['minutes'].min(), 2)
        longest = round(filtered_data['minutes'].max(), 2)
        duration_summary = (
            f"Task durations range widely—from **{shortest} to {longest} minutes**—indicating a mix of quick wins and deeper work.\n\n"
            f"A total of **{outliers.shape[0]} tasks** were flagged as outliers, either exceptionally short or unusually long. "
            f"This could signal errors, bottlenecks, or tasks that deserve process review.\n\n"
            f"Histogram and boxplot distributions help assess whether most tasks fall within acceptable time bands."
        )
        st.info(duration_summary)
    else:
        st.info("No data available for plotting task duration distribution.")
        
    # User Achievements
    if 'user_first_name' in filtered_data.columns and 'user_achievements_count' in filtered_data.columns:
        # Show the max achievement count per user (not sum)
        achievement_df = filtered_data.groupby('user_first_name')['user_achievements_count'].max().reset_index()
        # Use a professional color palette: Plotly's 'Blues'
        fig_ach = px.bar(
            achievement_df,
            x='user_first_name',
            y='user_achievements_count',
            title="Max Achievements per User",
            color='user_achievements_count',
            color_continuous_scale='Blues',
            labels={'user_first_name': 'User', 'user_achievements_count': 'Max Achievements'}
        )
        st.plotly_chart(fig_ach, use_container_width=True)

       # Ensure datetime is clean
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce").dt.tz_localize(None)
    filtered_data = filtered_data.sort_values("started_at")

     # Grouping based on selected time range
    if time_group == "Yearly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.year.astype(str)
    elif time_group == "Monthly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.strftime('%b %Y')
    elif time_group == "Weekly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.to_period("W").apply(lambda p: p.start_time.strftime("%b %-d, %Y"))
    elif time_group == "Daily":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.strftime('%b %-d, %Y')

    # Group by user and time, then COUNT rows
    grouped_count = (
        filtered_data.groupby(["Full_Name", "TimeGroup"])
        .size()
        .reset_index(name="Task Count")
    )

    # Sort time groups
    grouped_count["TimeGroupSort"] = pd.to_datetime(grouped_count["TimeGroup"].str.split(",").str[-1], errors='coerce')
    grouped_count = grouped_count.sort_values("TimeGroupSort")

    # Color palette for users
    unique_users = grouped_count["Full_Name"].nunique()
    color_palette = get_color_palette(unique_users)

    # Plotting
    st.subheader(f"📊 Task Count per User - {time_group} View")

    fig = px.bar(
        grouped_count,
        x="TimeGroup",
        y="Task Count",
        color="Full_Name",
        barmode="group",
        title=f"Number of Tasks Logged by Users ({time_group})",
        labels={"TimeGroup": "Time", "Task Count": "Number of Tasks"},
        color_discrete_sequence=color_palette,
        height=500
    )

    fig.update_xaxes(type="category", tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# --- Place this after your other tab definitions, inside your main app file ---
with tab4:
    st.header("🏆 Top & Bottom 5 User Time Stats")
    
    required_cols = ['Full_Name', 'minutes']
    missing_cols = [col for col in required_cols if col not in filtered_data.columns]
    if missing_cols:
        st.error(f"Missing columns in data: {missing_cols}")
        st.write("Available columns:", filtered_data.columns.tolist())
        st.stop()
    
    if filtered_data.empty:
        st.info("No data available. Please upload data or adjust your filters.")
    else:
        # Total time per user
        total_time = filtered_data.groupby('Full_Name')['minutes'].sum().reset_index()
        avg_time = filtered_data.groupby('Full_Name')['minutes'].mean().reset_index()
    
        # Top and bottom 5 by total time spent
        top5_total = total_time.sort_values('minutes', ascending=False).head(5)
        bottom5_total = total_time.sort_values('minutes', ascending=True).head(5)
    
        # Top and bottom 5 by average time spent
        top5_avg = avg_time.sort_values('minutes', ascending=False).head(5)
        bottom5_avg = avg_time.sort_values('minutes', ascending=True).head(5)
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Top 5 Users by Total Time Spent")
            st.dataframe(top5_total.rename(columns={'Full_Name': 'User', 'minutes': 'Total Minutes'}), use_container_width=True)
            fig1 = px.bar(top5_total, x='Full_Name', y='minutes',
                        title='Top 5 Users by Total Time Spent', labels={'Full_Name': 'User', 'minutes': 'Total Minutes'})
            st.plotly_chart(fig1, use_container_width=True)
    
            st.subheader("Top 5 Users by Average Time Spent")
            st.dataframe(top5_avg.rename(columns={'Full_Name': 'User', 'minutes': 'Avg Minutes'}), use_container_width=True)
            fig3 = px.bar(top5_avg, x='Full_Name', y='minutes',
                        title='Top 5 Users by Average Time Spent', labels={'Full_Name': 'User', 'minutes': 'Avg Minutes'})
            st.plotly_chart(fig3, use_container_width=True)
    
        with col2:
            st.subheader("Lowest 5 Users by Total Time Spent")
            st.dataframe(bottom5_total.rename(columns={'Full_Name': 'User', 'minutes': 'Total Minutes'}), use_container_width=True)
            fig2 = px.bar(bottom5_total, x='Full_Name', y='minutes',
                        title='Lowest 5 Users by Total Time Spent', labels={'Full_Name': 'User', 'minutes': 'Total Minutes'})
            st.plotly_chart(fig2, use_container_width=True)
    
            st.subheader("Lowest 5 Users by Average Time Spent")
            st.dataframe(bottom5_avg.rename(columns={'Full_Name': 'User', 'minutes': 'Avg Minutes'}), use_container_width=True)
            fig4 = px.bar(bottom5_avg, x='Full_Name', y='minutes',
                        title='Lowest 5 Users by Average Time Spent', labels={'Full_Name': 'User', 'minutes': 'Avg Minutes'})
            st.plotly_chart(fig4, use_container_width=True)
            
        st.markdown("### 🧠 Insight")
        if not top5_total.empty and not bottom5_total.empty and not top5_avg.empty and not bottom5_avg.empty:
            top_total_user = top5_total.iloc[0]
            bottom_total_user = bottom5_total.iloc[0]
            top_avg_user = top5_avg.iloc[0]
            bottom_avg_user = bottom5_avg.iloc[0]
            summary_text = (
                f"🏅 **{top_total_user['Full_Name']}** recorded the highest total time with **{top_total_user['minutes']:.1f} minutes**, "
                f"demonstrating strong overall contribution.\n\n"
                f"📉 On the other end, **{bottom_total_user['Full_Name']}** had the lowest total time at **{bottom_total_user['minutes']:.1f} minutes**, "
                f"which may suggest limited engagement, lighter workload, or inconsistent logging.\n\n"
                f"⏱️ For average time per task, **{top_avg_user['Full_Name']}** led with **{top_avg_user['minutes']:.1f} minutes**, "
                f"possibly reflecting complex assignments or deeper task involvement.\n\n"
                f"⚡ Meanwhile, **{bottom_avg_user['Full_Name']}** averaged just **{bottom_avg_user['minutes']:.1f} minutes** per task, "
                f"hinting at either rapid task completion or fragmented time tracking."
            )
            st.info(summary_text) # This line is inside the 'if' block
        else: # This 'else' now correctly matches the 'if' above it
            st.warning("Not enough data available to generate insight summaries.")
    

    with tab5:
        st.subheader(" ☁️ Word Cloud and Treemap of Tasks")
    
        # Check for required columns first
        required_cols = ['task', 'task_wo_punct_split_wo_stopwords_lemmatized']
        missing_cols = [col for col in required_cols if col not in filtered_data.columns]
        if missing_cols:
            st.error(f"Missing columns in data: {missing_cols}")
            st.write("Available columns:", filtered_data.columns.tolist())
            st.stop()
    
        # Word Cloud generation using original 'task' column
        tasks_series = filtered_data['task'].dropna().astype(str)
        text = " ".join(tasks_series.values)
    
        if not text.strip():
            st.info("No task data available for word cloud or treemap.")
        else:
            # Word Cloud plot
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)
    
            # Word Cloud Insight below word cloud
            task_counts = tasks_series.value_counts()
            top_wc_task = task_counts.index[0] if not task_counts.empty else None
            wc_summary = (
                f"The word cloud visualizes the most frequently logged tasks. "
                f"**{top_wc_task}** appears most often, suggesting it's central to team operations.\n\n"
                f"Frequent mentions may reflect routine responsibilities, while missing or rare task types could indicate under-reporting "
                f"or areas with less activity.\n\n"
                f"Use this to understand recurring themes or evaluate if task tracking is comprehensive."
                if top_wc_task else "No task text was available to analyze frequency trends."
            )
            st.markdown("### 🧠 Insight")
            st.info(wc_summary)
    
            # TREEMAP based on exploded lemmatized words column
            exploded_lemmas = filtered_data.explode('task_wo_punct_split_wo_stopwords_lemmatized')
            exploded_lemmas_series = exploded_lemmas['task_wo_punct_split_wo_stopwords_lemmatized'].dropna().astype(str)
            lemma_counts = exploded_lemmas_series.value_counts().head(50)
    
            if not lemma_counts.empty:
                top_words_df = pd.DataFrame({
                    'Word': lemma_counts.index,
                    'Count': lemma_counts.values
                })
    
                treemap_fig = px.treemap(
                    top_words_df,
                    path=['Word'],
                    values='Count',
                    color='Count',
                    color_continuous_scale='Greens',
                    title="Top 50 Lemmatized Words Treemap"
                )
                st.plotly_chart(treemap_fig, use_container_width=True)
    
                # Treemap Insight below treemap
                most_common_lemma = lemma_counts.idxmax()
                least_common_lemma = lemma_counts.index[-1]
                treemap_insight = (
                    f"The treemap highlights the **top 50 most frequent lemmatized words** by size and color intensity. "
                    f"**{most_common_lemma}** is the largest segment, indicating it is the dominant word within this subset.\n\n"
                    f"Words like **{least_common_lemma}** appear less frequently but are still significant enough to be in the top 50. "
                    f"This visualization helps identify key themes and variations of task descriptions."
                )
                st.markdown("### 🧠 Treemap Insight")
                st.info(treemap_insight)
            else:
                st.info("No lemmatized word frequency data available for the selected filters.")

with tab6:
    st.subheader("👤 User Drilldown")
    st.caption("Sidebar filters (including user selection) affect the data below.")

    # Ensure 'date' column exists
    if 'started_at' in filtered_data.columns:
        filtered_data['date'] = pd.to_datetime(filtered_data['started_at'], errors='coerce').dt.date

    # Check required columns
    required_cols = ['Full_Name', 'Hours', 'date', 'task']
    missing_cols = [col for col in required_cols if col not in filtered_data.columns]
    if missing_cols:
        st.error(f"Missing columns in data: {missing_cols}")
        st.write("Available columns:", filtered_data.columns.tolist())
        st.stop()

    # Get users from filtered dataset
    user_list = filtered_data['Full_Name'].dropna().unique()
    if len(user_list) == 0:
        st.info("No users available for drilldown with current filters.")
        st.stop()

    if len(user_list) == 0:
        st.info("No users available for drilldown with current filters. Please adjust your filters.")
        st.stop()

    if len(user_list) > 1:
        st.warning("Multiple users are currently selected. Please filter one user to view individual insights.")
        st.stop()

    # Single user for drilldown
    selected_user = user_list[0]
    user_df = filtered_data[filtered_data['Full_Name'] == selected_user]

    # Summary metrics
    col1, col2 = st.columns(2)
    total_minutes = int(user_df['minutes'].sum())
    avg_minutes = round(user_df['minutes'].mean(), 2) if not user_df.empty else 0
    col1.metric("Total Minutes", total_minutes)
    col2.metric("Average Task Time", avg_minutes)

    # Task details
    st.markdown(f"### 📝 Task Details of {selected_user}")
    cols_to_show_user = ['date', 'task', 'Hours', 'minutes']
    available_cols_user = [col for col in cols_to_show_user if col in user_df.columns]
    if available_cols_user:
        # Use pandas Styler to apply color scale to 'Hours' and 'minutes'
        styled_df = (
        user_df[available_cols_user]
        .sort_values('date', ascending=True)
        .style
        .background_gradient(subset=['Hours'], cmap='cividis')
        .background_gradient(subset=['minutes'], cmap='cividis')
        )
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No available columns to display for task details.")

    # Main Insight
    st.markdown("### 🧠 Insight")
    user_task_count = user_df.shape[0]
    user_total = int(user_df['minutes'].sum())

    user_summary = (
        f"**{selected_user}** has completed **{user_task_count} tasks** totaling **{user_total} minutes**. "
        f"Their task time distribution offers insight into workload balance.\n\n"
        f"If one task type dominates, it may highlight specialization or possible over-dependence on this user for certain duties.\n\n"
        f"Use this view to evaluate both individual performance and role focus."
    )
    st.info(user_summary)

    # 📌 Additional Insights
    st.markdown("### 📌 Additional Insights")

    if user_df.empty:
        st.info("Not enough data to generate additional insights.")
    else:
        # Most time-consuming task
        top_task_row = (
            user_df.groupby('task')['minutes']
            .sum()
            .reset_index()
            .sort_values(by='minutes', ascending=False)
            .head(1)
        )
        top_task = top_task_row['task'].values[0]
        top_task_minutes = round(top_task_row['minutes'].values[0], 2)

        # Most productive day (by total time)
        productive_day = (
            user_df.groupby('date')['minutes']
            .sum()
            .reset_index()
            .sort_values(by='minutes', ascending=False)
            .head(1)
        )
        top_day = productive_day['date'].values[0]
        top_day_minutes = round(productive_day['minutes'].values[0], 2)

        # Average tasks per day
        tasks_per_day = user_df.groupby('date').size()
        avg_tasks_per_day = round(tasks_per_day.mean(), 2)

        # Best day of the week
        user_df['day_of_week'] = pd.to_datetime(user_df['date']).dt.day_name()
        dow_avg = user_df.groupby('day_of_week')['minutes'].mean().reset_index()
        dow_avg = dow_avg.sort_values(by='minutes', ascending=False)
        best_dow = dow_avg.iloc[0]['day_of_week']
        best_dow_avg = round(dow_avg.iloc[0]['minutes'], 2)

        # Render insight list
        st.markdown(f"""
        - 🧱 **Most Time-Consuming Task:** `{top_task}` with **{top_task_minutes} minutes** logged.
        - 📅 **Most Productive Day:** `{top_day}` with **{top_day_minutes} minutes** logged.
        - 📊 **Average Tasks per Day:** {avg_tasks_per_day}
        - 📈 **Best Performing Day of Week:** **{best_dow}** (Average time spent: **{best_dow_avg} minutes**)
        """)

        st.success("These insights can help identify user strengths, optimal working days, and task load distribution.")


    with tab6:
        pass

    with tab7:
        st.subheader(" ☁️ Word Cloud of Tasks")

        # Generate text for word cloud
        text = " ".join(filtered_data['task'].dropna().astype(str).values)
        if not text.strip():
            st.info("No task data available for word cloud.")
        else:
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        # AI Insight
        st.markdown("### 🧠 Insight")
        task_counts = filtered_data['task'].value_counts()
        top_wc_task = task_counts.index[0] if not task_counts.empty else None

        wc_summary = (
            f"The word cloud visualizes the most frequently logged tasks. "
            f"**{top_wc_task}** appears most often, suggesting it's central to team operations.\n\n"
            f"Frequent mentions may reflect routine responsibilities, while missing or rare task types could indicate under-reporting "
            f"or areas with less activity.\n\n"
            f"Use this to understand recurring themes or evaluate if task tracking is comprehensive."
            if top_wc_task else "No task text was available to analyze frequency trends."
        )
        st.info(wc_summary)
