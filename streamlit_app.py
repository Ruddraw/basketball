import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
import numpy as np
import seaborn as sns
import requests


st.title("NBA Player Stats Explorer")

st.markdown("""
this pp usese simple web scraping of NBA player's stat data.
* **Python libraries:** base64, Pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/)
""")

st.sidebar.header("User input features")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2024))))

# web scraping


@st.cache_data
def load_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    response = requests.get(url)  # Fetch the content of the URL
    response.raise_for_status()  # Check for HTTP errors
    html = response.text  # Get the HTML content as a string
    df = pd.read_html(html, header=0)[0]  # Parse the HTML content
    raw = df.drop(df[df.Age == 'Age'].index)  # Remove repeating headers
    raw = raw.fillna(0)  # Fill missing values with 0
    playerstats = raw.drop(['Rk'], axis=1)  # Drop unnecessary columns
    return playerstats


playerstats = load_data(selected_year)

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect(
    'Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(
    selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(
    df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download NBA player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806


def filedownload(df):
    csv = df.to_csv(index=False)
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href


st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)


# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')

    # Read the CSV file
    df = pd.read_csv('output.csv')

    # Select only numeric columns for correlation
    df_numeric = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr = df_numeric.corr()

    # Create mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(7, 5))  # Create a figure and axis
    sns.heatmap(corr, mask=mask, vmax=1, square=True,
                annot=True, cmap='coolwarm', ax=ax)

    # Display the plot in Streamlit
    st.pyplot(fig)
