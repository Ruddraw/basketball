import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
import numpy as np
import seaborn as sns
import requests

st.title("NBA Player Stats Explorer")

st.markdown("""
This app uses simple web scraping of NBA player's stat data.
* **Python libraries:** base64, Pandas, Streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/)
""")

st.sidebar.header("User input features")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2024))))

# Web scraping

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
    
    # Convert percentage columns to numeric by removing '%'
    for col in ['FG%', '3P%', '2P%', 'eFG%', 'FT%']:
        playerstats[col] = playerstats[col].replace('%', '', regex=True).astype(float) / 100
    
    # Convert other columns to numeric if possible
    numeric_cols = ['G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    for col in numeric_cols:
        playerstats[col] = pd.to_numeric(playerstats[col], errors='coerce')
    
    return playerstats

playerstats = load_data(selected_year)

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download NBA player stats data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')

    # Filter out non-numeric columns
    df_numeric = df_selected_team.select_dtypes(include=[np.number])

    if df_numeric.empty:
        st.write("No numeric data available for correlation.")
    else:
        corr = df_numeric.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the heatmap
        sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True, cmap='coolwarm', ax=ax)
        
        # Display the heatmap
        st.pyplot(fig)
