# %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("maso0dahmed/football-players-data")

# print("Path to dataset files:", path)

# %%
import gower
import pandas as pd
df  = pd.read_csv('/Users/baonguyen/.cache/kagglehub/datasets/maso0dahmed/football-players-data/versions/1/fifa_players.csv')

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
# One-hot encode 'positions' with a prefix
position_dummies = df['positions'].str.get_dummies(sep=',').add_prefix('pos_')

# One-hot encode 'national_team_position' with a different prefix
ntp_dummies = df['national_team_position'].str.get_dummies(sep=',').add_prefix('ntp_')

# Concatenate the dummy dataframes to the original dataframe
df = pd.concat([df, position_dummies, ntp_dummies], axis=1)

# Optionally, drop the original columns
df.drop(['positions', 'national_team_position'], axis=1, inplace=True)


# %%
from sklearn.preprocessing import LabelEncoder

# Initialize the encoder
le = LabelEncoder()

# Fit and transform the 'preferred_foot' column, creating a new column
df['preferred_foot_encoded'] = le.fit_transform(df['preferred_foot'])
df.drop('preferred_foot',axis=1,inplace=True)


# %%
for col in df.columns:
    mode_val = df[col].mode()[0]  # get the first mode value
    df[col].fillna(mode_val, inplace=True)


# %%
# drop uneccessary cols
df.drop(['birth_date','nationality','wage_euro','value_euro','body_type','release_clause_euro','national_team','national_jersey_number'],axis=1,inplace=True)


# %%
df.head()

# %%
from sklearn.preprocessing import StandardScaler
# Select numeric columns
# Select all numeric columns first
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Filter out columns that start with 'pos' or 'ntp'
filtered_numeric_cols = [col for col in numeric_cols if not (col.startswith('pos') or col.startswith('ntp') or col.endswith('(1-5)'))]

print(filtered_numeric_cols)

# Scale the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[filtered_numeric_cols])

# Convert the scaled data back into a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in filtered_numeric_cols])

# Merge the scaled data back into the original DataFrame
df = pd.concat([df, scaled_df], axis=1)

# Now drop the original numeric columns
df.drop(columns=filtered_numeric_cols, inplace=True)

# %%
df.head()

# %%
from sklearn.metrics import pairwise_distances

# Extract only the numeric columns for positions
clean_df = df.drop(['full_name','name','scaled_age','scaled_height_cm','scaled_weight_kgs'], axis=1).values


# %%
import numpy as np
import gower
# Calculate the pairwise distances (using Euclidean as an example)
dist_matrix = gower.gower_matrix(clean_df)
# dist_matrix_gower = gower.gower_matrix(clean_df)
def get_top_n_nearest(dist_matrix, index, n=10):
    """
    Returns the indices and distances of the top n nearest neighbors
    for the given index (excluding the index itself).
    """
    # Get the distances for the given index
    distances = dist_matrix[index]
    
    # Sort the indices based on distance
    sorted_indices = np.argsort(distances)
    
    # Remove the index itself (distance = 0)
    sorted_indices = sorted_indices[sorted_indices != index]
    
    # Select the top n indices and their corresponding distances
    top_n_indices = sorted_indices[:n]
    top_n_distances = distances[top_n_indices]
    
    return top_n_indices, top_n_distances



# %%
# Example: For player at index 0, get top 10 nearest neighbors
# index = 0  # change this to any valid index
# nearest_indices, nearest_distances = get_top_n_nearest(dist_matrix, index, n=10)

# print(f"Top 10 nearest neighbors for index {index}:")
# for i, (idx, dist) in enumerate(zip(nearest_indices, nearest_distances), start=1):
#     print(f"{i}. Index: {idx}, Distance: {dist}")
# print('\n\n')
# print(df.loc[index, 'name'])

# for i,idx in enumerate(nearest_indices):
#     print(i+1,'.',df.loc[idx, 'name'])


# %%
import streamlit as st
# --- Streamlit App ---
st.title("Player Nearest Neighbors Finder")

# Text input to enter a player's name
player_input = st.text_input("Enter player's name:")

if player_input:
    # Filter df for names matching the input (case insensitive)
    suggestions = df[
    (df['full_name'].str.contains(player_input, case=False, na=False)) | 
    (df['name'].str.contains(player_input, case=False, na=False))
]
    
    if not suggestions.empty:
        # Display a selectbox with the suggestions
        selected_player = st.selectbox("Select a player", suggestions['full_name'].tolist())
        
        # Find the index of the selected player in the DataFrame
        player_index = df[df['full_name'] == selected_player].index[0]
        
        st.write(f"Selected player: **{selected_player}**")
        
        # Get the top 10 nearest neighbors using the precomputed distance matrix
        nearest_indices, nearest_distances = get_top_n_nearest(dist_matrix, player_index, n=10)
        
        st.subheader("Top 10 Nearest Neighbors:")
        for idx, dist in zip(nearest_indices, nearest_distances):
            st.write(f"- **{df.loc[idx, 'name']}** (Distance: {dist:.2f})")
    else:
        st.write("No player found matching that name.")


