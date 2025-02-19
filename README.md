Soccer Player Similarity Finder
This project is designed to help you explore and find similar soccer players based on a combination of their statistics and position data. It uses both numeric and categorical data from soccer player datasets and calculates similarity between players using distance metrics (e.g., cosine similarity, Gower distance).

Features
Data Preprocessing:

Scale numeric features (e.g., height, weight, goals, etc.) using StandardScaler.
One-hot encode categorical features (e.g., preferred foot, playing positions).
Combine numeric and categorical data into a single feature matrix.
Distance Calculation:

Compute pairwise distances using various metrics such as cosine distance (or Gower distance for mixed data).
Retrieve the top 10 nearest neighbors for a given player.
Interactive Web App:

Built with Streamlit, allowing you to search for a player by name.
Displays similar players along with similarity/distance scores.

Installation
Clone the Repository:

bash
Copy
git clone https://github.com/yourusername/your-repository.git
cd your-repository
Create a Virtual Environment (Optional):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

pip install -r requirements.txt
Dependencies may include packages such as:

pandas
numpy
scikit-learn
streamlit
(and optionally gower if you use Gower distance)
Usage
Data Preparation:

The project includes scripts to preprocess your soccer player data:

Scaling Numeric Data: Numeric columns (e.g., height, weight) are scaled for equal contribution in distance calculations.
One-Hot Encoding: Categorical fields like playing positions and preferred foot are converted into dummy variables.
Feature Combination: The processed numeric and categorical data are merged to form the final feature matrix.
Running the Streamlit App:

Start the interactive app by running:


streamlit run app.py
In the app:

Type the name of a soccer player.
Choose the correct player from suggestions.
View the top 10 similar players based on their statistical and positional data.
Distance Calculation:

The app uses functions to compute pairwise distances and then returns the nearest neighbors for the selected player. You can adjust the distance metric (cosine, Gower, or a custom weighted distance) based on your preference.

Contributing
Contributions are welcome! If you find issues or have suggestions for improvement, please open an issue or submit a pull request.
