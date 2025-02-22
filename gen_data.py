import pandas as pd

# Load the dataset
data = pd.read_csv('Tweets.csv')

# Select relevant columns
data = data[['text', 'airline_sentiment']]
data.columns = ['text', 'sentiment']

# Save to your raw_data.csv
data.to_csv('data/raw_data.csv', index=False)