import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Define sentiments
sentiments = ['positive', 'negative', 'neutral']

# Generate synthetic data
def generate_synthetic_data(num_samples=10000):
    data = []
    for _ in range(num_samples):
        text = fake.sentence()  # Generate a fake sentence
        sentiment = random.choice(sentiments)  # Randomly assign a sentiment
        data.append({'text': text, 'sentiment': sentiment})
    return pd.DataFrame(data)

# Generate 10,000 samples
synthetic_data = generate_synthetic_data(num_samples=10000)

# Save to CSV
synthetic_data.to_csv('data/raw_data.csv', index=False)
print("Synthetic data saved to 'data/raw_data.csv'")