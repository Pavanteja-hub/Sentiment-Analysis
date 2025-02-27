import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stopword list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize each word to its base form
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin the tokens into a single string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


# Load raw data
raw_data = pd.read_csv('data/raw_data.csv')

# Preprocess the text data
raw_data['text'] = raw_data['text'].apply(preprocess_text)

# Save processed data
raw_data.to_csv('data/processed_data.csv', index=False)
print("Processed data saved to 'data/processed_data.csv'")
