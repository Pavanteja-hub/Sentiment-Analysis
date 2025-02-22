from flask import Flask, render_template, request
import joblib
from src.data_processing import preprocess_text

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = joblib.load('models/logistic_regression_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if model is None or vectorizer is None:
            return render_template('index.html', sentiment="Model not loaded. Please check the logs.")
        
        text = request.form['text']
        processed_text = preprocess_text(text)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        return render_template('index.html', sentiment=prediction)

if __name__ == '__main__':
    app.run(debug=True)