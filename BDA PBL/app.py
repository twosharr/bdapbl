from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('emotions.csv')

# Pre-processing
stop_words = set(stopwords.words('english'))
punctuation = string.punctuation

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in punctuation]
    return ' '.join(tokens)

data['clean_text'] = data['Sentences'].apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Label'])

# Sentiment Classification
classifier = LinearSVC()
classifier.fit(X, y)

# Evaluate the model
y_pred = classifier.predict(X)
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred, target_names=label_encoder.classes_)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    vectorized_text = vectorizer.transform([text])
    prediction = classifier.predict(vectorized_text)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return render_template('index.html', text=text, predicted_label=predicted_label, accuracy=accuracy, classification_rep=classification_rep)

if __name__ == '__main__':
    app.run(debug=True)
