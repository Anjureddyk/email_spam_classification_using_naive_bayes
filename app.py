from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Route to handle the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the classification request
@app.route('/classify', methods=['POST'])
def classify():
    # Get the email content from the form input
    email_content = request.form['Body']

    # Load the training data
    training_data = pd.read_csv('completeSpamAssassin.csv')

    # Fill missing values with empty string
    training_data['Body'].fillna('', inplace=True)

    # Create a CountVectorizer to convert text into features
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(training_data['Body'])

    # Train the Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(features, training_data['Label'])

    # Convert the email content into features using the trained CountVectorizer
    email_features = vectorizer.transform([email_content])

    # Make the classification prediction
    prediction = classifier.predict(email_features)[0]

    # Map the prediction to the corresponding label
    classification_result = "Spam" if prediction == 1 else "Not Spam"

    # Render the result page with the classification result
    return render_template('result.html', result=classification_result)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
