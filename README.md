# email_spam_classification_using_naive_bayes
This code is a Flask application that can be used to classify email messages as spam or not spam. The application uses a Naive Bayes classifier that is trained on a dataset of spam and not spam email messages. 

## Features
Integration with Flask web framework for seamless user interaction.

Utilization of the Naive Bayes classifier, specifically the Multinomial Naive Bayes algorithm, known for its effectiveness in text classification tasks.

Preprocessing of training data using pandas to handle missing values.

Conversion of textual data into numerical features using the CountVectorizer from scikit-learn.

Training the classifier on a labeled dataset of spam and non-spam emails.

Real-time classification of incoming emails based on learned patterns and characteristics.

Mapping of classification results to human-readable labels (Spam or Not Spam).

Evaluation of system performance using metrics such as accuracy, precision, recall, and F1-score.

Robustness to imbalanced datasets and generalizability across various email sources and formats.

## Requirements
Python (>=3.6)

Flask (>=2.0.0)

scikit-learn (>=0.24.0)

pandas (>=1.2.0)





