"""
Spam Detector using Naive Bayes
This script uses machine learning to classify emails as spam or not spam (ham).
It trains a Naive Bayes classifier on a pre-existing dataset of emails and their labels.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the spam dataset
df=pd.read_csv('spam.csv', encoding='latin-1')

# Prepare features (email text) and labels (spam/ham)
x=df[['v2']]  # v2 contains the email text
y=df['v1']    # v1 contains the labels (spam or ham)

# Convert text to numerical features using CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(x.values.ravel())

# Create and train the Naive Bayes model
model=MultinomialNB()
model.fit(x,y)

# Get user input
input_mail=input('enter mail :')

# Convert user input to numerical features
mail=cv.transform([input_mail])

# Make prediction
s=model.predict(mail)

# Display result
if s[0]=='ham':
    print('not a Spam')
else:
    print('Spam')