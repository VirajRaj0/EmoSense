#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import re


# Install required libraries if not already installed

import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer

from wordcloud import WordCloud

import preprocessor as p

from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
# Define the data directory
data_dir = "./input"

# Check the contents of the data directory
print("Contents of the data directory:")
print(os.listdir(data_dir))

# Load the Sentiment140 dataset
encoding = 'ISO-8859-1'
col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
dataset = pd.read_csv(os.path.join(data_dir, 'sentiment140/training.1600000.processed.noemoticon.csv'), encoding=encoding, names=col_names)

# Display a sample of the dataset
print("\nSample of the Sentiment140 dataset:")
print(dataset.head())

# Sample 8000 tweets from the dataset for the analysis
df = dataset.copy().sample(8000, random_state=42)
df["label"] = 0
df = df[['text', 'label']]
df.dropna(inplace=True)

# Load the dataset containing depressive tweets scraped by Twint
col_names = ['id', 'text']
df2 = pd.read_csv(os.path.join(data_dir, 'depressive-tweets-processed/depressive_tweets_processed.csv'), sep='|', header=None, usecols=[0, 5], nrows=3200, names=col_names)
df2['label'] = 1
df2 = df2[['text', 'label']]

# Concatenate the normal and depressive tweets dataset
df = pd.concat([df, df2])
df = df.sample(frac=1)  # Shuffle the dataset

# Define a function to expand contractions in tweets
contractions = pd.read_json(os.path.join(data_dir, 'english-contractions/contractions.json'), typ='series')
contractions = contractions.to_dict()

c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

# Define a function to clean and preprocess tweets
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        tweet = tweet.lower()
        tweet = BAD_SYMBOLS_RE.sub(' ', tweet)
        tweet = p.clean(tweet)
        
        # Expand contractions
        tweet = expandContractions(tweet)

        # Remove punctuation
        tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(tweet) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        tweet = ' '.join(filtered_sentence)
        
        cleaned_tweets.append(tweet)
        
    return cleaned_tweets

# Clean and preprocess the tweets
X = clean_tweets([tweet for tweet in df['text']])

# Perform word analysis using WordCloud for depressive tweets
depressive_tweets = [clean_tweets([t for t in df2['text']])]
depressive_words = ' '.join(list(map(str, depressive_tweets)))
depressive_wc = WordCloud(width=512, height=512, collocations=False, colormap="Blues").generate(depressive_words)

# Display the WordCloud
plt.figure(figsize=(10, 8), facecolor='k')
plt.imshow(depressive_wc)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# Tokenization using the Tokenizer class
MAX_NUM_WORDS = 10000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X)
word_vector = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index
vocab_size = len(word_index)

# Define the maximum sequence length and pad the input tensor
MAX_SEQ_LENGTH = 140
input_tensor = pad_sequences(word_vector, maxlen=MAX_SEQ_LENGTH)

# Baseline model using TF-IDF and Support Vector Machine (SVM)
corpus = df['text'].values.astype('U')
tfidf = TfidfVectorizer(max_features=MAX_NUM_WORDS) 
tdidf_tensor = tfidf.fit_transform(corpus)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(tdidf_tensor, df['label'].values, test_size=0.3)

# Train the baseline SVM model
baseline_model = SVC()
baseline_model.fit(x_train, y_train)
predictions = baseline_model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Baseline SVM Model Accuracy:", accuracy)
print(classification_report(y_test, predictions, digits=5))

# LTSM model with word embedding using Google News Word2Vec
EMBEDDING_FILE = os.path.join(data_dir, 'googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin.gz')
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
EMBEDDING_DIM = 300
embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))

# Create word embedding matrix from Word2Vec
for (word, idx) in word_index.items():
    if word in word2vec.vocab and idx < MAX_NUM_WORDS:
        embedding_matrix[idx] = word2vec.word_vec(word)

# Define the LTSM model architecture
inp = Input(shape=(MAX_SEQ_LENGTH,))
x = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(1, activation="sigmoid")(x)

# Compile and train the LTSM model
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x_train, x_test, y_train, y_test = train_test_split(input_tensor, df['label'].values, test_size=0.3)
model.fit(x_train, y_train, batch_size=16, epochs=10)
preds = model.predict(x_test)
preds = np.round(preds.flatten())
print("LTSM Model Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, digits=5))

# Testing with other models (Naive Baye's, Linear Support Vector, and Logistic Regression)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(X, df.label, test_size=0.3, random_state=42)

# Naive Baye's model
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
print('Naive Baye\'s Model Accuracy:', accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=5))

# Linear Support Vector model
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
print('Linear Support Vector Model Accuracy:', accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=5))

# Logistic Regression model
logreg = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                  ])
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Logistic Regression Model Accuracy:', accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=5))
