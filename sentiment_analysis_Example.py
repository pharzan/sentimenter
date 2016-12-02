
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# review.csv contains two columns
# first column is the review content (quoted)
# second column is the assigned sentiment (positive or negative)

# preprocess creates the term frequency matrix for the review data set
d =                 ['Farzan is a great guy.',
                    'Farzan is clever and smart also intelligent.',
                    'Farzan is a developer.',
                    'Farzan is good at software development',
                    'Farzan is a winner.',
                    'Farzan is a good guy and knows alot about computers and software.',
                    'Everyone wants to be good smart and clever like Farzan.',
                    'Payam is a normal guy',
                    'Payam drinks alot.',
                    'Payam is a loser and he is bad.',
                    'Nobody wants to be like payam.']

target=[1,1,1,1,1,1,1,0,0,0,0]
df['Positive']=pd.read_csv('positive.txt',sep='/n')

# from sklearn.datasets import fetch_20newsgroups
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(d)

# #From occurrences to frequencies
# from sklearn.feature_extraction.text import TfidfTransformer
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)
# X_train_tf.shape

# #Training a classifier
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_train_tfidf.shape

# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB().fit(X_train_tfidf, target)

# docs_new = ['he is bad', 'He is a loser and not a winner']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicted = clf.predict(X_new_tfidf)
# print(predicted)