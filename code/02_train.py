import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from joblib import dump, load
import random


if __name__ == '__main__':
    

    # Input
    path_train_test = "data/party_train_prepared.csv.gz"

    # Output
    path_model = "models/party_clf_facebook_and_google_2022.joblib"
    path_model_smooth = "models/party_clf_facebook_and_google_2022_smooth.joblib"

    # Read in the data
    df = pd.read_csv(path_train_test, encoding='UTF-8')

    # Split train-test at the sponsor level to avoid the same sponsor's ads end up in both train and test
    unique_spnosors = df[['sponsor_id', 'party_all']].drop_duplicates(['sponsor_id', 'party_all'])
    train_sponsors, test_sponsors = ms.train_test_split(unique_spnosors, test_size=0.1, stratify=unique_spnosors.party_all, random_state=1802)

    train = df[df.sponsor_id.isin(set(train_sponsors.sponsor_id))]
    test = df[df.sponsor_id.isin(set(test_sponsors.sponsor_id))]

    print("Training Data Shape: ", train.shape)
    print(train.value_counts('party_all'))
    print("Testing Data Shape: ", test.shape)
    print(test.value_counts('party_all'))

    # Define model
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('logreg', LogisticRegression(solver='lbfgs', penalty = 'l2', multi_class='multinomial',
                    C = 0.02, random_state = 1802),)
    ])


    # Smoothed model
    clf_smooth = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('logreg', LogisticRegression(penalty = 'l2', C = 0.4, multi_class='multinomial', random_state = 123, max_iter=500),)])


    # Train
    clf.fit(train['text'], train['party_all'])
    clf_smooth.fit(train['text'], train['party_all'])

    # Calculate precision, recall, F-score, support, and accuracy
    predicted = clf.predict(test['text'])
    predicted_sm = clf_smooth.predict(test['text'])
    
    print('Regular Logistic Regression:')
    print('Classes: ', clf.classes_)
    print(metrics.classification_report(test['party_all'], predicted))

    print('Smoothed Logistic Regression:')
    print('Classes: ', clf_smooth.classes_)
    print(metrics.classification_report(test['party_all'], predicted_sm))

    # Save models to disk
    dump(clf, path_model, compress = 3)
    dump(clf_smooth, path_model_smooth, compress = 3)

