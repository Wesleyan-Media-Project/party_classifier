# Party classifier, 
# Split train/test by pd_id
# All fields

import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
from joblib import dump, load
import random

# Load data, restrict to only 1.18m
d = pd.read_csv("../data/facebook/118m_with_page_id_based_training_data.csv", encoding='UTF-8', keep_default_na = False)

# All fields
cols = ['disclaimer', 'page_name', 'ad_creative_body', 'ad_creative_link_caption', 'ad_creative_link_description', 'ad_creative_link_title', 'ocr', 'asr']
d['combined'] = d[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

d2 = d[d['split'].isin(['train','test'])]
d2 = d2.drop_duplicates(subset=['combined'])

# Split by pd-id (previously assigned)
train = d2[d2['split'] == 'train']
test = d2[d2['split'] == 'test']

# Define model
log_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('cal', CalibratedClassifierCV(LogisticRegression(C=10, solver='newton-cg'), cv=2, method="sigmoid"),)
])

# Train
log_clf.fit(train['combined'], train['party_all_usable'])

# Assess performance performance
predicted = log_clf.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted))

# Save model to disk
dump(log_clf, 'models/party_clf_all_fields_split_by_party_uniform_page_id.joblib')
