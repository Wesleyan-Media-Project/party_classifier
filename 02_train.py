# Party classifier, 
# Split train/test by pd_id
# All fields

import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
from joblib import dump, load
import random

# Input
path_train_test = "data/facebook/118m_with_page_id_based_training_data.csv.gz"
path_train_test_text = "../fb_2020/fb_2020_140m_adid_text_clean.csv.gz"
# Output
path_model = 'models/party_clf.joblib'

# Load train/test metadata
d = pd.read_csv(path_train_test, encoding='UTF-8', keep_default_na = False)
# Load train/test text
d_text = pd.read_csv(path_train_test_text, encoding='UTF-8', keep_default_na = False)
# Merge
d = d.merge(d_text, on = "ad_id")

# All fields
cols = ['disclaimer', 'page_name', 'ad_creative_body', 'ad_creative_link_caption', 'ad_creative_link_description', 'ad_creative_link_title', 'aws_ocr_text', 'google_asr_text']
d['combined'] = d[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Remove duplicate texts
d = d.drop_duplicates(subset=['combined'])

# Split by pd-id (previously assigned)
train = d[d['split'] == 'train']
test = d[d['split'] == 'test']

# Define model
rf_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('cal', CalibratedClassifierCV(RandomForestClassifier(random_state = 123), cv=2, method="sigmoid"),)
])

# Train
rf_clf.fit(train['combined'], train['party_all_usable'])

# Assess performance performance
predicted = rf_clf.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted))

#               precision    recall  f1-score   support
# 
#          DEM       0.86      0.91      0.89     15366
#        OTHER       0.84      0.05      0.10       698
#          REP       0.82      0.81      0.82      9270
# 
#     accuracy                           0.85     25334
#    macro avg       0.84      0.59      0.60     25334
# weighted avg       0.85      0.85      0.84     25334

# Save model to disk
dump(rf_clf, path_model, compress = 3)
