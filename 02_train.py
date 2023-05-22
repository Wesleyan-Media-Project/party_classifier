# Party classifier, 
# Split train/test by pd_id
# All fields

import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
from joblib import dump, load
import random

# Input
path_train_test = "data/facebook/140m_with_page_id_based_training_data.csv.gz"
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
cols = ['ad_creative_body', 'ad_creative_link_caption', 'ad_creative_link_description', 'ad_creative_link_title', 'aws_ocr_text', 'google_asr_text']
d['combined'] = d[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Remove duplicate texts
d = d.drop_duplicates(subset=['combined'])

# Split by pd-id (previously assigned)
train = d[d['split'] == 'train']
test = d[d['split'] == 'test']

# Define model
clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('logreg', LogisticRegression(penalty = 'l2', C = 0.02, random_state = 123),)
])

# Train
clf.fit(train['combined'], train['party_all_usable'])

# Assess performance performance
predicted = clf.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted))

#               precision    recall  f1-score   support
# 
#          DEM       0.78      0.95      0.86     15366
#        OTHER       0.00      0.00      0.00       698
#          REP       0.87      0.62      0.73      9270
# 
#     accuracy                           0.80     25334
#    macro avg       0.55      0.52      0.53     25334
# weighted avg       0.79      0.80      0.79     25334

# Save model to disk
dump(clf, path_model, compress = 3)

#----
# import seaborn as sns
# 
# pp = log_clf.predict_proba(test['combined'])
# 
# plot = sns.kdeplot(data=pp[:,0])
# fig = plot.get_figure()
# fig.savefig("out2.png") 
# 
# import scipy
# scipy.stats.kurtosis(pp[:,0])
