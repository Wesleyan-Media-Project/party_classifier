# Party classifier, 
# Split train/test by pd_id
# All fields

import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from joblib import dump, load
import random

# Input
path_train_test = "data/2020_fb_and_google_with_page_id_based_training_data.csv.gz"
path_train_test_text_f = "../fb_2020/fb_2020_140m_adid_text_clean.csv.gz"
path_train_test_text_g = "../google_2020/google_2020_adid_text_clean.csv.gz"
# Output
path_model = 'models/party_clf_facebook_and_google_2020.joblib'
path_model_smooth = 'models/party_clf_facebook_and_google_2020_smooth.joblib'

# Load train/test metadata
d = pd.read_csv(path_train_test, encoding='UTF-8', keep_default_na = False)
# Load train/test text
d_text_f = pd.read_csv(path_train_test_text_f, encoding='UTF-8', keep_default_na = False)
d_text_g = pd.read_csv(path_train_test_text_g, encoding='UTF-8', keep_default_na = False)

# Combine all text fields and remove duplicate texts
# Facebook
cols_f = ['ad_creative_body', 'ad_creative_link_description', 'ad_creative_link_title', 'aws_ocr_text', 'google_asr_text']
d_text_f['combined'] = d_text_f[cols_f].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
d_text_f = d_text_f.drop_duplicates(subset=['combined'])
d_text_f = d_text_f[['ad_id', 'combined']]
# Google
cols_g = ['scraped_ad_title', 'scraped_ad_content', 'aws_ocr_text', 'google_asr_text']
d_text_g['combined'] = d_text_g[cols_g].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
d_text_g = d_text_g.drop_duplicates(subset=['combined'])
d_text_g = d_text_g[['ad_id', 'combined']]

# Combine Google and FB
d_text = pd.concat([d_text_f, d_text_g], axis = 0)

# Merge
d = d.merge(d_text, on = "ad_id")

d = d[d['party_all_usable'].isin(['DEM', 'REP'])]

# Split by pd-id (previously assigned)
train = d[d['split'] == 'train']
test = d[d['split'] == 'test']

# Equalize number of D/R
train_D = train[train['party_all_usable'] == 'DEM']
train_R = train[train['party_all_usable'] == 'REP']
train_D = shuffle(train_D)
train_D = train_D[:train_R.shape[0]]
train = pd.concat([train_D, train_R], 0)

test_D = test[test['party_all_usable'] == 'DEM']
test_R = test[test['party_all_usable'] == 'REP']
test_D = shuffle(test_D)
test_D = test_D[:test_R.shape[0]]
test = pd.concat([test_D, test_R], 0)

# Define model
clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('logreg', LogisticRegression(random_state = 123),)
])

clf_smooth = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('logreg', LogisticRegression(penalty = 'l2', C = 0.02, random_state = 123),)
])

# Train
clf.fit(train['combined'], train['party_all_usable'])
clf_smooth.fit(train['combined'], train['party_all_usable'])

# Assess performance performance
predicted = clf.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted))

#               precision    recall  f1-score   support
# 
#          DEM       0.82      0.86      0.84      9193
#          REP       0.85      0.81      0.83      9193
# 
#     accuracy                           0.84     18386
#    macro avg       0.84      0.84      0.84     18386
# weighted avg       0.84      0.84      0.84     18386

predicted_sm = clf_smooth.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted_sm))

#               precision    recall  f1-score   support
# 
#          DEM       0.80      0.80      0.80      9193
#          REP       0.80      0.80      0.80      9193
# 
#     accuracy                           0.80     18386
#    macro avg       0.80      0.80      0.80     18386
# weighted avg       0.80      0.80      0.80     18386

# Save model to disk
dump(clf, path_model, compress = 3)
dump(clf_smooth, path_model_smooth, compress = 3)


# Predicted probability distribution on the test set
import seaborn as sns
import matplotlib.pyplot as plt

pp = clf.predict_proba(test['combined'])
plot = sns.kdeplot(data=pp[:,0], c = 'blue')
plot = sns.kdeplot(data=pp[:,1], c = 'red')
fig = plot.get_figure()
fig.savefig("analysis/probs_on_balanced_test_set.png")
plt.clf()

pp = clf_smooth.predict_proba(test['combined'])
plot = sns.kdeplot(data=pp[:,0], c = 'blue')
plot = sns.kdeplot(data=pp[:,1], c = 'red')
fig = plot.get_figure()
fig.savefig("analysis/probs_on_balanced_test_set_smooth.png")
plt.clf()
