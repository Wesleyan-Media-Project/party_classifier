# Party classifier, 
# Split train/test by pd_id (FB) and advertiser_id (Google)
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
import ast

# Input
path_train_test = "data/2020_fb_and_google_with_page_id_based_training_data.csv.gz"
# fb_2020_140m_adid_text_clean.csv.gz is an output from repo fb_2020
path_train_test_text_f = "../fb_2020/fb_2020_140m_adid_text_clean.csv.gz"
path_train_test_text_g = "../google_2020/google_2020_adid_text_clean.csv.gz"
# Output
path_model = 'models/party_clf_facebook_and_google_2020.joblib'
path_model_smooth = 'models/party_clf_facebook_and_google_2020_smooth.joblib'

# Load train/test metadata
d = pd.read_csv(path_train_test, encoding='UTF-8', keep_default_na = False)

# Load the text data and remove the detected entities from them
# Read in and process FB
df_el_f = pd.read_csv("../entity_linking/facebook/data/entity_linking_results_140m.csv.gz")

# Convert list-like columns back to actual lists
cols_to_convert = [col for col in df_el_f.columns if col.endswith('_start') or col.endswith('_end')]
for col in cols_to_convert:
  df_el_f[col] = df_el_f[col].apply(ast.literal_eval)

# Function to remove substrings based on start and end indices
def remove_substrings(s, starts, ends):
    if pd.isna(s):
          return s
    # Combine starts and ends into a list of tuples and sort them by start index descending
    intervals = sorted(zip(starts, ends), key=lambda x: x[0], reverse=True)

    # Remove substrings from end to start
    for start, end in intervals:
        if start < end:  # Ensure valid indices
            s = s[:start] + s[end:]

    return s

# Apply the removal of substrings
cols_f = ['ad_creative_body', 'ad_creative_link_description', 'ad_creative_link_title', 'ocr', 'asr']
for c in cols_f:
  df_el_f[c] = df_el_f.apply(lambda row: remove_substrings(row[c], row[c + '_start'], row[c + '_end']), axis=1)


# Read in and process Google
df_el_g = pd.read_csv("../entity_linking/google/data/entity_linking_results_google_2020.csv.gz")

# Convert list-like columns back to actual lists
cols_to_convert = [col for col in df_el_g.columns if col.endswith('_start') or col.endswith('_end')]
for col in cols_to_convert:
  df_el_g[col] = df_el_g[col].apply(ast.literal_eval)

# Apply the removal of substrings
cols_g = ['scraped_ad_title', 'scraped_ad_content', 'ocr', 'asr']
for c in cols_g:
  df_el_g[c] = df_el_g.apply(lambda row: remove_substrings(row[c], row[c + '_start'], row[c + '_end']), axis=1)


# Combine all text fields and remove duplicate texts
# Facebook
df_el_f['combined'] = df_el_f[cols_f].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
df_el_f = df_el_f.drop_duplicates(subset=['combined'])
df_el_f = df_el_f[['ad_id', 'combined']]
# Google
df_el_g['combined'] = df_el_g[cols_g].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
df_el_g = df_el_g.drop_duplicates(subset=['combined'])
df_el_g = df_el_g[['ad_id', 'combined']]

# Combine Google and FB
d_text = pd.concat([df_el_f, df_el_g], axis = 0)

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
train = pd.concat([train_D, train_R])

# test_D = test[test['party_all_usable'] == 'DEM']
# test_R = test[test['party_all_usable'] == 'REP']
# test_D = shuffle(test_D)
# test_D = test_D[:test_R.shape[0]]
# test = pd.concat([test_D, test_R], 0)

# Define model
clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('logreg', LogisticRegression(random_state = 123),)
])

clf_smooth = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('logreg', LogisticRegression(penalty = 'l2', C = 0.4, random_state = 123, max_iter=200),)
])

# Train
clf.fit(train['combined'], train['party_all_usable'])
clf_smooth.fit(train['combined'], train['party_all_usable'])

# Assess performance performance
predicted = clf.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted))

#               precision    recall  f1-score   support
# 
#          DEM       0.84      0.86      0.85      1078
#          REP       0.86      0.85      0.85      1158
# 
#     accuracy                           0.85      2236
#    macro avg       0.85      0.85      0.85      2236
# weighted avg       0.85      0.85      0.85      2236

predicted_sm = clf_smooth.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted_sm))

#               precision    recall  f1-score   support
# 
#          DEM       0.83      0.84      0.84      1078
#          REP       0.85      0.84      0.85      1158
# 
#     accuracy                           0.84      2236
#    macro avg       0.84      0.84      0.84      2236
# weighted avg       0.84      0.84      0.84      2236

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
