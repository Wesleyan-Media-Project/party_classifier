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
path_train_test = "../../data/2024/2024_fb_gg_tv_training_data.csv.gz"
path_train_test_balanced = "../../data/2024/2024_fb_gg_tv_training_data_balanced.csv.gz"
path_train_test_balanced_platform = "../../data/2024/2024_fb_gg_tv_training_data_balanced_platform.csv.gz"
path_train_test_no_tv = "../../data/2024/2024_fb_gg_tv_training_data_no_tv.csv.gz"

# fb_2020_140m_adid_text_clean.csv.gz is an output from repo fb_2020
path_entity_fb = "../../../entity_linking_2024/data/entity_linking_results_meta2024.csv.gz"
path_entity_gg = "../../../entity_linking_2024/data/entity_linking_results_google24.csv.gz"
path_entity_tv = "../../../entity_linking_2024/data/entity_linking_results_tv2024.csv.gz"

# Output
path_model = '../../models/party_clf_fb_gg_tv_2024.joblib'
path_model_balanced = '../../models/party_clf_fb_gg_tv_2024_balanced.joblib'
path_model_balanced_platform = '../../models/party_clf_fb_gg_tv_2024_balanced_platform.joblib'
path_model_no_tv = '../../models/party_clf_fb_gg_tv_2024_no_tv.joblib'

path_model_smooth = '../../models/party_clf_fb_gg_tv_2024_smooth.joblib'
path_model_smooth_balanced = '../../models/party_clf_fb_gg_tv_2024_smooth_balanced.joblib'
path_model_smooth_balanced_platform = '../../models/party_clf_fb_gg_tv_2024_smooth_balanced_platform.joblib'
path_model_smooth_no_tv = '../../models/party_clf_fb_gg_tv_2024_smooth_no_tv.joblib'

# Load train/test metadata
d = pd.read_csv(path_train_test, encoding='UTF-8', keep_default_na = False)
d_b = pd.read_csv(path_train_test_balanced, encoding='UTF-8', keep_default_na = False)
d_b_p = pd.read_csv(path_train_test_balanced_platform, encoding='UTF-8', keep_default_na = False)
d_no_tv = pd.read_csv(path_train_test_no_tv, encoding='UTF-8', keep_default_na = False)

d.party_all_usable.value_counts()
d_b.party_all_usable.value_counts()
d_b_p.party_all_usable.value_counts()

# Filter to relevant fields and group spans per ad+field
def flatten_lists(series):
    return [x for sublist in series for x in sublist]

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

# Load the text data and remove the detected entities from them
# Read in and process FB
df_el_f = pd.read_csv(path_entity_fb)

# Convert list-like columns back to actual lists
# text_start/text_end are already lists after ast.literal_eval
cols_to_convert = ['text_start', 'text_end']
for col in cols_to_convert:
    df_el_f[col] = df_el_f[col].apply(ast.literal_eval)

cols_f = ['ad_creative_body', 'ad_creative_link_description', 'ad_creative_link_title',
          'ad_creative_link_caption', 'ocr_text', 'asr_text']

df_grouped = (df_el_f[df_el_f['field'].isin(cols_f)]
              .groupby(['ad_id', 'field'])
              .agg(text=('text', 'first'),
                   starts=('text_start', flatten_lists),
                   ends=('text_end', flatten_lists))
              .reset_index())
# Apply removal
df_grouped['text_clean'] = df_grouped.apply(
    lambda row: remove_substrings(row['text'], row['starts'], row['ends']), axis=1
)

# Pivot to wide format and build combined
df_wide = (df_grouped[['ad_id', 'field', 'text_clean']]
           .pivot(index='ad_id', columns='field', values='text_clean')
           .reset_index())
df_wide.columns.name = None

df_wide['combined'] = df_wide[[c for c in cols_f if c in df_wide.columns]].apply(
    lambda row: ' '.join(row.dropna().astype(str)), axis=1
)
df_wide['combined'] = df_wide['combined'].str.replace(r'\{\{[^}]+\}\}', '', regex=True)
df_wide['combined'] = df_wide['combined'].str.replace('"', '', regex=False)

df_el_f = df_wide[['ad_id', 'combined']]

# Read in and process Google
df_el_g = pd.read_csv(path_entity_gg)

cols_to_convert = ['text_start', 'text_end']
for col in cols_to_convert:
    df_el_g[col] = df_el_g[col].apply(ast.literal_eval)

cols_g = ['ad_text', 'ocr_text', 'asr_text']

df_grouped_g = (df_el_g[df_el_g['field'].isin(cols_g)]
                .groupby(['ad_id', 'field'])
                .agg(text=('text', 'first'),
                     starts=('text_start', flatten_lists),
                     ends=('text_end', flatten_lists))
                .reset_index())

df_grouped_g['text_clean'] = df_grouped_g.apply(
    lambda row: remove_substrings(row['text'], row['starts'], row['ends']), axis=1
)

df_wide_g = (df_grouped_g[['ad_id', 'field', 'text_clean']]
             .pivot(index='ad_id', columns='field', values='text_clean')
             .reset_index())
df_wide_g.columns.name = None

df_wide_g['combined'] = df_wide_g[[c for c in cols_g if c in df_wide_g.columns]].apply(
    lambda row: ' '.join(row.dropna().astype(str)), axis=1
)

df_el_g = df_wide_g[['ad_id', 'combined']]

# Read in and process TV
df_el_t = pd.read_csv(path_entity_tv)

cols_to_convert = ['text_start', 'text_end']
for col in cols_to_convert:
    df_el_t[col] = df_el_t[col].apply(ast.literal_eval)

cols_t = ['ocr_text', 'asr_text']

df_grouped_t = (df_el_t[df_el_t['field'].isin(cols_t)]
                .groupby(['alt', 'field'])
                .agg(text=('text', 'first'),
                     starts=('text_start', flatten_lists),
                     ends=('text_end', flatten_lists))
                .reset_index())

df_grouped_t['text_clean'] = df_grouped_t.apply(
    lambda row: remove_substrings(row['text'], row['starts'], row['ends']), axis=1
)

df_wide_t = (df_grouped_t[['alt', 'field', 'text_clean']]
             .pivot(index='alt', columns='field', values='text_clean')
             .reset_index())
df_wide_t.columns.name = None

df_wide_t['combined'] = df_wide_t[[c for c in cols_t if c in df_wide_t.columns]].apply(
    lambda row: ' '.join(row.dropna().astype(str)), axis=1
)

df_wide_t['ad_id'] = df_wide_t['alt'].copy()

df_el_t = df_wide_t[['ad_id', 'combined']]

# Combine Google and FB
d_text = pd.concat([df_el_f, df_el_g, df_el_t], axis = 0)

# Merge
d = d.merge(d_text, on = "ad_id")
d['platform'].value_counts()
d['party_all'].value_counts()

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

#          DEM       1.00      0.91      0.95      5343
#          REP       0.81      1.00      0.89      1930

#     accuracy                           0.94      7273
#    macro avg       0.90      0.96      0.92      7273
# weighted avg       0.95      0.94      0.94      7273

predicted_sm = clf_smooth.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted_sm))

#               precision    recall  f1-score   support

#          DEM       1.00      0.91      0.95      5343
#          REP       0.80      1.00      0.89      1930

#     accuracy                           0.93      7273
#    macro avg       0.90      0.95      0.92      7273
# weighted avg       0.95      0.93      0.94      7273

# Save model to disk
dump(clf, path_model, compress = 3)
dump(clf_smooth, path_model_smooth, compress = 3)


# Merge with balanced sample
d = d_b.merge(d_text, on = "ad_id")
d['platform'].value_counts()
d['party_all'].value_counts()

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

#          DEM       1.00      0.96      0.98      2368
#          REP       0.96      1.00      0.98      1930

#     accuracy                           0.98      4298
#    macro avg       0.98      0.98      0.98      4298
# weighted avg       0.98      0.98      0.98      4298


predicted_sm = clf_smooth.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted_sm))

#               precision    recall  f1-score   support

#          DEM       1.00      0.96      0.98      2368
#          REP       0.95      1.00      0.97      1930

#     accuracy                           0.98      4298
#    macro avg       0.97      0.98      0.97      4298
# weighted avg       0.98      0.98      0.98      4298

# Save model to disk
dump(clf, path_model_balanced, compress = 3)
dump(clf_smooth, path_model_smooth_balanced, compress = 3)


# Merge with balanced_platform sample
d = d_b_p.merge(d_text, on = "ad_id")
d['platform'].value_counts()
d['party_all'].value_counts()

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

#          DEM       1.00      0.95      0.98      4078
#          REP       0.91      1.00      0.95      1930

#     accuracy                           0.97      6008
#    macro avg       0.96      0.98      0.96      6008
# weighted avg       0.97      0.97      0.97      6008

predicted_sm = clf_smooth.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted_sm))

#               precision    recall  f1-score   support

#          DEM       1.00      0.95      0.98      4078
#          REP       0.91      1.00      0.95      1930

#     accuracy                           0.97      6008
#    macro avg       0.96      0.98      0.96      6008
# weighted avg       0.97      0.97      0.97      6008

# Save model to disk
dump(clf, path_model_balanced_platform, compress = 3)
dump(clf_smooth, path_model_smooth_balanced_platform, compress = 3)

# No TV data for training
# Merge
d = d_no_tv.merge(d_text, on = "ad_id")
d['platform'].value_counts()
d['party_all'].value_counts()

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

#          DEM       1.00      0.93      0.96      5343
#          REP       0.83      1.00      0.91      1930

#     accuracy                           0.95      7273
#    macro avg       0.91      0.96      0.93      7273
# weighted avg       0.95      0.95      0.95      7273

predicted_sm = clf_smooth.predict(test['combined'])
print(metrics.classification_report(test['party_all_usable'], predicted_sm))
#               precision    recall  f1-score   support

#          DEM       1.00      0.92      0.96      5343
#          REP       0.81      1.00      0.90      1930

#     accuracy                           0.94      7273
#    macro avg       0.91      0.96      0.93      7273
# weighted avg       0.95      0.94      0.94      7273

# Save model to disk
dump(clf, path_model_no_tv, compress = 3)
dump(clf_smooth, path_model_smooth_no_tv, compress = 3)




# Predicted probability distribution on the test set
import seaborn as sns
import matplotlib.pyplot as plt

pp = clf.predict_proba(test['combined'])
plot = sns.kdeplot(data=pp[:,0], c = 'blue')
plot = sns.kdeplot(data=pp[:,1], c = 'red')
fig = plot.get_figure()
fig.savefig("../../analysis/probs_on_balanced_test_set_2024.png")
plt.clf()

pp = clf_smooth.predict_proba(test['combined'])
plot = sns.kdeplot(data=pp[:,0], c = 'blue')
plot = sns.kdeplot(data=pp[:,1], c = 'red')
fig = plot.get_figure()
fig.savefig("../../analysis/probs_on_balanced_test_set_smooth_2024.png")
plt.clf()

