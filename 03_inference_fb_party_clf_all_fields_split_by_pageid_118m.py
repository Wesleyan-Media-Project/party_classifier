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

# 1.18m dataset
df118 = pd.read_csv("data/facebook/118m_with_page_id_based_training_data.csv", encoding='UTF-8', keep_default_na = False)
df118['combined'] = df118['disclaimer'] + df118['page_name'] + df118['ad_creative_body'] + df118['ad_creative_link_caption'] + df118['ad_creative_link_description'] + df118['ad_creative_link_title'] + df118['ocr'] + df118['asr']

# Load model
log_clf = load('models/party_clf_all_fields_split_by_party_uniform_page_id.joblib')

# Predicted probabilities
pp = log_clf.predict_proba(df118['combined'])
df118['prob_dem'] = pp[:,0]
df118['prob_other'] = pp[:,1]
df118['prob_rep'] = pp[:,2]
df118['predicted_party_all'] = log_clf.classes_[np.argmax(pp, axis = 1)]

# Keep only the relevant variables
df118 = df118[['ad_id', 'prob_dem', 'prob_other', 'prob_rep', 'predicted_party_all']]

# Save
df118.to_csv("data/facebook/party_clf_Facebook_118m_results_all_fields_split_by_party_uniform_pageid.csv", index = False)
