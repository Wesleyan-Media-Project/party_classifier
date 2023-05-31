import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
from joblib import dump, load

# Input
path_inference_data = "../fb_2020/fb_2020_140m_adid_text_clean.csv.gz"
path_inference_data_vars = "../fb_2020/fb_2020_140m_adid_var1.csv.gz"
path_model = "models/party_clf_facebook_and_google_2020.joblib"
path_model_smooth = "models/party_clf_facebook_and_google_2020_smooth.joblib"

# Output
path_predictions = "data/facebook/party_predictions_fb_2020_140m_both_model.csv.gz"

# Inference dataset
df = pd.read_csv(path_inference_data, encoding='UTF-8', keep_default_na = False, dtype = 'str')
# All fields
cols = ['ad_creative_body', 'ad_creative_link_caption', 'ad_creative_link_description', 'ad_creative_link_title', 'aws_ocr_text', 'google_asr_text']
# Combine and clean up
df['combined'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df['combined'] = df['combined'].str.strip()
df['combined'] = df['combined'].str.replace(' +', ' ', regex = True) # Remove double (and triple etc.) whitespaces inside
df = df[['ad_id', 'combined']]
# Deduplicate by text to save time during inference
df = df.groupby(['combined'])['ad_id'].apply(list)
df = df.to_frame().reset_index()
# Remove empty ads
df = df[df['combined'] != ""]

# Regular model
# Load model
clf = load(path_model)
# Predicted probabilities
pp = clf.predict_proba(df['combined'])
df['prob_dem'] = pp[:,0]
df['prob_rep'] = pp[:,1]

# Smooth model
# Load model
clf_smooth = load(path_model_smooth)
# Predicted probabilities
pp = clf_smooth.predict_proba(df['combined'])
df['prob_dem_smooth'] = pp[:,0]
df['prob_rep_smooth'] = pp[:,1]

df = df.explode('ad_id')

# Keep only the relevant variables
df = df[['ad_id', 'prob_dem', 'prob_rep', 'prob_dem_smooth', 'prob_rep_smooth']]

# Save
df.to_csv(path_predictions, index = False)
