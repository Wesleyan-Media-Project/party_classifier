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
path_inference_data = "fb_2022_adid_text.csv.gz"
path_inference_data_vars = "fb_2022_adid_var1.csv.gz"
# Load the best performing model - smoothed logistic regression
# path_model = "models/party_clf_facebook_and_google_2022.joblib"
path_model_smooth = "models/party_clf_facebook_and_google_2022_smooth.joblib"

# Output
path_predictions = "data/party_predictions_fb_2022.csv.gz"

# Inference dataset
df = pd.read_csv(path_inference_data, encoding='UTF-8', keep_default_na = False, dtype = 'str')

# All fields
cols = ['page_name', 'disclaimer', 'ad_creative_body', 'aws_ocr_text_vid', 'aws_ocr_text_img', 'google_asr_text', \
		'ad_creative_link_title','ad_creative_link_caption', 'ad_creative_link_description',]
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

# Load model
# Best performing model (smoothed)
clf = load(path_model_smooth)
# Predicted probabilities
pp = clf.predict_proba(df['combined'])
df['prob_dem'] = pp[:, 0]
df['prob_other'] = pp[:, 1]
df['prob_rep'] = pp[:, 2]
df['predicted_party_all'] = clf.classes_[np.argmax(pp, axis = 1)]

df = df.explode('ad_id')

# Merge in sponsor ID (pd_id)
fb_vars = pd.read_csv(path_inference_data_vars,usecols=['ad_id', 'pd_id'])
df = df.merge(fb_vars, on='ad_id')
# Get the majority voted party label within sponsor
maj_vote = df[['pd_id', 'predicted_party_all']].groupby(['pd_id'])['predicted_party_all'].agg(pd.Series.mode)
maj_vote = pd.DataFrame(maj_vote).reset_index()
maj_vote.columns = ['pd_id', 'predicted_party_all_majvote']
df = df.merge(maj_vote, how = 'left', on = 'pd_id')


# Keep only the relevant variables
df = df[['ad_id', 'prob_dem', 'prob_rep', 'prob_other', 'predicted_party_all', 'predicted_party_all_majvote']]

# Save
df.to_csv(path_predictions, index = False)
