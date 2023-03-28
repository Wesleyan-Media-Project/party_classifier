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

# Input
path_inference_data = "../google_2020/google_2020_adid_text_clean.csv.gz"
path_inference_data_vars = "../google_2020/google_2020_adid_var1.csv.gz"
path_model = "models/party_clf.joblib"

# Output
path_predictions = "data/google/party_predictions_google_2020.csv.gz"

# Inference dataset
df = pd.read_csv(path_inference_data, encoding='UTF-8', keep_default_na = False, dtype = 'str')
# All fields
cols = ['advertiser_name', 'scraped_ad_title', 'scraped_ad_url', 'scraped_ad_content', 'aws_ocr_text', 'google_asr_text']
# Combine and clean up
df['combined'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df['combined'] = df['combined'].str.strip() # Remove whitesaces at end of string
df['combined'] = df['combined'].str.replace(' +', ' ', regex = True) # Remove double (and triple etc.) whitespaces inside
df = df[['ad_id', 'combined']]
# Deduplicate by text to save time during inference
df = df.groupby(['combined'])['ad_id'].apply(list)
df = df.to_frame().reset_index()
# Remove empty ads
df = df[df['combined'] != ""]

# Load model
log_clf = load(path_model)

# Predicted probabilities
pp = log_clf.predict_proba(df['combined'])
df['prob_dem'] = pp[:,0]
df['prob_other'] = pp[:,1]
df['prob_rep'] = pp[:,2]
df['predicted_party_all'] = log_clf.classes_[np.argmax(pp, axis = 1)]

df = df.explode('ad_id')

# Merge in advertiser_id
df_vars = pd.read_csv(path_inference_data_vars, encoding='UTF-8')
df_vars = df_vars[['ad_id', 'advertiser_id']]
df = df.merge(df_vars, on = 'ad_id')
# Create a variable where all party classifications of a pd-id
# are assigned by majority vote
dft = df[['advertiser_id', 'predicted_party_all']]
maj_vote = dft.groupby(['advertiser_id'])['predicted_party_all'].agg(pd.Series.mode)
maj_vote = pd.DataFrame(maj_vote)
maj_vote = maj_vote.reset_index()
# In case of ties, make it OTHER
maj_vote['predicted_party_all'][[f is not str for f in maj_vote['predicted_party_all'].apply(type)]] = 'OTHER'
maj_vote.columns = ['advertiser_id', 'predicted_party_all_majvote']
df = df.merge(maj_vote, how = 'left', on = 'advertiser_id')

# Keep only the relevant variables
df = df[['ad_id', 'prob_dem', 'prob_other', 'prob_rep', 'predicted_party_all', 'predicted_party_all_majvote']]

# Save
df.to_csv(path_predictions, index = False)
