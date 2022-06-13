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
df118 = pd.read_csv("data/facebook/118m_with_page_id_based_training_data.csv", encoding='UTF-8', keep_default_na = False, dtype = 'str')
# temporary fix for missing ids ---
missing_pdids = 'pd-' + df118['page_id'][df118['pd_id'] == ''] + "-1"
df118['pd_id'][df118['pd_id'] == ''] = missing_pdids
# ---
df118['combined'] = df118['disclaimer'] + df118['page_name'] + df118['ad_creative_body'] + df118['ad_creative_link_caption'] + df118['ad_creative_link_description'] + df118['ad_creative_link_title'] + df118['ocr'] + df118['asr']

# Load model
log_clf = load('models/party_clf_all_fields_split_by_party_uniform_page_id.joblib')

# Predicted probabilities
pp = log_clf.predict_proba(df118['combined'])
df118['prob_dem'] = pp[:,0]
df118['prob_other'] = pp[:,1]
df118['prob_rep'] = pp[:,2]
df118['predicted_party_all'] = log_clf.classes_[np.argmax(pp, axis = 1)]

# Create a variable where all party classifications of a pd-id
# are assigned by majority vote
dft = df118[['pd_id', 'predicted_party_all']]
maj_vote = dft.groupby(['pd_id'])['predicted_party_all'].agg(pd.Series.mode)
maj_vote = pd.DataFrame(maj_vote)
maj_vote = maj_vote.reset_index()
# In case of ties, make it OTHER
maj_vote['predicted_party_all'][[f is not str for f in maj_vote['predicted_party_all'].apply(type)]] = 'OTHER'
maj_vote.columns = ['pd_id', 'predicted_party_all_majvote']
df118 = df118.merge(maj_vote, how = 'left', on = 'pd_id')

# Keep only the relevant variables
df118 = df118[['ad_id', 'prob_dem', 'prob_other', 'prob_rep', 'predicted_party_all', 'predicted_party_all_majvote']]

# Save
df118.to_csv("data/facebook/party_all_fb_118m.csv", index = False)
