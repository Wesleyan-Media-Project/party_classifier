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
dfggl = pd.read_csv("data/google/all_fields_concatenated.csv")
dfggl = dfggl.replace(np.nan, '', regex=True)

# Load model
log_clf = load('models/party_clf_all_fields_split_by_party_uniform_page_id.joblib')

# Predicted probabilities
pp = log_clf.predict_proba(dfggl['text'])
dfggl['prob_dem'] = pp[:,0]
dfggl['prob_other'] = pp[:,1]
dfggl['prob_rep'] = pp[:,2]
dfggl['predicted_party_all'] = log_clf.classes_[np.argmax(pp, axis = 1)]

dfggl = dfggl.drop(['text'], axis = 1)

# Create a variable where all party classifications of a pd-id
# are assigned by majority vote
dft = dfggl[['advertiser_id', 'predicted_party_all']]
maj_vote = dft.groupby(['advertiser_id'])['predicted_party_all'].agg(pd.Series.mode)
maj_vote = pd.DataFrame(maj_vote)
maj_vote = maj_vote.reset_index()
# In case of ties, make it OTHER
maj_vote['predicted_party_all'][[f is not str for f in maj_vote['predicted_party_all'].apply(type)]] = 'OTHER'
maj_vote.columns = ['advertiser_id', 'predicted_party_all_majvote']
dfggl = dfggl.merge(maj_vote, how = 'left', on = 'advertiser_id')

# Keep only the relevant variables
dfggl = dfggl[['ad_id', 'prob_dem', 'prob_other', 'prob_rep', 'predicted_party_all', 'predicted_party_all_majvote']]

# Save
dfggl.to_csv("data/google/party_all_google.csv", index = False)
