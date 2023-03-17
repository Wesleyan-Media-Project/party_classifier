# Party Classifier

## Description
This repository contains the code for a party classifier for the 1.18m Facebook dataset. The classifier outputs class labels (DEM/REP/OTHER), class labels aggregated to the pd_id-level (advertiser_id for Google) via majority vote (OTHER in case of a tie), as well as class probabilities. Training is done on the portion of the 1.18m dataset for which party_all is known, based on merging with the most recent WMP entities file (v051822). Only pages for which all of their pd_ids are associated with the same party_all are used. For training, the data is split by assigning 70% of the page_ids to training, and 30% of the page_ids to test. Ergo, all ads associated with a specific page_id can only be in either training or test.

The following fields are used in the classifier by concatenating them in the following order, separated by a single space: disclaimer, page_name, ad_creative_body, ad_creative_link_caption, ad_creative_link_description, ad_creative_link_title, ocr, asr. Prior to the train/test split, the concatenated ads are de-duplicated, so that only one version of every concatenated ad content can go into either train/test (we could potentially only de-duplicate within page_ids, but currently don't).

The classifier is logistic regression, with C=10 and solver='newton-cg' (hyperparameters determined by Jielu, likely out of date), with CalibratedClassifierCV wrapped around it for smoother class probabilities (i.e. so that not all probabilities are either >0.99 or <0.01).

## Performance
Performance on held-out test set:
```
              precision    recall  f1-score   support

         DEM       0.88      0.89      0.89     16394
       OTHER       0.46      0.22      0.30       558
         REP       0.79      0.80      0.79      8580

    accuracy                           0.85     25532
   macro avg       0.71      0.64      0.66     25532
weighted avg       0.84      0.85      0.84     25532
```

## Requirements
### Data
In addition to the files being tracked, the following files are required (not uploaded because too large)

`datasets/facebook/fb_2020_adid_06092022.csv` (`Delta Lab/Data/facebook_2020/fb_2020_adid_06092022.csv`) \
`datasets/facebook/fb_2020_adid_06092022.csv` (`Delta Lab/Data/entities_fb_2020/wmp_fb_entities_v051822.csv`) \
`datasets/facebook/118m_all_ads.csv` (tbd)

## Results
The 03_ files apply the trained model to the 1.18m dataset as well as the Google dataset. The results from that are uploaded to `Delta Lab/Data/facebook_2020_party_all/` and `Delta Lab/Data/google_2020_party_all/`


