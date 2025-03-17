import os
import pandas as pd
import numpy as np

if __name__ == '__main__':

    ########## Load data ##################
    # Facebook 2022 paths
    path_input_f22 = "fb_2022_adid_var1.csv.gz"
    text_f22 = "fb_2022_adid_text.csv.gz"
    f22_ent = "wmpentity_2022_012125_mergedFECids.dta"

    # Google 2022 paths
    path_input_g22 = "g2022_adid_var1.csv.gz"
    text_g22 = "g2022_adid_text.csv.gz"
    g22_ent = "2022_google_entities_20240303_woldidstomerge.csv"

    # Output data path
    path_output_data = "data/party_train_prepared.csv.gz"


    ################ Training data from Meta 2022 ##############
    f22 = pd.read_csv(path_input_f22, usecols=['ad_id', 'pd_id'])
    f22text = pd.read_csv(text_f22, usecols=['ad_id', 'page_name', 'disclaimer', 'ad_creative_body', 
                                             'aws_ocr_text_img', 'aws_ocr_text_vid', 'google_asr_text', 
                                             'ad_creative_link_title', 'ad_creative_link_caption', 'ad_creative_link_description',
                      ])
    f22ent = pd.read_stata(f22_ent)
    f22ent = f22ent[['pd_id', 'party_all']]

    f22 = f22.merge(f22ent, on='pd_id', how='left')

    # Drop uncoded sponsors
    f22 = f22[f22.party_all != '']
    f22 = f22[f22.party_all.notna()]


    text_cols = ['page_name', 'disclaimer', 'ad_creative_body',
                     'google_asr_text', 'aws_ocr_text_img', 'aws_ocr_text_vid',
               'ad_creative_link_title', 'ad_creative_link_caption', 'ad_creative_link_description',]
    f22text[text_cols] = f22text[text_cols].fillna('')
    f22text['text'] = f22text[text_cols].agg(lambda x: ' '.join(x), axis=1)
    f22text['text'] = f22text['text'].str.strip()

    f22 = f22.merge(f22text, on='ad_id', how='inner')

    # drop duplicates to reduce data leakage
    f22 = f22[['ad_id', 'pd_id', 'text', 'party_all']].drop_duplicates('text')

    ################# Training data from Google 2022 ######################
    g22 = pd.read_csv(path_input_g22, usecols=["ad_id", "advertiser_id"])
    g22text = pd.read_csv(text_g22, usecols=["ad_id", "advertiser_name", "ad_title", "ad_text", "aws_ocr_img_text", "aws_ocr_video_text", 
                                             "google_asr_text", "description"])
    g22ent = pd.read_csv(g22_ent, usecols=["advertiser_id_tomerge", "party_all"])

    g22 = g22.merge(g22ent, left_on='advertiser_id', right_on='advertiser_id_tomerge', how='left')
    g22.drop('advertiser_id_tomerge', axis=1, inplace=True)

    # Drop NAs -- uncoded advertisers
    g22 = g22[g22.party_all.notna()]

    # Re-label third parties or independents as OTHER
    g22.loc[~g22.party_all.isin(['REP', 'DEM']), 'party_all'] = 'OTHER'


    # Concatenate textual fields and pre-process text strings
    text_cols = ["advertiser_name", "ad_title", "ad_text", "aws_ocr_img_text", "aws_ocr_video_text", 
                                             "google_asr_text", "description"]
    g22text[text_cols] = g22text[text_cols].fillna('')
    g22text['text'] = g22text[text_cols].agg(lambda x: ' '.join(x), axis=1)
    g22text['text'] = g22text['text'].str.replace('\n', ' ')
    g22text['text'] = g22text['text'].str.strip()

    g22 = g22.merge(g22text, on='ad_id', how='inner')
    g22 = g22[['ad_id', 'advertiser_id', 'text', 'party_all']].drop_duplicates('text')

    f22.rename(columns={'pd_id': 'sponsor_id'}, inplace=True)
    g22.rename(columns={'advertiser_id': 'sponsor_id'}, inplace=True)

    ####### Combine Meta and Google 2022 to form data for training and testing ########
    train = pd.concat([f22, g22])

    train.to_csv(path_output_data, index=False,
         compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})


    



