# CREATIVE --- Ad-level Party Classifier

Welcome! This repository contains scripts that train and apply a machine learning model to classify political advertisements based on their content and determine which political party (Democratic, Republican, or Other) the ads belong to.

This repo is part of the [Cross-platform Election Advertising Transparency Initiative (CREATIVE)](https://www.creativewmp.com/). CREATIVE is an academic research project that has the goal of providing the public with analysis tools for more transparency of political ads across online platforms. In particular, CREATIVE provides cross-platform integration and standardization of political ads collected from Google and Facebook. CREATIVE is a joint project of the [Wesleyan Media Project (WMP)](https://mediaproject.wesleyan.edu/) and the [privacy-tech-lab](https://privacytechlab.org/) at [Wesleyan University](https://www.wesleyan.edu).

To analyze the different dimensions of political ad transparency we have developed an analysis pipeline. The scripts in this repo are part of the Data Classification step in our pipeline.

![A picture of the repo pipeline with this repo highlighted](Creative_Pipelines.png)

## Table of Contents

- [1. Overview](#1-overview)
- [2. Data](#2-data)
- [3. Setup](#3-setup)
  - [3.1 Training](#31-training)
  - [3.2 Model](#32-model)
  - [3.3 Performance](#33-performance)
- [4. Thank You](#4-thank-you)

## 1. Overview

This repo contains scripts for a multinomial ad-level party classifier that classifies ads into DEM/REP/OTHER. The difference to the [other party classifier](https://github.com/Wesleyan-Media-Project/party_classifier_pdid) is that for this classifier the training data consists of **individual** ads whose pd_id has `party_all` coded in the [WMP entity file](https://github.com/Wesleyan-Media-Project/datasets/tree/main/wmp_entity_files) which is a list of all the unique sponsors of ads on Google and Facebook. By contrast, the other party classifier concatenates all ads of a pd_id into one. In situations where you need clear and specific predictions about political party affiliations for ads, it is better to use the [other party classifier](https://github.com/Wesleyan-Media-Project/party_classifier_pdid). This is because the other party classifier operates under the assumption that all ads associated with a single pd_id will belong to the same party, leading to more consistent and potentially more accurate predictions about party affiliation when viewing the ads collectively rather than individually. The main purpose of this ad-level classifier is to get predictions for individual ads, which can then be used to express the degree to which an ad belongs to either party.

## 2. Data

Data processed and generated by the scripts in this repository are stored as compressed CSV files (csv.gz) in the `/data` folder. The outputs include class labels (DEM/REP/OTHER) and aggregated labels at the pd_id level (advertiser_id for Google) determined by a majority vote. In case of a tie in which the classifier can't decide the party, the label defaults to OTHER. In addition to the class labels, the classifier computes probabilities that indicate the likelihood of each ad belonging to the DEM, REP, or OTHER categories. However, to obtain more accurate class probabilities, we recommend you use the [other party classifier](https://github.com/Wesleyan-Media-Project/party_classifier_pdid).

## 3. Setup

To start setting up the repo and run the scripts, first clone this repo to your local directory:

```bash
git clone https://github.com/Wesleyan-Media-Project/party_classifier.git
```

Then, ensure you have the required dependencies installed.
The scripts are tested on R 4.2, 4.3, 4.4 and Python 3.9 and 3.10. The packages we used are described in requirements_r.txt and requirements_py.txt. You can install the required Python packages by running:

```bash
pip install -r requirements_py.txt
```

For R, you can install the required packages by running:

```bash
Rscript -e 'install.packages(readLines("requirements_r.txt"))'
```

The scripts are numbered in the order in which they should be run. For example, you should follow the order 01, 02, 03, etc according to the file names. Scripts that directly depend on one another are ordered sequentially. Scripts with the same number are alternatives, usually they are the same scripts on different data, or with minor variations. For example, `03_inference_google_2022.ipynb` and `03_inference_google_2022_both_model.ipynb` are applying the party classifiers trained on different datasets. Inference scripts on 2022 political advertising datasets contain "\_2022" in the filenames.

If you want to use the trained model we provide, you can also only run the inference script since the model files are already present in the `/models` folder.

### 3.1 Training

Note: If you do not want to train models from scratch, you can use the trained model we provide [here](https://github.com/Wesleyan-Media-Project/party_classifier/tree/main/models), and skip to 3.4.

To run this repo, you first need to train a classification model. We have two training scripts that use two different training data:

1. Training that is done using the portion of the Facebook 2020 dataset for which party_all is known, based on merging with the most recent WMP entities file (v090622) `wmp_fb_entities_v090622.csv`. You need the following files for this:

   - [fb_2020/fb_2020_140m_adid_text_clean.csv.gz](https://figshare.wesleyan.edu/account/articles/26093257)
   - [fb_2020/fb_2020_140m_adid_var1.csv.gz](https://figshare.wesleyan.edu/account/articles/26093254)
   - [datasets/wmp_entity_files/Facebook/2020/wmp_fb_entities_v090622.csv](https://github.com/Wesleyan-Media-Project/datasets/blob/main/wmp_entity_files/Facebook/2020/wmp_fb_entities_v090622.csv)

2. Training that is done using the portion of the Facebook AND Google 2020 dataset for which party_all is known, based on merging with the most recent WMP entities file (v090622) `wmp_fb_entities_v090622.csv`. You need the following files for this:
   - [fb_2020/fb_2020_140m_adid_text_clean.csv.gz](https://figshare.wesleyan.edu/account/articles/26093257)
   - [fb_2020/fb_2020_140m_adid_var1.csv.gz](https://figshare.wesleyan.edu/account/articles/26093254)
   - google_2020/google_2020_adid_var1.csv.gz (PROVIDE FIGSHARE LINK ONCE READY)
   - google_2020/google_2020_adid_text_clean.csv.gz (PROVIDE FIGSHARE LINK ONCE READY)
   - [datasets/wmp_entity_files/Google/2020/wmp_google_entities_v040521.dta](https://github.com/Wesleyan-Media-Project/datasets/blob/main/wmp_entity_files/Google/2020/wmp_google_entities_v040521.dta)
   - [datasets/wmp_entity_files/Facebook/2020/wmp_fb_entities_v090622.csv](https://github.com/Wesleyan-Media-Project/datasets/blob/main/wmp_entity_files/Facebook/2020/wmp_fb_entities_v090622.csv)

For our training data, only pages for which all of their pd_ids are associated with the same party_all are used. For training, the data is split by assigning 70% of the page_ids to training, and 30% of the page_ids to test. Ergo, all ads associated with a specific page_id can only be in either training or test.

The reason we split on page_id and not pd_id is because because different pd_ids of the same page are always going to be similar. If we use pd_id we could end up with some pd_ids of the same page_id ending up in the training set, and some in the test set, which would be unfair.

The following fields are used in the classifier by concatenating them in the following order, separated by a single space:

| disclaimer | page_name | ad_creative_body | ad_creative_link_caption | ad_creative_link_description | ad_creative_link_title | ocr | asr |
| ---------- | --------- | ---------------- | ------------------------ | ---------------------------- | ---------------------- | --- | --- |

Prior to the train/test split, the concatenated ads are de-duplicated, so that only one version of every concatenated ad content can go into either train/test (we could potentially only de-duplicate within page_ids, but currently don't).

### 3.2 Model

We use two versions of logistic regression classifier: one with L2 regulation and one without. We found that regulation might provide more accurate results.

You can find the trained models we provide [here](https://github.com/Wesleyan-Media-Project/party_classifier/tree/main/models). For more information about the models, you can look at the notes in the `/notes` folder.

### 3.3 Performance

Here is the model performance on the held-out test set:

```

               precision    recall  f1-score   support

          DEM       0.86      0.91      0.89     15366
        OTHER       0.84      0.05      0.10       698
          REP       0.82      0.81      0.82      9270

     accuracy                           0.85     25334
    macro avg       0.84      0.59      0.60     25334
 weighted avg       0.85      0.85      0.84     25334
```

### 3.4 Inference

Once you have your model ready, you can run the inference scripts. All the inference scripts are named starting with 03\_. For Facebook 2022 inference, you will need [fb_2022_adid_text.csv.gz](https://figshare.wesleyan.edu/account/articles/26124295)  and [fb_2022_adid_var1.csv.gz](https://figshare.wesleyan.edu/account/articles/26124340
). For Google 2022 inference, you need [g2022_adid_01062021_11082022_text.csv.gz](https://figshare.wesleyan.edu/account/articles/26124343).

In this repository, the 2020 inference scripts are written in Python with the file name ending with `.py`. To run the 2020 inference scripts, you can use the following command that calls `python` to execute the script. For example, to run the Google 2020 inference script `03_inference_google_2020.py`, you can use the following command:

```bash
python3 03_inference_google_2020.py
```

On the other hand, the 2022 inference scripts are written in Jupyter Notebook with the file name ending with `.ipynb`. To run the 2022 inference scripts, you can open the Jupyter Notebook interface by type the following in your terminal:

```bash
jupyter notebook
```

After you open the Jupyter Notebook interface, you can navigate to the folder where you have cloned the repo and open the script you want to run.

Then, click on the first code cell to select it.
Run each cell sequentially by clicking the Run button or pressing Shift + Enter.

## 4. Thank You

<p align="center"><strong>We would like to thank our supporters!</strong></p><br>

<p align="center">This material is based upon work supported by the National Science Foundation under Grant Numbers 2235006, 2235007, and 2235008.</p>

<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2235006">
    <img class="img-fluid" src="nsf.png" height="150px" alt="National Science Foundation Logo">
  </a>
</p>

<p align="center">The Cross-Platform Election Advertising Transparency Initiative (CREATIVE) is a joint infrastructure project of the Wesleyan Media Project and privacy-tech-lab at Wesleyan University in Connecticut.

<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://www.creativewmp.com/">
    <img class="img-fluid" src="CREATIVE_logo.png"  width="220px" alt="CREATIVE Logo">
  </a>
</p>

<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://mediaproject.wesleyan.edu/">
    <img src="wmp-logo.png" width="218px" height="100px" alt="Wesleyan Media Project logo">
  </a>
</p>

<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://privacytechlab.org/" style="margin-right: 20px;">
    <img src="./plt_logo.png" width="200px" alt="privacy-tech-lab logo">
  </a>
</p>
