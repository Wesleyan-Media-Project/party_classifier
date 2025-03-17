#!/bin/bash

# Training on Meta and Google 2022
python code/01_prepare_train.py
python code/02_train.py
# Inference on FB 2022
python code/03_inference_fb_2022.py
# Inference on Google 2022
python code/03_inference_google_2022.py