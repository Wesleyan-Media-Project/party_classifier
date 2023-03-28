#!/bin/bash

# Training on FB 2020
Rscript --no-environ --no-save 01_create_training_data.R
python 02_train.py
# Inference on FB 2020
python 03_inference_fb_140m.py