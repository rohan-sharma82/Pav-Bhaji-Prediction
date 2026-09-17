# Pav Bhaji Image Classification

Classifying social-media food posts as **Pav Bhaji** or **Non-Pav Bhaji** using only their caption and hashtag text — no image pixels involved.

## Overview

This project tests whether a dish can be identified purely from what people write about a photo. It covers a full text-analytics pipeline: parsing nested JSON metadata, cleaning multilingual/emoji-heavy captions, TF-IDF vectorization, and benchmarking classifiers against a heavily imbalanced target.

## Dataset

| Attribute | Value |
|---|---|
| Post records | 1,500 |
| Images available (source of ground-truth labels) | 452 |
| Pav Bhaji (class 1) | 183 (12.2%) |
| Non-Pav Bhaji (class 0) | 1,317 (87.8%) |
| Class ratio | 7.2 : 1 |

## Features Used

Only `caption_text` and `tags_text` were used as model inputs — the only fields in the JSON that are human-readable descriptions of image content. `image_id` (post shortcode) is kept solely as a join key to reattach labels, not as a predictive feature. Numeric/structural fields (dimensions, URLs, engagement counts, location, timestamps) were excluded as irrelevant to text-based classification.

## Pipeline

1. **Ingestion** — unpack archive, load nested JSON
2. **Extraction** — pull caption text and flatten hashtag lists
3. **Key building** — regex-parse image filenames, keep shortcode as ID
4. **Labelling** — map images from source folders, join labels to posts
5. **Cleaning** — lowercase, strip punctuation, tokenize, remove stopwords
6. **Vectorization** — TF-IDF, capped at 5,000 features

## Models & Results

| Model | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|---|---|---|---|---|
| Logistic Regression | 0.877 | 0.00 | 0.00 | 0.00 |
| Random Forest + SMOTE family | ~0.88 | 0.00 | 0.00 | 0.00 |
| **XGBoost (cost-sensitive)** | 0.843 | 0.25 | 0.14 | 0.18 |

Baseline models hit 88% accuracy by predicting the majority class every time — an accuracy-paradox trap given the 7.2:1 imbalance. Only cost-sensitive XGBoost produced any real signal on the minority class; synthetic oversampling (SMOTE/BorderlineSMOTE/ADASYN) never helped in this sparse, high-dimensional text space.

## Key Takeaway

Model tuning wasn't the bottleneck — **label coverage was**. Roughly two-thirds of posts have no matching image and default to class 0, introducing noise no amount of resampling can fix.

## Tech Stack

Python · pandas · NLTK · scikit-learn · imbalanced-learn · XGBoost
