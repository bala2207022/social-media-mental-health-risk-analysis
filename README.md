# Social Media Engagement & Mental Health Risk Analysis

### AI-Powered Early Detection System Using Reddit, NLP, Feature Engineering & Machine Learning

## 1. Introduction

This project analyzes social media behavior to detect early signs of
mental health risk. Using Reddit mental health communities, the pipeline
collects posts and comments, labels emotion, builds features, predicts
risk, and visualizes results.

## 2. Project Architecture

    MH_project/
    │
    ├── api/                        
    │   ├── reddit_connect.py
    │   ├── run_all.py
    │   ├── run_pipeline.py
    │
    ├── data/                       
    │   ├── posts.csv
    │   ├── comments.csv
    │   ├── posts_labeled.csv
    │   ├── comments_labeled.csv
    │   ├── emotions.csv
    │   ├── emotions_daily.csv
    │   ├── users.csv
    │   ├── feature_importance_rf.csv
    │   ├── features_window.csv
    │   ├── model_scores.csv
    │
    ├── dashboard/                  
    │   └── dashboard.py
    │
    ├── models/
    │   └── train_risk_models.py
    │
    ├── json/
    │   ├── label_emotions.py
    │   ├── label_emotions_binary.py
    │
    └── out/
        ├── features_window.csv
        └── model_scores.csv

## 3. Data Collection

Uses PRAW to fetch Reddit posts, comments, timestamps, engagement,
anonymous user IDs.

## 4. Emotion Labeling

Uses lexicon rules, emoji patterns, confidence scoring. Outputs
posts_labeled.csv, comments_labeled.csv, emotions.csv.

## 5. Feature Engineering

Generates user-level features stored in features_window.csv.

## 6. Machine Learning Model

RandomForestClassifier predicts mental health risk categories and risk
scores.

## 7. Dashboard

Streamlit dashboard visualizes risk scores and user patterns.

## 8. Pipeline Commands

    pip install -r requirements.txt
    python api/run_pipeline.py
    python models/train_risk_models.py
    streamlit run dashboard/dashboard.py

## 9. Ethical Notes

Data is public and anonymized. Not for clinical diagnosis.

## 10. Conclusion

Complete AI-based mental health risk detection system with NLP, ML, and
dashboard visualization.
