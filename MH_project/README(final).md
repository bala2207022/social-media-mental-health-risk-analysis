# 🧠 Social Media Engagement & Mental Health Risk Analysis

An AI-powered pipeline that analyzes social media activity (starting with Reddit) to explore how engagement patterns relate to early mental health risk indicators. The system includes data collection, NLP-based emotion detection, machine learning classification, and an interactive dashboard for visualization.

> ⚠️ Academic Research Only — This project is not a clinical diagnosis tool.

## 🎯 Research Goal
- Analyze how users engage with social platforms.
- Detect emotional patterns in posts and comments.
- Build ML models to estimate mental health risk.
- Visualize emotional trends and behavioral changes.
- Provide a scalable framework for future social media sources.

## 🔍 Project Workflow
1. **Data Collection**  
   Collect posts, comments, and engagement data from Reddit using API scripts.

2. **Emotion & Sentiment Analysis (NLP)**  
   Clean text, detect emotions, label each post/comment, and build daily emotion summaries.

3. **Feature Engineering**  
   Generate behavioral and emotional metrics such as posting frequency, sentiment ratios, engagement intensity, emotional swings, and window-based trend features.

4. **Risk Modeling**  
   Train a Random Forest classifier that predicts Low / Moderate / High mental health risk based on engineered features.

5. **Dashboard Visualization**  
   Explore users, risk scores, feature importance, emotional timelines, and behavior trends using a Streamlit dashboard.

## 📁 Project Structure
MH_project/  
├── api/  
│   ├── reddit_connect.py  
│   ├── run_pipeline.py  
│   └── run_all.py  
├── data/  
│   ├── posts.csv / posts_labeled.csv  
│   ├── comments.csv / comments_labeled.csv  
│   ├── emotions.csv / emotions_daily.csv  
│   └── users.csv  
├── models/  
│   └── train_risk_models.py  
├── dashboard/  
│   └── dashboard.py  
└── out/  
    ├── model_scores.csv  
    └── features_window.csv  

## 🚀 Quick Start
### Install Dependencies
```
pip install -r requirements.txt
```

### Collect & Process Data
```
python api/run_pipeline.py
```

### Train Model
```
python models/train_risk_models.py
```

### Launch Dashboard
```
streamlit run dashboard/dashboard.py
```
Open in browser: http://localhost:8501

## 📊 Analysis & Insights

### Engagement vs Emotion
- Measures posting frequency, comment frequency, and average engagement.
- Computes negative, positive, and neutral sentiment ratios.
- High-risk users often show more negative emotional patterns and irregular engagement.

### Emotional Trends Over Time
- Daily emotional scores reveal mood shifts.
- Sentiment timelines highlight periods of emotional decline.
- Sharp increases in negative emotion often correlate with elevated risk.

### Window-Based Trends
- Uses 10-day (or N-day) sliding windows to capture recent emotional behavior.
- Calculates metrics like:
  - Negative ratio in last N days
  - Average sentiment in last N days
  - Emotional stability or fluctuation
  - Activity trends
- These features improve early detection of potential risk.

### Feature Importance
The model identifies which factors contribute most to risk, typically:
- Negative-to-positive sentiment ratio  
- Posting frequency patterns  
- Sudden emotional drops  
- Emotional variability  
- Engagement consistency  
This helps interpret why the user was classified in a particular risk category.

### User-Level Risk Profiles
Each anonymized user includes:  
- Engagement summary  
- Emotion summary  
- Risk level (Low, Moderate, High)  
- An emotional timeline chart  
- Feature values that drove the risk decision  

## 🔗 Live Demos
- Presentation: https://bala2207022.github.io/ppt-showcase/  
- Dashboard Video: https://bala2207022.github.io/ppt-showcase/

## 🛡️ Data & Privacy
- Only uses public social media content.
- All usernames are anonymized.
- No private or sensitive data collected.
- Strictly for educational and research use.

## 🛠️ Tech Stack
| Layer | Technology |
|-------|------------|
| Language | Python |
| API | PRAW |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Plotly / Altair |
| Dashboard | Streamlit |

## 📝 Example Workflow
```
python api/run_pipeline.py
python models/train_risk_models.py
streamlit run dashboard/dashboard.py
```

## 💡 Use Cases
- Research on emotional trends in social media.
- End-to-end NLP + ML + Dashboard academic projects.
- Analysis of engagement behavior related to mental health.
- Experimentation with feature engineering and risk modeling.

## ⚖️ Ethical & Practical Notes
- Not a medical tool.
- Social media does not represent complete mental health.
- Must be used responsibly within academic guidelines.
- Model outcomes can be biased due to limited or skewed data.

## 🤝 Contributing
1. Fork the repository  
2. Create a feature branch  
3. Submit a merge request with a clear explanation  
