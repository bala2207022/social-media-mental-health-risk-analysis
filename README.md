# 🧠 Mental Health Risk Detection from Reddit

**An AI-powered system that analyzes Reddit posts and comments to detect early mental health risk indicators using NLP, machine learning, and interactive visualization.**

> ⚠️ **Academic Research Only** — This is for research purposes only, not clinical diagnosis.

---

## 📋 What This Project Does

This project follows a **complete data science pipeline**:

1. **Collect** → Fetch Reddit posts and comments using PRAW API
2. **Label** → Detect emotions using NLP (positive, negative, neutral)
3. **Engineer** → Create features like sentiment trends, engagement patterns
4. **Model** → Train a Random Forest classifier to predict risk levels (Low, Moderate, High)
5. **Visualize** → Display results in an interactive dashboard

---

## 📁 Project Structure

```
MH_project/
│
├── api/
│   ├── reddit_connect.py         # Connect to Reddit API
│   ├── run_pipeline.py           # Main data collection pipeline
│   └── run_all.py                # Run everything at once
│
├── data/                          # Raw and processed data
│   ├── posts.csv / posts_labeled.csv
│   ├── comments.csv / comments_labeled.csv
│   ├── emotions.csv / emotions_daily.csv
│   └── users.csv
│
├── models/
│   └── train_risk_models.py       # Train ML classifier & generate scores
│
├── dashboard/
│   └── dashboard.py               # Interactive visualization dashboard
│
└── out/                           # Final outputs
    ├── model_scores.csv           # Risk scores for each user
    └── features_window.csv        # Engineered features
```

---

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Collect & Process Data**
```bash
python api/run_pipeline.py
```

### **3. Train Model & Generate Risk Scores**
```bash
python models/train_risk_models.py
```

### **4. View Interactive Dashboard**
```bash
streamlit run dashboard/dashboard.py
```

Then open `http://localhost:8501` in your browser.

---

## 📊 What Each Component Does

| Component | Purpose | Output |
|-----------|---------|--------|
| **reddit_connect.py** | Fetch posts, comments, user data from Reddit | `posts.csv`, `comments.csv` |
| **label_emotions.py** | Detect emotion/sentiment in text (NLP) | `posts_labeled.csv`, `comments_labeled.csv` |
| **train_risk_models.py** | Build ML model, predict risk scores | `model_scores.csv` |
| **dashboard.py** | Visualize user profiles, risk trends, feature importance | Interactive web app |

---

## 📈 Key Features Generated

The model uses these features to predict mental health risk:

- **Posting Frequency** — How active is the user?
- **Sentiment Ratio** — What % of posts are negative?
- **Engagement Metrics** — Likes, comments, time spent
- **Emotional Trends** — Changes in emotion over time
- **Activity Windows** — Recent behavior patterns (10-day default)

---

## 🎯 Risk Classification

The model outputs three risk levels for each user:

| Risk Level | Description | Action |
|-----------|-------------|--------|
| **Low** | Minimal emotional distress signals | Monitor periodically |
| **Moderate** | Some negative patterns detected | Consider follow-up |
| **High** | Strong indicators of mental health risk | Prioritize intervention |

---

## 🔗 Live Demos

- **📺 Full Presentation (PPT):** [View Here](https://bala2207022.github.io/ppt-showcase/)
- **📊 Dashboard Demo Video:** [View Here](https://bala2207022.github.io/ppt-showcase/)

---

## ✅ Data & Privacy

✓ **Public data only** — All data from public Reddit posts  
✓ **Fully anonymized** — User names replaced with anonymous IDs  
✓ **No private data** — No passwords, emails, or personal info collected  
✓ **Research-only** — Not intended for real-world medical use  

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core language |
| **PRAW** | Reddit API wrapper |
| **Pandas** | Data processing |
| **Scikit-learn** | Machine learning |
| **Plotly** | Interactive charts |
| **Streamlit** | Web dashboard |

---

## 📝 Example Workflow

```bash
# Step 1: Collect Reddit data
$ python api/run_pipeline.py
→ Creates: posts.csv, comments.csv

# Step 2: Label emotions & engineer features
→ Creates: posts_labeled.csv, emotions.csv

# Step 3: Train model & score users
$ python models/train_risk_models.py
→ Creates: out/model_scores.csv

# Step 4: Launch dashboard
$ streamlit run dashboard/dashboard.py
→ View at: http://localhost:8501
```

---

## 📚 How to Use This Code

1. **For Research** → Modify the feature engineering to test new hypotheses
2. **For Learning** → Study the NLP pipeline and ML workflow
3. **For Extension** → Add new data sources (Twitter, Discord, etc.)

---

## ⚖️ Important Notes

- **This is NOT a diagnostic tool** — Only for research/analysis
- **Consult professionals** — Mental health issues require expert guidance
- **Respect privacy** — Always anonymize data and follow platform ToS
- **Bias awareness** — Social media signals don't capture full picture

---

## 🤝 Contributing

Found a bug or want to improve? Feel free to:
1. Report issues
2. Suggest features
3. Submit pull requests

---
