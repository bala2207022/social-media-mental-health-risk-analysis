# api/label_emotions.py
from pathlib import Path
import re
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = (HERE.parent / "data").resolve()

POSTS_IN    = DATA / "posts_latest.csv"
COMMENTS_IN = DATA / "comments_latest.csv"
EMO_OUT     = DATA / "emotions_latest.csv"
POSTS_OUT   = DATA / "posts_labeled.csv"
COMMENTS_OUT= DATA / "comments_labeled.csv"

NEG_WORDS = {
    "sad","depress","anxiet","anxious","worry","worried","panic","lonely","alone","worthless",
    "hopeless","tired","exhaust","burnout","stress","stressed","cry","tears","fear","scared",
    "angry","rage","guilt","ashamed","suicid","self-harm","hurt","pain","empty"
}
POS_WORDS = {
    "happy","joy","grateful","gratitude","excited","motivat","energy","energetic","inspire",
    "confident","proud","calm","peace","optimis","hopeful","relief","love","progress","improve","win"
}

def lexicon_label(text: str) -> tuple[str,float,str]:
    """Return (label, score, source) using keyword hits."""
    if not isinstance(text, str) or not text.strip():
        return ("Neutral", 0.0, "lexicon")
    t = text.lower()
    neg_hits = sum(1 for w in NEG_WORDS if w in t)
    pos_hits = sum(1 for w in POS_WORDS if w in t)
    if neg_hits > pos_hits and neg_hits > 0:
        return ("Negative", min(1.0, 0.4 + 0.1*neg_hits), "lexicon")
    if pos_hits > neg_hits and pos_hits > 0:
        return ("Positive", min(1.0, 0.4 + 0.1*pos_hits), "lexicon")
    return ("Neutral", 0.0, "lexicon")

pipe = None
try:
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
except Exception:
    pipe = None  
def refine_with_model(text: str, cur_label: str, cur_score: float):
    if pipe is None or not isinstance(text, str) or not text.strip():
        return (cur_label, cur_score, "lexicon")
    try:
        pred = pipe(text[:512])[0]  
        lbl = pred["label"].upper()
        score = float(pred["score"])
        if lbl in ("NEGATIVE","POSITIVE"):
            final = "Negative" if lbl=="NEGATIVE" else "Positive"
            # prefer model if confident, else keep lexicon
            if score >= 0.70 or cur_label=="Neutral":
                return (final, score, "model")
        return (cur_label, cur_score, "lexicon")
    except Exception:
        return (cur_label, cur_score, "lexicon")

# Process helper 
def process_df(df: pd.DataFrame, id_col: str, text_cols: list[str], entity_type: str):
    rows = []
    if df.empty: 
        return pd.DataFrame(columns=["entity_type","entity_id","binary_label","score","source"])
    for _, r in df.iterrows():
        txt = " ".join([str(r.get(c,"")) for c in text_cols])
        base_label, base_score, source = lexicon_label(txt)
        final_label, final_score, final_source = refine_with_model(txt, base_label, base_score)
        rows.append({
            "entity_type": entity_type,
            "entity_id": r[id_col],
            "binary_label": final_label if final_label in ("Positive","Negative") else "Neutral",
            "score": round(final_score, 4),
            "source": final_source
        })
    return pd.DataFrame(rows)

#  Load input 
posts = pd.read_csv(POSTS_IN) if POSTS_IN.exists() else pd.DataFrame()
comments = pd.read_csv(COMMENTS_IN) if COMMENTS_IN.exists() else pd.DataFrame()

# Make emotions table 
emo_posts = process_df(posts, "post_id", ["title","body_text"], "post")
emo_comments = process_df(comments, "comment_id", ["body_text"], "comment")
emotions = pd.concat([emo_posts, emo_comments], ignore_index=True)

#  Save emotions and labeled copies 
emotions.to_csv(EMO_OUT, index=False)

def attach_labels(df, id_col, entity_type):
    if df.empty or emotions.empty: 
        return df
    subset = emotions[emotions["entity_type"]==entity_type][[ "entity_id","binary_label","score" ]]
    return df.merge(subset, left_on=id_col, right_on="entity_id", how="left").drop(columns=["entity_id"])

posts_labeled = attach_labels(posts, "post_id", "post")
comments_labeled = attach_labels(comments, "comment_id", "comment")

posts_labeled.to_csv(POSTS_OUT, index=False)
comments_labeled.to_csv(COMMENTS_OUT, index=False)

print("Saved:")
print(f"  - {EMO_OUT.name}  ({len(emotions)} rows)")
print(f"  - {POSTS_OUT.name} ({len(posts_labeled)} rows)")
print(f"  - {COMMENTS_OUT.name} ({len(comments_labeled)} rows)")
print(f"Folder: {DATA}")
