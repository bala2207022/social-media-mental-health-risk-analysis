from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
DATA = (HERE.parent / "data").resolve()
POSTS_IN     = DATA / "posts_latest.csv"
COMMENTS_IN  = DATA / "comments_latest.csv"
EMO_OUT      = DATA / "emotions_latest.csv"
POSTS_OUT    = DATA / "posts_labeled_latest.csv"
COMMENTS_OUT = DATA / "comments_labeled_latest.csv"
DAILY_OUT    = DATA / "emotions_daily_latest.csv"

print(" Using data folder:", DATA)
print("   posts_latest.csv exists:", POSTS_IN.exists())
print("   comments_latest.csv exists:", COMMENTS_IN.exists())

NEG = {
    "sad","depress","anxiet","anxious","worry","worried","panic","lonely","alone","worthless",
    "hopeless","tired","exhaust","burnout","stress","stressed","cry","tears","fear","scared",
    "angry","rage","guilt","ashamed","suicid","self-harm","hurt","pain","empty","helpless"
}
POS = {
    "happy","joy","grateful","gratitude","excited","motivat","energy","energetic","inspire",
    "confident","proud","calm","peace","optimis","hopeful","relief","love","progress","improve","win"
}

def lex_label(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Negative" 
    t = text.lower()
    neg = sum(1 for w in NEG if w in t)
    pos = sum(1 for w in POS if w in t)
    if pos > neg and pos > 0: return "Positive"
    return "Negative"


posts = pd.read_csv(POSTS_IN) if POSTS_IN.exists() else pd.DataFrame()
comments = pd.read_csv(COMMENTS_IN) if COMMENTS_IN.exists() else pd.DataFrame()
print(f"Loaded: posts={len(posts)}, comments={len(comments)}")

def label_posts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    print("Labeling posts…")
    out = df.copy()
    out["binary_label"] = (out["title"].astype(str) + " " + out["body_text"].astype(str)).map(lex_label)
    return out

def label_comments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    print("Labeling comments…")
    out = df.copy()
    out["binary_label"] = out["body_text"].astype(str).map(lex_label)
    return out

posts_labeled = label_posts(posts)
comments_labeled = label_comments(comments)

def make_emotions(posts_lab: pd.DataFrame, comments_lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not posts_lab.empty:
        rows.append(posts_lab[["post_id","binary_label"]].rename(columns={"post_id":"entity_id"}).assign(entity_type="post"))
    if not comments_lab.empty:
        rows.append(comments_lab[["comment_id","binary_label"]].rename(columns={"comment_id":"entity_id"}).assign(entity_type="comment"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["entity_type","entity_id","binary_label"])

emotions = make_emotions(posts_labeled, comments_labeled)


def to_utc_date(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.date

daily_frames = []
if not posts_labeled.empty:
    p = posts_labeled.copy()
    p["date"] = to_utc_date(p["created_utc"])
    daily_frames.append(p.groupby(["date","binary_label"], as_index=False).size().rename(columns={"size":"count"}))
if not comments_labeled.empty:
    c = comments_labeled.copy()
    c["date"] = to_utc_date(c["created_utc"])
    daily_frames.append(c.groupby(["date","binary_label"], as_index=False).size().rename(columns={"size":"count"}))
daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(columns=["date","binary_label","count"])
if not daily.empty:
    daily = (daily.groupby(["date","binary_label"], as_index=False)["count"].sum()
                   .sort_values(["date","binary_label"]))

posts_labeled.to_csv(POSTS_OUT, index=False)
comments_labeled.to_csv(COMMENTS_OUT, index=False)
emotions.to_csv(EMO_OUT, index=False)
daily.to_csv(DAILY_OUT, index=False)

print("\nSaved:")
print("  -", EMO_OUT.name, f"({len(emotions)})")
print("  -", POSTS_OUT.name, f"({len(posts_labeled)})")
print("  -", COMMENTS_OUT.name, f"({len(comments_labeled)})")
print("  -", DAILY_OUT.name, f"({len(daily)})")
print(" Folder:", DATA)
