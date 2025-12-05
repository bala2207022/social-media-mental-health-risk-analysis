
import time, hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import praw

CLIENT_ID     = "o3aaZRliCC_BDipBDx6u1A"
CLIENT_SECRET = "vIb01dMIfZY8U2YSHeyCSS8eQab3mQ"
USER_AGENT    = "EmotionAware/1.0 by balakrishna"

HERE = Path(__file__).resolve().parent
DATA = (HERE.parent / "data").resolve()
DATA.mkdir(parents=True, exist_ok=True)

SUBREDDITS = [
    "mentalhealth", "depression", "Anxiety", "OffMyChest",
    "happiness", "selfimprovement", "stress"
]
POSTS_PER_SUB      = 120      
COMMENTS_PER_POST  = 50       
PAUSE              = 0.15    

def anon(name: str | None) -> str | None:
    if not name: return None
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]

def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def connect():
    r = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    r.read_only = True
    return r

def extract(reddit):
    posts, comments = [], []
    users = {}

    print("collecting...")
    for sub in SUBREDDITS:
        print(f"  • r/{sub}")
        for p in reddit.subreddit(sub).hot(limit=POSTS_PER_SUB):
            uname = getattr(p.author, "name", None)
            uid = anon(uname)
            if uname and uid and uname not in users:
                users[uname] = uid

            posts.append({
                "platform": "reddit",
                "subreddit": sub,
                "post_id": p.id,
                "user_name": uname,
                "user_id_hash": uid,
                "title": (p.title or "")[:5000],
                "body_text": (p.selftext or "")[:50000],
                "upvotes": int(getattr(p, "score", 0) or 0),
                "num_comments": int(getattr(p, "num_comments", 0) or 0),
                "created_utc": iso_utc(p.created_utc),
                "permalink": f"https://reddit.com{p.permalink}",
            })

            try:
                p.comments.replace_more(limit=0)
                taken = 0
                for c in p.comments.list():
                    cun = getattr(c.author, "name", None)
                    cuid = anon(cun)
                    if cun and cuid and cun not in users:
                        users[cun] = cuid
                    comments.append({
                        "platform": "reddit",
                        "post_id": p.id,
                        "comment_id": c.id,
                        "user_name": cun,
                        "user_id_hash": cuid,
                        "body_text": (c.body or "")[:50000],
                        "upvotes": int(getattr(c, "score", 0) or 0),
                        "created_utc": iso_utc(c.created_utc),
                    })
                    taken += 1
                    if taken >= COMMENTS_PER_POST: break
                time.sleep(PAUSE)
            except Exception as e:
                print("     (comments skipped)", e)

    df_posts = pd.DataFrame(posts).drop_duplicates(subset=["post_id"])
    df_comments = pd.DataFrame(comments).drop_duplicates(subset=["comment_id"])
    df_users = (pd.DataFrame(
        [{"platform":"reddit","user_name":k,"user_id_hash":v} for k, v in users.items()]
    ).drop_duplicates(subset=["user_id_hash"]).sort_values("user_name"))

  
    df_users.to_csv(DATA / "users.csv", index=False)
    df_posts.to_csv(DATA / "posts.csv", index=False)
    df_comments.to_csv(DATA / "comments.csv", index=False)

    print("\n extract complete")
    print(f"   users:    {len(df_users)}  -> users.csv")
    print(f"   posts:    {len(df_posts)}  -> posts.csv")
    print(f"   comments: {len(df_comments)} -> comments.csv")
    return df_users, df_posts, df_comments

NEG = {
    "sad","depress","anxiet","anxious","worry","worried","panic","lonely","alone",
    "worthless","hopeless","tired","exhaust","burnout","stress","stressed","cry",
    "tears","fear","scared","angry","rage","guilt","ashamed","suicid","self-harm",
    "hurt","pain","empty","helpless"
}
POS = {
    "happy","joy","grateful","gratitude","excited","motivat","energy","energetic",
    "inspire","confident","proud","calm","peace","optimis","hopeful","relief",
    "love","progress","improve","win"
}
def label_text(txt: str) -> str:
    if not isinstance(txt, str) or not txt.strip(): return "Negative"
    t = txt.lower()
    pos = sum(1 for w in POS if w in t)
    neg = sum(1 for w in NEG if w in t)
    return "Positive" if pos > neg and pos > 0 else "Negative"

def add_labels(df_posts, df_comments):
    if not df_posts.empty:
        pl = df_posts.copy()
        pl["binary_label"] = (pl["title"].astype(str) + " " + pl["body_text"].astype(str)).map(label_text)
        pl.to_csv(DATA / "posts.csv", index=False)            
        pl.to_csv(DATA / "posts_labeled.csv", index=False)    
    else:
        pl = pd.DataFrame()

    if not df_comments.empty:
        cl = df_comments.copy()
        cl["binary_label"] = cl["body_text"].astype(str).map(label_text)
        cl.to_csv(DATA / "comments.csv", index=False)         
        cl.to_csv(DATA / "comments_labeled.csv", index=False)
    else:
        cl = pd.DataFrame()

    frames = []
    if not pl.empty:
        frames.append(pl[["post_id","binary_label"]].rename(columns={"post_id":"entity_id"}).assign(entity_type="post"))
    if not cl.empty:
        frames.append(cl[["comment_id","binary_label"]].rename(columns={"comment_id":"entity_id"}).assign(entity_type="comment"))
    emotions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["entity_type","entity_id","binary_label"])
    emotions.to_csv(DATA / "emotions.csv", index=False)

    print("\n labeling complete")
    print(f"   posts_labeled:    {len(pl)}")
    print(f"   comments_labeled: {len(cl)}")
    print(f"   emotions:         {len(emotions)}")
    return pl, cl, emotions

def to_utc_date(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.date

def build_daily(pl, cl):
    parts = []
    if not pl.empty:
        p = pl.copy(); p["date"] = to_utc_date(p["created_utc"])
        parts.append(p.groupby(["date","binary_label"], as_index=False).size().rename(columns={"size":"count"}))
    if not cl.empty:
        c = cl.copy(); c["date"] = to_utc_date(c["created_utc"])
        parts.append(c.groupby(["date","binary_label"], as_index=False).size().rename(columns={"size":"count"}))
    if parts:
        daily = pd.concat(parts, ignore_index=True)
        daily = (daily.groupby(["date","binary_label"], as_index=False)["count"].sum()
                      .sort_values(["date","binary_label"]))
    else:
        daily = pd.DataFrame(columns=["date","binary_label","count"])
    daily.to_csv(DATA / "emotions_daily.csv", index=False)
    print("\n daily summary complete")
    print(f"   emotions_daily: {len(daily)}")
    print(f"\n all files in: {DATA}")

if __name__ == "__main__":
    reddit = connect()
    u, p, c = extract(reddit)
    pl, cl, em = add_labels(p, c)
    build_daily(pl, cl)
