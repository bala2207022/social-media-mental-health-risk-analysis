
import os, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import praw

CLIENT_ID     = "o3aaZRliCC_BDipBDx6u1A"
CLIENT_SECRET = "vIb01dMIfZY8U2YSHeyCSS8eQab3mQ"
USER_AGENT    = "EmotionAware/1.0 by balakrishna"

SUBREDDITS = ["mentalhealth", "depression", "happiness", "stress", "OffMyChest"]
POSTS_PER_SUB = 100          
COMMENTS_PER_POST = 50       
SLEEP_BETWEEN_POSTS = 0.15  


HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE.parent / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)
reddit.read_only = True

def anon_user(name: str | None) -> str | None:
    """Hash username for anonymity; return None if author is deleted."""
    if not name:
        return None
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]

posts_rows = []
comments_rows = []
user_map = {}  

print(" Collecting Reddit data...\n")

for sub in SUBREDDITS:
    print(f"  • r/{sub}")
    for post in reddit.subreddit(sub).hot(limit=POSTS_PER_SUB):
       
        author_name = getattr(post.author, "name", None)
        uid = anon_user(author_name)
        if author_name and uid and author_name not in user_map:
            user_map[author_name] = uid

        
        posts_rows.append({
            "platform": "reddit",
            "subreddit": sub,
            "post_id": post.id,
            "user_name": author_name,
            "user_id_hash": uid,
            "title": (post.title or "")[:5000],
            "body_text": (post.selftext or "")[:50000],
            "upvotes": int(getattr(post, "score", 0) or 0),
            "num_comments": int(getattr(post, "num_comments", 0) or 0),
            "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
            "permalink": f"https://reddit.com{post.permalink}",
        })

        
        try:
            post.comments.replace_more(limit=0)
            count = 0
            for c in post.comments.list():
                c_author = getattr(c.author, "name", None)
                c_uid = anon_user(c_author)
                if c_author and c_uid and c_author not in user_map:
                    user_map[c_author] = c_uid

                comments_rows.append({
                    "platform": "reddit",
                    "post_id": post.id,
                    "comment_id": c.id,
                    "user_name": c_author,
                    "user_id_hash": c_uid,
                    "body_text": (c.body or "")[:50000],
                    "upvotes": int(getattr(c, "score", 0) or 0),
                    "created_utc": datetime.fromtimestamp(c.created_utc, tz=timezone.utc).isoformat(),
                })
                count += 1
                if count >= COMMENTS_PER_POST:
                    break
            time.sleep(SLEEP_BETWEEN_POSTS)
        except Exception as e:
            print(f"   comments skipped: {e}")

posts_df = pd.DataFrame(posts_rows)
comments_df = pd.DataFrame(comments_rows)
users_df = pd.DataFrame(
    [{"platform": "reddit", "user_name": name, "user_id_hash": uid}
     for name, uid in user_map.items()]
).sort_values("user_name")

def to_utc_date(series):
    """Convert any timestamp (aware or naive) to UTC date safely."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.date


beh_parts = []

if not posts_df.empty:
    p = posts_df.copy()
    p["date"] = to_utc_date(p["created_utc"])
    beh_posts = (p.dropna(subset=["user_id_hash"])
                   .groupby(["user_id_hash", "date"], as_index=False)
                   .agg(total_posts=("post_id", "count")))
    beh_parts.append(beh_posts)

if not comments_df.empty:
    c = comments_df.copy()
    c["date"] = to_utc_date(c["created_utc"])
    beh_comments = (c.dropna(subset=["user_id_hash"])
                      .groupby(["user_id_hash", "date"], as_index=False)
                      .agg(total_comments=("comment_id", "count")))
    beh_parts.append(beh_comments)

if beh_parts:
    behavior_df = beh_parts[0]
    for extra in beh_parts[1:]:
        behavior_df = behavior_df.merge(extra, on=["user_id_hash", "date"], how="outer")
    behavior_df["total_posts"] = behavior_df.get("total_posts", 0).fillna(0).astype(int)
    behavior_df["total_comments"] = behavior_df.get("total_comments", 0).fillna(0).astype(int)
else:
    behavior_df = pd.DataFrame(columns=["user_id_hash", "date", "total_posts", "total_comments"])

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
paths = {
    "users": DATA_DIR / f"users_{stamp}.csv",
    "posts": DATA_DIR / f"posts_{stamp}.csv",
    "comments": DATA_DIR / f"comments_{stamp}.csv",
    "behavior_summary": DATA_DIR / f"behavior_summary_{stamp}.csv",
}

users_df.to_csv(paths["users"], index=False)
posts_df.to_csv(paths["posts"], index=False)
comments_df.to_csv(paths["comments"], index=False)
behavior_df.to_csv(paths["behavior_summary"], index=False)

users_df.to_csv(DATA_DIR / "users_latest.csv", index=False)
posts_df.to_csv(DATA_DIR / "posts_latest.csv", index=False)
comments_df.to_csv(DATA_DIR / "comments_latest.csv", index=False)
behavior_df.to_csv(DATA_DIR / "behavior_summary_latest.csv", index=False)


print("\n DONE! Reddit data collection complete.")
print(f"   Users:              {len(users_df):>6} → {paths['users'].name}")
print(f"   Posts:              {len(posts_df):>6} → {paths['posts'].name}")
print(f"   Comments:           {len(comments_df):>6} → {paths['comments'].name}")
print(f"   Behavior summary:   {len(behavior_df):>6} → {paths['behavior_summary'].name}")
print(f"\n All files saved in: {DATA_DIR}")
