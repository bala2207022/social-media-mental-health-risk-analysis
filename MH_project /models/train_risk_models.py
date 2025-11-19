# models/train_risk_models.py
# ------------------------------------------------------------
# Clean risk model trainer for MH_project
# Outputs:
#   out/model_scores.csv    (small, dashboard-ready)
#   out/features_window.csv (debug / backup)
# ------------------------------------------------------------

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ----------------------- CONSTANT PATHS -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data"
DEFAULT_OUT  = PROJECT_ROOT / "out"

# ----------------------- CLI ARGS -----------------------------
def get_args():
    ap = argparse.ArgumentParser("Train emotion-risk model (clean)")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--window-days", type=int, default=10)
    ap.add_argument("--save-out", type=str, default=str(DEFAULT_OUT))
    return ap.parse_args()

# ----------------------- HELPERS ------------------------------
DT_COLS   = ["created_at","timestamp","date","dt","time","created_utc"]
USER_COLS = ["user_id","author","userid","user","user_name"]
SENT_COLS = ["sentiment","label","polarity"]
LIKE_COLS = ["likes","score","upvotes"]
COM_COLS  = ["comments","num_comments","comment_count"]
ENG_COLS  = ["engagement_seconds","duration_seconds","time_spent_seconds"]

def read_first(data_dir: Path, candidates):
    """Pick first non-empty CSV from list."""
    for name in candidates:
        p = data_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    print(f"[info] loaded {name} ({len(df)} rows)")
                    return df, name
            except Exception as e:
                print(f"[warn] failed {name}: {e}")
    return None, None

def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def fix_user(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    for c in USER_COLS:
        if c in df.columns:
            df = df.rename(columns={c: "user_id"})
            df["user_id"] = df["user_id"].astype(str).str.strip()
            df = df[df["user_id"] != ""]
            return df
    print("[warn] no user_id column; columns:", list(df.columns)[:10])
    return df

def fix_sentiment(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    for c in SENT_COLS:
        if c in df.columns:
            df = df.rename(columns={c: "sentiment"})
            break
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
        mp = {
            "pos":"positive","positive":"positive","+":"positive","1":"positive",
            "neg":"negative","negative":"negative","-":"negative","-1":"negative",
            "neu":"neutral","neutral":"neutral","0":"neutral"
        }
        df["sentiment"] = df["sentiment"].map(lambda x: mp.get(x, x))
    return df

def fix_dates(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    for c in DT_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.rename(columns={c: "date"})
            df = df[~df["date"].isna()]
            return df
    return df

def filter_window(df: pd.DataFrame, today: datetime, days: int):
    if df is None or df.empty or "date" not in df.columns:
        return df
    start = today - timedelta(days=days)
    return df[(df["date"] >= start) & (df["date"] <= today)]

def add_numeric(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    df = df.copy()
    like_col = pick(df, LIKE_COLS)
    com_col  = pick(df, COM_COLS)
    eng_col  = pick(df, ENG_COLS)

    df["likes"] = pd.to_numeric(df[like_col], errors="coerce").fillna(0) if like_col else 0
    df["coms"]  = pd.to_numeric(df[com_col],  errors="coerce").fillna(0) if com_col  else 0
    df["secs"]  = pd.to_numeric(df[eng_col],  errors="coerce").fillna(0) if eng_col else 0
    return df

def agg_user(df: pd.DataFrame, tag: str):
    """Aggregate per user: count, negative posts, neg likes, neg engagement, total engagement."""
    if df is None or df.empty or "user_id" not in df.columns:
        return pd.DataFrame(columns=["user_id"])
    df = df.copy()
    has_sent = "sentiment" in df.columns
    rows = []
    for uid, part in df.groupby("user_id"):
        if has_sent:
            neg_mask = part["sentiment"] == "negative"
            neg_posts = int(neg_mask.sum())
            neg_likes = float(part.loc[neg_mask, "likes"].sum())
            neg_eng   = float(part.loc[neg_mask, "secs"].sum())
        else:
            neg_posts = 0
            neg_likes = 0.0
            neg_eng   = 0.0
        rows.append({
            "user_id": uid,
            f"{tag}_count": int(len(part)),
            f"{tag}_neg_posts": neg_posts,
            f"{tag}_neg_likes": neg_likes,
            f"{tag}_neg_eng":   neg_eng,
            f"{tag}_eng":       float(part["secs"].sum()),
        })
    return pd.DataFrame(rows)

# ------------------ FEATURE ENGINEERING ------------------------
def build_features(posts, comments, today: datetime):
    posts    = add_numeric(fix_sentiment(fix_user(fix_dates(posts))))
    comments = add_numeric(fix_sentiment(fix_user(fix_dates(comments))))

    P = agg_user(posts, "p")
    C = agg_user(comments, "c")
    feats = P.merge(C, on="user_id", how="outer").fillna(0)

    if feats.empty:
        return feats

    feats["total_events"] = feats["p_count"] + feats["c_count"]
    feats["neg_posts"]    = feats["p_neg_posts"] + feats["c_neg_posts"]
    feats["neg_likes"]    = feats["p_neg_likes"] + feats["c_neg_likes"]
    feats["neg_engagement"] = feats["p_neg_eng"] + feats["c_neg_eng"]

    feats["neg_ratio"] = np.where(
        feats["total_events"] > 0,
        feats["neg_posts"] / feats["total_events"],
        0.0,
    )

    feats["engagement_seconds"] = feats["p_eng"] + feats["c_eng"]

    # 0–1 engagement score based on total events
    lo, hi = feats["total_events"].min(), feats["total_events"].max()
    if hi > lo:
        feats["engagement_score"] = (feats["total_events"] - lo) / (hi - lo)
    else:
        feats["engagement_score"] = 0.0

    feats["date"] = today.strftime("%Y-%m-%d")
    return feats

def add_labels_quantiles(feats: pd.DataFrame):
    """
    ALWAYS create 'label' column using neg_ratio quantiles.
    Low  = lowest 33%, Moderate = middle 33%, High = top 33%.
    """
    if feats.empty:
        feats["label"] = []
        return feats

    # if all same, add tiny noise so quantiles differ
    if feats["neg_ratio"].nunique() == 1:
        feats["neg_ratio"] = feats["neg_ratio"] + np.random.uniform(-1e-6, 1e-6, size=len(feats))

    q1, q2 = feats["neg_ratio"].quantile(0.33), feats["neg_ratio"].quantile(0.66)
    feats["label"] = feats["neg_ratio"].apply(
        lambda v: "Low" if v <= q1 else ("Moderate" if v <= q2 else "High")
    )
    return feats

# -------------------------- MAIN -------------------------------
def main():
    args = get_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.save_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[debug] data_dir:", data_dir.resolve())
    print("[debug] out_dir :", out_dir.resolve())

    today = datetime.now()

    posts, _ = read_first(data_dir, ["posts_labeled.csv", "posts.csv"])
    comments, _ = read_first(data_dir, ["comments_labeled.csv", "comments.csv"])

    if posts is None and comments is None:
        print("NO posts/comments CSVs found in", data_dir)
        sys.exit(1)

    posts_w    = filter_window(posts, today, args.window_days) if posts is not None else None
    comments_w = filter_window(comments, today, args.window_days) if comments is not None else None

    rows = lambda d: 0 if d is None else len(d)
    if (rows(posts_w) + rows(comments_w)) < 10:
        print(f"[info] too few rows in last {args.window_days} days; using ALL data")
        posts_w, comments_w = posts, comments

    feats = build_features(posts_w, comments_w, today)
    if feats.empty or "user_id" not in feats.columns:
        print("NO per-user features; check your CSVs.")
        sys.exit(1)

    # ALWAYS create label → no KeyError
    feats = add_labels_quantiles(feats)

    # ----------------- MODEL -----------------
    X_cols = [
        "neg_ratio",
        "engagement_score",
        "total_events",
        "neg_posts",
        "neg_likes",
        "neg_engagement",
    ]
    for c in X_cols:
        if c not in feats.columns:
            feats[c] = 0.0

    X = feats[X_cols].astype(float)
    y = feats["label"].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(np.unique(y_enc)) < 2:
        print("[warn] only one class; skipping RF training")
        feats["p_high"] = 0.0
        feats["p_moderate"] = 0.0
        feats["p_low"] = 0.0
        feats["bucket"] = y.str.title()
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
        )

        rf = RandomForestClassifier(n_estimators=350, random_state=42)
        rf.fit(X_tr, y_tr)

        proba = rf.predict_proba(X)
        classes = list(le.classes_)  # e.g. ["High","Low","Moderate"]

        def get_p(name):
            return proba[:, classes.index(name)] if name in classes else np.zeros(len(feats))

        feats["p_high"]     = get_p("High")
        feats["p_moderate"] = get_p("Moderate")
        feats["p_low"]      = get_p("Low")

        feats["bucket"] = feats[["p_low", "p_moderate", "p_high"]].idxmax(axis=1)\
                              .str.replace("p_", "").str.title()

    # ---------- risk_score (0–1000) ----------
    raw = (
        2.0 * feats["p_high"]
        + 1.0 * feats["p_moderate"]
        + 0.6 * feats["neg_ratio"]
        + 0.3 * feats["engagement_score"]
    )
    rmin, rmax = raw.min(), raw.max()
    if rmax > rmin:
        feats["risk_score"] = 1000.0 * (raw - rmin) / (rmax - rmin)
    else:
        feats["risk_score"] = 500.0

    # ----------------- SAVE CLEAN CSV -----------------
    keep = [
        "user_id", "date",
        "risk_score", "bucket",
        "p_high", "p_moderate", "p_low",
        "total_events", "neg_ratio", "neg_posts",
        "neg_likes", "neg_engagement",
        "engagement_seconds", "engagement_score",
    ]

    feats[keep].to_csv(out_dir / "model_scores.csv", index=False)
    print("[saved]", out_dir / "model_scores.csv")

    feats.to_csv(out_dir / "features_window.csv", index=False)
    print("[saved]", out_dir / "features_window.csv")

if __name__ == "__main__":
    main()
