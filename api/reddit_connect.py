import praw

reddit = praw.Reddit(
    client_id="o3aaZRliCC_BDipBDx6u1A",             
    client_secret="vIb01dMIfZY8U2YSHeyCSS8eQab3mQ",  
    user_agent="EmotionAware/1.0 by balakrishna"    
)

reddit.read_only = True

print("Connected to Reddit successfully!\n")

for post in reddit.subreddit("mentalhealth").hot(limit=5):
    print("-", post.title)
