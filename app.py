from flask import Flask, render_template
import praw
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Reddit API credentials
CLIENT_ID = "aIs57lI5mkP_FNveAu6-_w"  # Replace with your actual Client ID
CLIENT_SECRET = "twuIqlLAfy464jMDO2tBcBtkQaw0bw"  # Replace with your actual Client Secret
USER_AGENT = "python:subreddit_keyword_analysis:v1.0 (by /u/HesiodAgain)"  # Replace with your User Agent

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

app = Flask(__name__)

# Define the subreddit and time range
SUBREDDIT_NAME = "politics"  # Subreddit to analyze
TIME_FILTER = "week"  # Options: "day", "week", "month", "year", "all"

# Fetch posts from the subreddit
def fetch_posts(subreddit_name, time_filter):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(time_filter=time_filter, limit=500):  # Limit to 500 posts
        posts.append({
            "Title": post.title,
            "Selftext": post.selftext,
            "Score": post.score,
            "URL": post.url
        })
    return posts

# Analyze posts
def analyze_posts(posts):
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()
    keywords = []
    sentiments = []
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for post in posts:
        text = f"{post['Title']} {post['Selftext']}"
        words = text.split()
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 3]
        keywords.extend(filtered_words)
        sentiment = sia.polarity_scores(text)['compound']
        sentiments.append(sentiment)
        if sentiment > 0.1:
            sentiment_counts["Positive"] += 1
        elif sentiment < -0.1:
            sentiment_counts["Negative"] += 1
        else:
            sentiment_counts["Neutral"] += 1
    keyword_counts = Counter(keywords)
    top_keywords = keyword_counts.most_common(20)
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return top_keywords, avg_sentiment, sentiment_counts

@app.route('/')
def index():
    print(f"Fetching posts from r/{SUBREDDIT_NAME}...")
    posts = fetch_posts(SUBREDDIT_NAME, TIME_FILTER)
    print("Analyzing posts...")
    top_keywords, avg_sentiment, sentiment_counts = analyze_posts(posts)
    
    # Pass data to the frontend
    return render_template(
        'index.html',
        subreddit=SUBREDDIT_NAME,
        top_keywords=top_keywords,
        avg_sentiment=avg_sentiment,
        sentiment_counts=sentiment_counts
    )

if __name__ == '__main__':
    app.run(debug=True)
