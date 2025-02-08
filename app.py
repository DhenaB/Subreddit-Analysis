from flask import Flask, render_template, request, jsonify, send_file
import praw
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import nltk
from dotenv import load_dotenv
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Reddit API credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

app = Flask(__name__)

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('vader_lexicon')

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
    return top_keywords, avg_sentiment, sentiment_counts, keywords

# Fetch posts from subreddit
def fetch_posts(subreddit_name, time_filter="week", limit=500):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(time_filter=time_filter, limit=limit):
        posts.append({
            "Title": post.title,
            "Selftext": post.selftext,
            "Score": post.score,
            "URL": post.url
        })
    return posts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        subreddit_name = data.get('subreddit')
        time_filter = data.get('time_filter', 'week')  # Default to
