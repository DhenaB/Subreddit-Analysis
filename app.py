from flask import Flask, render_template, request, jsonify
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
import yfinance as yf

# Load environment variables
load_dotenv()

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('vader_lexicon')

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

# Analyze posts
def analyze_posts(posts):
    stop_words = set(stopwords.words('english'))  # Requires 'stopwords' dataset
    sia = SentimentIntensityAnalyzer()  # Requires 'vader_lexicon' dataset
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


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        subreddit_name = data.get('subreddit')
        time_filter = data.get('time_filter', 'week')  # Default to "week"

        if not subreddit_name:
            return jsonify({"error": "Subreddit name is required"}), 400

        print(f"Fetching posts for subreddit: {subreddit_name} with time filter: {time_filter}")
        posts = fetch_posts(subreddit_name, time_filter=time_filter)
        print(f"Fetched {len(posts)} posts")

        print("Analyzing posts...")
        top_keywords, avg_sentiment, sentiment_counts, keywords = analyze_posts(posts)
        print(f"Analysis complete. Top keywords: {top_keywords}")

        # Generate Word Cloud
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
            wordcloud_path = BytesIO()
            wordcloud.to_image().save(wordcloud_path, format='PNG')
            wordcloud_base64 = base64.b64encode(wordcloud_path.getvalue()).decode('utf-8')  # Encode as base64
        except Exception as e:
            print(f"Error generating Word Cloud: {e}")
            wordcloud_base64 = None

        # Generate Sentiment Distribution Bar Chart
        try:
            sentiment_labels = list(sentiment_counts.keys())
            sentiment_values = list(sentiment_counts.values())
            plt.figure(figsize=(6, 4))
            plt.bar(sentiment_labels, sentiment_values, color=['green', 'blue', 'red'])
            plt.title('Sentiment Distribution')
            sentiment_chart_path = BytesIO()
            plt.savefig(sentiment_chart_path, format='png')
            plt.close()
            sentiment_chart_base64 = base64.b64encode(sentiment_chart_path.getvalue()).decode('utf-8')  # Encode as base64
        except Exception as e:
            print(f"Error generating Sentiment Chart: {e}")
            sentiment_chart_base64 = None

        # Generate Top Keywords Bar Chart
        try:
            keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])
            plt.figure(figsize=(8, 4))
            plt.barh(keywords_df['Keyword'], keywords_df['Frequency'], color='orange')
            plt.xlabel('Frequency')
            plt.title('Top Keywords')
            keywords_chart_path = BytesIO()
            plt.savefig(keywords_chart_path, format='png')
            plt.close()
            keywords_chart_base64 = base64.b64encode(keywords_chart_path.getvalue()).decode('utf-8')  # Encode as base64
        except Exception as e:
            print(f"Error generating Keywords Chart: {e}")
            keywords_chart_base64 = None

        return jsonify({
            "avg_sentiment": avg_sentiment,
            "top_keywords": [{"keyword": k, "frequency": v} for k, v in top_keywords],
            "sentiment_counts": sentiment_counts,
            "wordcloud": wordcloud_base64,
            "sentiment_chart": sentiment_chart_base64,
            "keywords_chart": keywords_chart_base64
        })
    except Exception as e:
        print(f"Error in /analyze: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
