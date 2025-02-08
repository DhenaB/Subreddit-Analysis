from flask import Flask, render_template, request, jsonify, send_file
import praw
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import nltk
from dotenv import load_dotenv
from io import BytesIO

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
    return top_keywords, avg_sentiment, sentiment_counts

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

@app.route('/analyze')
def analyze():
    subreddit_name = request.args.get('subreddit')
    if not subreddit_name:
        return jsonify({"error": "Subreddit name is required"}), 400

    try:
        posts = fetch_posts(subreddit_name)
        top_keywords, avg_sentiment, sentiment_counts = analyze_posts(posts)
        return jsonify({
            "avg_sentiment": avg_sentiment,
            "top_keywords": [{"keyword": k, "frequency": v} for k, v in top_keywords],
            "sentiment_counts": sentiment_counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download')
def download():
    subreddit_name = request.args.get('subreddit')
    if not subreddit_name:
        return "Subreddit name is required", 400

    try:
        posts = fetch_posts(subreddit_name)
        top_keywords, avg_sentiment, sentiment_counts = analyze_posts(posts)

        # Create a DataFrame for top keywords
        df_keywords = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])

        # Save DataFrame to an Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_keywords.to_excel(writer, sheet_name='Top Keywords', index=False)
        output.seek(0)

        # Return the Excel file as a downloadable response
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"{subreddit_name}_analysis.xlsx"
        )
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
