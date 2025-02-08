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
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
        wordcloud_path = BytesIO()
        wordcloud.to_image().save(wordcloud_path, format='PNG')
        wordcloud_base64 = base64.b64encode(wordcloud_path.getvalue()).decode('utf-8')

        # Generate Sentiment Distribution Bar Chart
        sentiment_labels = list(sentiment_counts.keys())
        sentiment_values = list(sentiment_counts.values())
        plt.figure(figsize=(6, 4))
        plt.bar(sentiment_labels, sentiment_values, color=['green', 'blue', 'red'])
        plt.title('Sentiment Distribution')
        sentiment_chart_path = BytesIO()
        plt.savefig(sentiment_chart_path, format='png')
        plt.close()
        sentiment_chart_base64 = base64.b64encode(sentiment_chart_path.getvalue()).decode('utf-8')

        # Generate Top Keywords Bar Chart
        keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])
        plt.figure(figsize=(8, 4))
        plt.barh(keywords_df['Keyword'], keywords_df['Frequency'], color='orange')
        plt.xlabel('Frequency')
        plt.title('Top Keywords')
        keywords_chart_path = BytesIO()
        plt.savefig(keywords_chart_path, format='png')
        plt.close()
        keywords_chart_base64 = base64.b64encode(keywords_chart_path.getvalue()).decode('utf-8')

        # Fetch stock market performance
        stock_market_performance = get_stock_market_performance()

        return jsonify({
            "avg_sentiment": avg_sentiment,
            "top_keywords": [{"keyword": k, "frequency": v} for k, v in top_keywords],
            "sentiment_counts": sentiment_counts,
            "wordcloud": wordcloud_base64,
            "sentiment_chart": sentiment_chart_base64,
            "keywords_chart": keywords_chart_base64,
            "stock_market_performance": stock_market_performance
        })
    except Exception as e:
        print(f"Error in /analyze: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download', methods=['GET'])
def download():
    try:
        subreddit_name = request.args.get('subreddit')
        time_filter = request.args.get('time_filter', 'week')  # Default to "week"

        if not subreddit_name:
            return "Subreddit name is required", 400

        print(f"Fetching posts for subreddit: {subreddit_name} with time filter: {time_filter}")
        posts = fetch_posts(subreddit_name, time_filter=time_filter)
        print(f"Fetched {len(posts)} posts")

        print("Analyzing posts...")
        top_keywords, avg_sentiment, sentiment_counts, _ = analyze_posts(posts)
        print(f"Analysis complete. Top keywords: {top_keywords}")

        # Create a DataFrame for top keywords
        df_keywords = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])
        print("Created DataFrame for top keywords")

        # Add sentiment counts to the DataFrame
        df_sentiment = pd.DataFrame(list(sentiment_counts.items()), columns=["Sentiment", "Count"])
        print("Created DataFrame for sentiment counts")

        # Save DataFrames to an Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_keywords.to_excel(writer, sheet_name='Top Keywords', index=False)
            df_sentiment.to_excel(writer, sheet_name='Sentiment Distribution', index=False)
        output.seek(0)
        print("Excel file generated successfully")

        # Return the Excel file as a downloadable response
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"{subreddit_name}_analysis.xlsx"
        )
    except Exception as e:
        print(f"Error in /download: {e}")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
import yfinance as yf

def get_stock_market_performance():
    try:
        # Define stock market indices
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC"
        }

        performance = {}
        for name, ticker in indices.items():
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                close_price = data['Close'].iloc[-1]
                open_price = data['Open'].iloc[0]
                change = close_price - open_price
                percent_change = (change / open_price) * 100
                performance[name] = {
                    "Close": round(close_price, 2),
                    "Change": round(change, 2),
                    "Percent Change": round(percent_change, 2)
                }
        return performance
    except Exception as e:
        print(f"Error fetching stock market performance: {e}")
        return None
