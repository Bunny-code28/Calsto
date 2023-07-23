from flask import Flask, request
import yfinance as yf
import numpy as np
from newsapi import NewsApiClient
from textblob import TextBlob

app = Flask(__name__)

# Initialize the News API client
newsapi = NewsApiClient(api_key='API KEY')

# Initialize the sentiment analyzer
analyzer = TextBlob

@app.route('/predict', methods=['POST'])
def predict():
    # Get the stock symbol from the request body
    stock_symbol = request.json['stock_symbol']

    # Get the historical data for the stock
    data = yf.download(stock_symbol, period="max")

    # Get news articles for the stock
    articles = newsapi.get_everything(q=stock_symbol)

    # Calculate the average sentiment for the articles
    sentiments = []
    for article in articles['articles']:
        # Get the full text of the article
        full_text = article['content']
        # Calculate the sentiment of the article
        sentiment = analyzer(full_text).sentiment.polarity
        sentiments.append(sentiment)

    if len(sentiments) > 0:
        average_sentiment = np.mean(sentiments)
    else:
        average_sentiment = 0

    # Create a transition matrix
    transition_matrix = np.zeros((500,500))
    for i in range(len(data) - 2):
        # Get the current state
        current_state = int(data["Close"][i])
        # Get the next state
        next_state = int(data["Close"][i + 1])
        # Update the transition matrix
        transition_matrix[current_state, next_state] += 1

    # Normalize the transition matrix
    for i in range(3):
        row_sum = np.sum(transition_matrix[i])
        if row_sum != 0:
            transition_matrix[i] /= row_sum

    # Create a probability distribution
    probability_distribution = np.zeros(500)
    probability_distribution[int(data["Close"][-2])] = 1

    # Generate a prediction based on historical data and news sentiment
    prediction = np.random.choice(len(probability_distribution), p=probability_distribution) * (1 + average_sentiment)

    return {"prediction": prediction}

if __name__ == '__main__':
    app.run()