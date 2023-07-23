import yfinance as yf
import numpy as np
from newsapi import NewsApiClient
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the News API client
newsapi = NewsApiClient(api_key='API KEY')

# Initialize the sentiment analyzer
analyzer = TextBlob

# Ask the user to enter a stock symbol
stock_symbol = input("Enter a stock symbol: ")

# Get the historical data for the stock
data = yf.download(stock_symbol, period="max")

# Get news articles for the stock
articles = newsapi.get_everything(q=stock_symbol)

# Get the historical data for the stock
stock_data = yf.download(stock_symbol, period="max")

# Ask the user to enter a cryptocurrency symbol
crypto_symbol = input("Enter a cryptocurrency symbol: ")

# Get the historical data for the cryptocurrency
crypto_data = yf.download(crypto_symbol, period="max")

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

# Print the prediction
print("The predicted price of the stock is {}.".format(prediction))
