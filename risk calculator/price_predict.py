import yfinance as yf
import numpy as np
from newsapi import NewsApiClient
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

newsapi = NewsApiClient(api_key='API KEY')

analyzer = SentimentIntensityAnalyzer()

stock_symbol = input("Enter a stock symbol: ")

data = yf.download(stock_symbol, period="max")

articles = newsapi.get_everything(q=stock_symbol)

sentiments = []
for article in articles['articles']:

    full_text = article['content']

    sentiment = analyzer.polarity_scores(full_text)['compound']
    sentiments.append(sentiment)

if len(sentiments) > 0:
    average_sentiment = np.mean(sentiments)
else:
    average_sentiment = 0

transition_matrix = np.zeros((500,500))
for i in range(len(data) - 2):
    
    current_state = int(data["Close"][i])

    next_state = int(data["Close"][i + 1])

    transition_matrix[current_state, next_state] += 1


for i in range(3):
    row_sum = np.sum(transition_matrix[i])
    if row_sum != 0:
        transition_matrix[i] /= row_sum


probability_distribution = np.zeros(500)
probability_distribution[int(data["Close"][-2])] = 1

prediction = np.random.choice(len(probability_distribution), p=probability_distribution) * (1 + average_sentiment)

print("The predicted price of the stock is {}.".format(prediction))
