import warnings
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import fredapi
import yfinance as yf
import requests
from textblob import TextBlob
from collections import defaultdict
import json

# Function to retrieve news articles related to the stock symbol
def get_news_articles(stock_symbol):
    # Enter your News API key here
    api_key = 'API KEY'
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={api_key}'
    response = requests.get(url)
    articles = json.loads(response.text)['articles']
    
    # Extract the full text of each article
    for article in articles:
        article['full_text'] = article['content']
    
    return articles

# Function to calculate sentiment score of the news articles
def calculate_sentiment_score(articles):
    sentiment_score = 0
    if len(articles) == 0:
        return sentiment_score
    
    # Calculate the sentiment score using TextBlob
    textblob_sentiment_score = 0
    for article in articles:
        text = article['title'] + ' ' + article['description']
        blob = TextBlob(text)
        textblob_sentiment_score += blob.sentiment.polarity
    textblob_sentiment_score /= len(articles)
    
    # Calculate the sentiment score using VADER
    sia = SentimentIntensityAnalyzer()
    vader_sentiment_score = 0
    for article in articles:
        text = article['title'] + ' ' + article['description']
        score = sia.polarity_scores(text)
        vader_sentiment_score += score['compound']
    vader_sentiment_score /= len(articles)
    
    # Combine the results of the two methods
    combined_sentiment_score = ( textblob_sentiment_score + vader_sentiment_score) / 2
    
    return combined_sentiment_score

def get_fred_data(api_key):
    fred = fredapi.Fred(api_key)
    api_key = 'd343bd8ae366e22edc14e5f727c183e2'
    
    # Get interest rates data
    interest_rates = fred.get_series('FEDFUNDS')
    
    # Get inflation rates data
    inflation_rates = fred.get_series('CPIAUCSL')
    
    # Get GDP growth rate data
    gdp_growth_rate = fred.get_series('A191RL1Q225SBEA')
    
    return interest_rates, inflation_rates, gdp_growth_rate

# Get data from FRED API
api_key = 'd343bd8ae366e22edc14e5f727c183e2'
interest_rates, inflation_rates, gdp_growth_rate = get_fred_data(api_key)


# Calculate percentage changes
interest_rates_changes = interest_rates.pct_change().dropna()
inflation_rates_changes = inflation_rates.pct_change().dropna()
gdp_growth_rate_changes = gdp_growth_rate.pct_change().dropna()


warnings.filterwarnings('ignore', category=RuntimeWarning)

# Welcome message
print('######Welcome to the Stock Analysis!######')

# Prompt user to enter stock symbol
stock_symbol = input('Enter the stock symbol: ')

# Use a longer period of historical data
period = 'max'

# Fetch historical stock data using yfinance
stock_data = yf.Ticker(stock_symbol)
hist = stock_data.history(period=period)

# Calculate the percentage change in stock price from one day to the next
price_changes = hist['Close'].pct_change().dropna()

# Discretize the price changes into a finite number of states
num_states = 28
states = np.digitize(price_changes, np.linspace(price_changes.min(), price_changes.max(), num_states))

# Increase the order of the Markov chain
order = 3

# Initialize a dictionary to store the transition counts
transitions = defaultdict(lambda: defaultdict(int))

# Count the number of times each state transition occurs
for i in range(order, len(states)):
    current_state = tuple(states[i - order:i])
    next_state = states[i]
    transitions[current_state][next_state] += 1

# Calculate the transition probabilities
transition_probabilities = defaultdict(dict)
for current_state, next_states in transitions.items():
    total_transitions = sum(next_states.values())
    for next_state, count in next_states.items():
        transition_probabilities[current_state][next_state] = count / total_transitions

# Make predictions using the Markov chain model
current_price = hist['Close'][-order:]
num_predictions = int(input('Enter the number of predictions to generate: '))

# Get news articles related to the stock symbol
articles = get_news_articles(stock_symbol)

# Calculate sentiment score of the news articles using TextBlob
sentiment_score = calculate_sentiment_score(articles)


# Get the trading volume data and calculate its percentage change from one day to the next
volume_data = hist['Volume']
volume_changes = volume_data.pct_change().dropna()

# Discretize the volume changes into a finite number of states and use it as an additional feature in the Markov chain model
volume_states = np.digitize(volume_changes, np.linspace(volume_changes.min(), volume_changes.max(), num_states))

# Print sentiment scores and trading volume data
print(f'Combined sentiment score: {round(sentiment_score, 2)}')
print(f'Trading volume data: {volume_data}')

# Generate predictions using the Markov chain model and incorporating additional features such as trading volume, news sentiment, technical indicators, interest rates, inflation rates, and GDP growth rate into the predictions.
for i in range(num_predictions):
    current_price_pct_change = current_price.pct_change().dropna()[-order:]
    current_volume_pct_change = volume_data.pct_change().dropna()[-order:]
    current_interest_rates_pct_change = interest_rates.pct_change().dropna()[-order:]
    current_inflation_rates_pct_change = inflation_rates.pct_change().dropna()[-order:]
    current_gdp_growth_rate_pct_change = gdp_growth_rate.pct_change().dropna()[-order:]
    
    # Discretize current price, volume, technical indicators, interest rates, inflation rates, and GDP growth rate percentage changes into states.
    current_price_state = tuple(np.digitize(current_price_pct_change, np.linspace(price_changes.min(), price_changes.max(), num_states)))
    current_volume_state = tuple(np.digitize(current_volume_pct_change, np.linspace(volume_changes.min(), volume_changes.max(), num_states)))
    current_interest_rates_state = tuple(np.digitize(current_interest_rates_pct_change, np.linspace(interest_rates_changes.min(), interest_rates_changes.max(), num_states)))
    current_inflation_rates_state = tuple(np.digitize(current_inflation_rates_pct_change, np.linspace(inflation_rates_changes.min(), inflation_rates_changes.max(), num_states)))
    current_gdp_growth_rate_state = tuple(np.digitize(current_gdp_growth_rate_pct_change, np.linspace(gdp_growth_rate_changes.min(), gdp_growth_rate_changes.max(), num_states)))



# Check if the current state is a key in the transition_probabilities dictionary.
if current_state in transition_probabilities:
    # Get the next state with the highest probability.
    next_state = max(transition_probabilities[current_state], key=transition_probabilities[current_state].get)
else:
    # Handle the case where the current state was not observed in the training data.
    # Example: Generate a random prediction.
    next_state = np.random.randint(1, num_states + 1)
    # Get the range of possible percentage changes for the next day's closing price.
    price_range = np.linspace(price_changes.min(), price_changes.max(), num_states)[next_state - 1:next_state + 1]

    # Generate a random percentage change from the range of possible percentage changes.
    next_price_change = np.random.choice(price_range)

    # Calculate the next day's closing price.
    next_price = current_price[-1] * (1 + next_price_change)

    # Update the current price with the next day's closing price.
    current_price = pd.concat([current_price, pd.Series(next_price, index=[current_price.index[-1] + pd.Timedelta(days=1)])])

    # Print the prediction.
    print(f'Prediction {i+1}: {round(next_price, 2)}')