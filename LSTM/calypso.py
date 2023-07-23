import yfinance as yf
import requests
import spacy
import numpy as np
from sklearn.linear_model import LinearRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import webbrowser

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set up stopwords
stop_words = set(stopwords.words('english'))

# Get the stock symbol
symbol = ""

# Function to get API key from environment variables
def get_api_key(api_name):
    return os.environ.get(api_name)

# Function to make a request to the FMP API
def make_fmp_request(endpoint, params={}):
    base_url = 'https://financialmodelingprep.com/api/v3'
    api_key = get_api_key('11995037e02d44dd44ec3b07a3cbdd80')
    params['apikey'] = api_key
    response = requests.get(f"{base_url}/{endpoint}", params=params)
    data = response.json()
    return data

# Function to make a request to the Alpha Vantage API
def make_alpha_vantage_request(function, symbol, params={}):
    base_url = 'https://www.alphavantage.co/query'
    api_key = get_api_key('MH1RYBNVOCBBSW4U')
    params['function'] = function
    params['symbol'] = symbol
    params['apikey'] = api_key
    response = requests.get(base_url, params=params)
    data = response.json()
    return data

# Get the historical data using the Alpha Vantage API
def get_historical_data(symbol, start_date, end_date):
    function = 'TIME_SERIES_DAILY'
    params = {
        'outputsize': 'full',
        'datatype': 'json'
    }
    params['start_date'] = start_date
    params['end_date'] = end_date
    data = make_alpha_vantage_request(function, symbol, params=params)
    return data

# Prepare the data
X = np.arange(len(data)).reshape(-1, 1)
y = data['Close'].values

# Train the model
model = LinearRegression()
model.fit(X, y)

# Function to predict future prices
def predict_prices(future_days):
    future_X = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
    predicted_prices = model.predict(future_X)
    return predicted_prices

# Function to process user query using NLP techniques
def process_query(query):
    query = query.lower()
    tokens = word_tokenize(query)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    processed_query = ' '.join(tokens)
    return processed_query

# Function to search the web and extract information
# Load the spaCy NLP model
nlp = spacy.load('en_core_web_sm')

def search_web(query):
    search_results = []
    
    # Perform search using Google API
    google_results = search_google(query)
    search_results.extend(google_results)
    
    # Perform search using DuckDuckGo API
    duckduckgo_results = search_duckduckgo(query)
    search_results.extend(duckduckgo_results)
    
    # Process and extract relevant information from search results
    relevant_results = process_search_results(search_results, query)
    
    return relevant_results

def search_google(query):
    # Implement code to perform search using Google API
    # Make a request to the Google API with the query
    # Extract and return relevant information from the search results
    # Example implementation using requests library
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key=AIzaSyB4mzXW5RDCOI7uwGRnA9MUt8w9fjJmwA0"
    response = requests.get(url)
    search_results = response.json()['items']
    return search_results

def search_duckduckgo(query):
    # Implement code to perform search using DuckDuckGo API
    # Make a request to the DuckDuckGo API with the query
    # Extract and return relevant information from the search results
    # Example implementation using requests library
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    search_results = response.json()['Results']
    return search_results

def process_search_results(search_results, query):
    relevant_results = []
    
    # Rank and score the search results based on relevance
    ranked_results = rank_search_results(search_results, query)
    
    for result in ranked_results:
        # Process each search result and extract relevant information
        title = result['title']
        link = result['link']
        description = result['description']
        
        # Apply NLP techniques to analyze the search result
        analyzed_result = analyze_search_result(title, description)
        
        # Add the analyzed result to the list of relevant results
        relevant_results.append({
            'title': title,
            'link': link,
            'description': description,
            'entities': analyzed_result['entities'],
            'sentiment': analyzed_result['sentiment']
        })
    
    return relevant_results

def rank_search_results(search_results, query):
    ranked_results = []
    
    # Calculate relevance scores for each search result
    for result in search_results:
        title = result['title']
        description = result['description']
        
        # Calculate relevance score using keyword presence and proximity
        score = calculate_relevance_score(title, description, query)
        
        # Assign the score to the search result
        result['score'] = score
        
        # Add the search result to the list of ranked results
        ranked_results.append(result)
    
    # Sort the search results based on the relevance score in descending order
    ranked_results = sorted(ranked_results, key=lambda x: x['score'], reverse=True)
    
    return ranked_results

def analyze_search_result(title, description):
    # Apply NLP techniques to analyze the search result
    
    # Perform Named Entity Recognition (NER) to identify entities
    analyzed_entities = extract_entities(title, description)
    
    # Perform sentiment analysis to gauge the sentiment of the information
    analyzed_sentiment = analyze_sentiment(title, description)
    
    return {
        'entities': analyzed_entities,
        'sentiment': analyzed_sentiment
    }

def extract_entities(title, description):
    # Extract entities using spaCy's Named Entity Recognition (NER)
    doc = nlp(title + ' ' + description)
    
    entities = []
    for entity in doc.ents:
        entities.append({
            'text': entity.text,
            'label': entity.label_
        })
    
    return entities

def analyze_sentiment(title, description):
    # Implement sentiment analysis logic to analyze the sentiment of the information
    # You can use various techniques like rule-based methods, machine learning models, etc.
    # Return the sentiment score or label
    
    # Example implementation using a rule-based method
    sentiment_score = 0
    sentiment_label = 'Neutral'
    
    # Analyze the sentiment of the title and description
    # Update the sentiment score and label accordingly
    
    return {
        'score': sentiment_score,
        'label': sentiment_label
    }

def calculate_relevance_score(title, description, query):
    # Implement relevance scoring logic to calculate the relevance score of the search result
    # You can consider factors like keyword presence, proximity, query term frequency, etc.
    # Return the relevance score
    
    # Example implementation using a simple keyword presence approach
    title_score = calculate_keyword_presence_score(title, query)
    description_score = calculate_keyword_presence_score(description, query)
    
    # Calculate the overall relevance score based on the individual scores
    relevance_score = (title_score + description_score) / 2
    
    return relevance_score

def calculate_keyword_presence_score(text, query):
    # Calculate the score based on the presence of query-related keywords in the text
    # You can consider different scoring strategies like exact match, fuzzy match, etc.
    # Return the keyword presence score
    
    # Example implementation using an exact match approach
    if query.lower() in text.lower():
        keyword_presence_score = 1.0
    else:
        keyword_presence_score = 0.0
    
    return keyword_presence_score

# Function to scrape relevant information from a web page
# Function to scrape relevant information from a web page
def scrape_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Example scraping logic:
    # Extract the title of the web page
    title = soup.title.string

    # Extract all paragraph elements from the web page
    paragraphs = soup.find_all('p')
    text_content = [paragraph.get_text() for paragraph in paragraphs]

    # Extract all image URLs from the web page
    images = soup.find_all('img')
    image_urls = [image['src'] for image in images]

    # Extract relevant information from the articles
    # ...

    # Create a dictionary to store the scraped information
    scraped_info = {
        'title': title,
        'text_content': text_content,
        'image_urls': image_urls,
        # Add more relevant information as needed
    }

    return scraped_info

# Function to scrape financial news
def scrape_financial_news():
    # Define the list of financial news outlets
    news_outlets = ['Bloomberg', 'CNBC', 'Financial Times']

    # Dictionary to store the scraped news articles
    news_articles = {}

    for outlet in news_outlets:
        # Scrape news articles from the website
        if outlet == 'Bloomberg':
            # Scrape from Bloomberg website
            # Implement your scraping logic here
            # Example:
            response = requests.get('https://www.bloomberg.com/')
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article')

            # Extract relevant information from the articles
            news_articles[outlet] = [article.get_text() for article in articles]

        elif outlet == 'CNBC':
            # Scrape from CNBC website
            # Implement your scraping logic here
            # Example:
            response = requests.get('https://www.cnbc.com/')
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article')

            # Extract relevant information from the articles
            news_articles[outlet] = [article.get_text() for article in articles]

        elif outlet == 'Financial Times':
            # Scrape from Financial Times website
            # Implement your scraping logic here
            # Example:
            response = requests.get('https://www.ft.com/')
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article')

            # Extract relevant information from the articles
            news_articles[outlet] = [article.get_text() for article in articles]

    return news_articles

# Function to provide personalized financial advice
def get_personalized_advice(income, expenses, savings, debt):
    # Add logic to generate personalized financial advice based on the input
    # ...
    return advice

# Financial planning module
def financial_planning(query):
    print("Welcome to the financial planning module!")

    # Process the query
    processed_query = process_query(query)

    # Search the web for the processed query
    search_results = search_web(processed_query)

    if search_results:
        # Scrape relevant information from the first search result
        relevant_info = scrape_web_page(search_results[0])
        print("Here is some relevant information:")
        print(relevant_info)

        # Offer additional options for the user
        print("\nWhat would you like to do next?")
        print("1. View more search results")
        print("2. Get personalized financial advice")
        print("3. Exit the financial planning module")

        while True:
            option = input("Enter the option number: ")
            if option == "1":
                print("More search results:")
                for i, result in enumerate(search_results, start=1):
                    print(f"{i}. {result}")
                break
            elif option == "2":
                income = float(input("Please enter your annual income: "))
                expenses = float(input("Please enter your monthly expenses: "))
                savings = float(input("Please enter your total savings: "))
                debt = float(input("Please enter your total debt: "))
                advice = get_personalized_advice(income, expenses, savings, debt)
                print("Here is your personalized financial advice:")
                print(advice)
                break
            elif option == "3":
                print("Exiting the financial planning module.")
                return
            else:
                print("Invalid option. Please choose a valid option.")
    else:
        print("Sorry, I couldn't find any information related to your query.")

# ...

# Main loop
while True:
    # Get the user's input
    user_input = input("What would you like to know? (Type 'finished' to exit) ")

    # ...

    elif processed_input.startswith("financial planning"):
        financial_planning(processed_input)

    # ...

# Function to perform a web search using DuckDuckGo API
def search_web(query, num_results=5):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    data = response.json()
    search_results = data['Results'][:num_results]
    result_urls = [result['FirstURL'] for result in search_results]
    return result_urls

# Debt management module
def debt_management():
    print("Welcome to the debt management module!")
    
    # Get user input
    current_debt = float(input("Enter your current debt amount: "))
    monthly_income = float(input("Enter your monthly income: "))
    monthly_expenses = float(input("Enter your monthly expenses: "))

    # Calculate the debt-to-income ratio
    debt_to_income_ratio = current_debt / monthly_income

    # Calculate the savings ratio
    savings_ratio = (monthly_income - monthly_expenses) / monthly_income

    # Assess the debt level and recommend a strategy based on the ratios
    if debt_to_income_ratio < 0.3 and savings_ratio > 0.2:
        strategy = "You are managing your debt well and have good savings."
    elif debt_to_income_ratio >= 0.3 and debt_to_income_ratio < 0.5 and savings_ratio > 0.1:
        strategy = "Your debt level is moderate. Focus on reducing debt while maintaining savings."
    elif debt_to_income_ratio >= 0.5 and debt_to_income_ratio < 0.8 and savings_ratio > 0.05:
        strategy = "Your debt level is high. Prioritize debt repayment and limit unnecessary expenses."
    elif debt_to_income_ratio >= 0.8 or savings_ratio < 0.05:
        strategy = "Your debt level is critical. Seek professional assistance to develop a debt repayment plan."

    # Choose the debt repayment method
    print("Choose a debt repayment method:")
    print("1. Snowball Method: Pay off debts from smallest balance to largest balance.")
    print("2. Avalanche Method: Pay off debts from highest interest rate to lowest interest rate.")
    print("3. Blizzard Method: Pay off debts from smallest balance to largest balance, except for high-interest debts.")

    while True:
        method_choice = input("Enter the option number to choose the method: ")
        if method_choice == "1":
            method_name = "Snowball Method"
            method_explanation = "With the Snowball Method, you prioritize paying off debts from the smallest balance to the largest balance. You make minimum payments on all debts and allocate any extra funds towards the debt with the smallest balance. Once the smallest debt is paid off, you move to the debt with the next smallest balance. This method provides psychological motivation as you see debts getting eliminated one by one."
            break
        elif method_choice == "2":
            method_name = "Avalanche Method"
            method_explanation = "With the Avalanche Method, you prioritize paying off debts from the highest interest rate to the lowest interest rate. You make minimum payments on all debts and allocate any extra funds towards the debt with the highest interest rate. Once the highest-interest debt is paid off, you move to the debt with the next highest interest rate. This method saves you more money on interest payments in the long run."
            break
        elif method_choice == "3":
            method_name = "Blizzard Method"
            method_explanation = "The Blizzard Method combines elements of the Snowball and Avalanche methods. You start by listing your debts in order of their balances, from smallest to largest. You make minimum payments on all debts and allocate any extra funds towards the debt with the smallest balance. However, if a debt has an interest rate higher than a certain threshold, you prioritize paying it off before moving to the next smallest balance. This method provides a balance between motivation and savings on interest payments."
            break
        else:
            print("Invalid option. Please choose a valid option.")

    strategy += f"\nDebt Repayment Method: {method_name}\n"
    strategy += f"Method Explanation: {method_explanation}\n"

    # Apply the chosen debt repayment method
    if current_debt > 0:
        # List of debts with their balances and interest rates
        debts = [
            {"name": "Credit Card A", "balance": 5000, "interest_rate": 0.18},
            {"name": "Credit Card B", "balance": 3000, "interest_rate": 0.12},
            {"name": "Student Loan", "balance": 10000, "interest_rate": 0.06},
            {"name": "Car Loan", "balance": 15000, "interest_rate": 0.08},
        ]

        if method_choice == "1":
            # Sort debts by balance in ascending order
            debts.sort(key=lambda x: x["balance"])
        elif method_choice == "2":
            # Sort debts by interest rate in descending order
            debts.sort(key=lambda x: x["interest_rate"], reverse=True)

        # Determine the minimum payment based on the chosen method
        minimum_payment = monthly_income * 0.1

        # Allocate extra funds towards the debt according to the chosen method
        extra_payment = monthly_income * 0.1

        # Calculate the total monthly debt payment
        total_payment = minimum_payment + extra_payment

        # Create a plan for debt repayment
        plan = "Debt Repayment Plan:\n"
        for debt in debts:
            plan += f"- {debt['name']}: Pay ${minimum_payment:.2f} (Minimum Payment) + ${extra_payment:.2f} (Extra Payment)\n"
            debt["balance"] -= total_payment
            if debt["balance"] <= 0:
                plan += f"  --> {debt['name']} paid off!\n"

        strategy += plan

    # Perform a web search to find additional debt management strategies
    query = f"debt management strategies for {current_debt} at {method_name}"
    search_results = search_web(query)

    if search_results:
        strategy += "\nHere are some additional debt management strategies you can consider:\n"
        for i, url in enumerate(search_results, start=1):
            strategy += f"{i}. {url}\n"
    else:
        strategy += "\nSorry, I couldn't find any specific additional debt management strategies for your query."

    return strategy


# Example usage
print(debt_management())
# ...

# Main loop
while True:
    # Get the user's input
    user_input = input("What would you like to know? (Type 'finished' to exit) ")

    # ...

    elif processed_input.startswith("debt management"):
        debt_management()

    # ...


def budgeting():
    print("Welcome to the budgeting module!")

    # Get user input for income, expenses, and savings
    income = float(input("Enter your monthly income: "))
    expenses = float(input("Enter your monthly expenses: "))
    savings = float(input("Enter your desired monthly savings: "))

    # Calculate the available budget
    available_budget = income - expenses

    if available_budget >= savings:
        # Sufficient budget for savings
        print("Congratulations! You have enough budget for your desired savings.")
    else:
        # Insufficient budget for savings
        deficit = savings - available_budget
        print("You have a budget deficit of $%.2f for your desired savings." % deficit)

        # Calculate the minimum expenses to cover
        minimum_expenses = income - savings

        if minimum_expenses <= 0:
            # Insufficient income to cover savings
            print("Your income is not sufficient to cover your desired savings.")
        else:
            # Calculate the remaining budget after covering minimum expenses
            remaining_budget = income - minimum_expenses

            if remaining_budget > 0:
                # Extra budget available after covering minimum expenses
                print("You can allocate $%.2f from the remaining budget for discretionary spending." % remaining_budget)
            else:
                # No extra budget available after covering minimum expenses
                print("You have no extra budget for discretionary spending.")

def investment_strategies():
    print("Welcome to the investment strategies module!")

    # Get user's investment preferences
    preferences = input("Please enter the investment markets you are willing to invest in (e.g., stocks, bonds, real estate): ")
    preferences = preferences.lower().split(", ")

    # Define the investment strategies based on user preferences
    strategies = []
    if "stocks" in preferences:
        strategies.append("Consider investing in a diversified portfolio of stocks from different sectors and regions.")
    if "bonds" in preferences:
        strategies.append("Allocate a portion of your portfolio to high-quality bonds to balance risk and generate income.")
    if "real estate" in preferences:
        strategies.append("Explore real estate investment options such as REITs or rental properties for potential long-term growth.")

    # Print the investment strategy suggestions
    if strategies:
        print("Here are three investment strategy suggestions based on your preferences:")
        for i, strategy in enumerate(strategies, start=1):
            print(f"{i}. {strategy}")
    else:
        print("Based on the broad investment markets, here are some investment opportunities available to you:")
        print("1. Stocks: Invest in individual stocks or exchange-traded funds (ETFs) to participate in the growth potential of different companies.")
        print("2. Bonds: Consider investing in government bonds or corporate bonds to generate fixed income and preserve capital.")
        print("3. Real Estate: Explore real estate investment trusts (REITs), crowdfunding platforms, or rental properties to benefit from the real estate market.")
        print("4. Mutual Funds or Index Funds: Invest in professionally managed funds that provide diversification across various assets.")

    # Provide learning resources for investment education
    print("\nTo learn more about investment strategies and how to get started, you can explore the following resources:")
    print("1. YouTube: Search for investment tutorial videos on YouTube.")
    print("2. Google: Search for beginner's guides to investing on Google.")

    while True:
        option = input("Enter the option number to open the corresponding resource, or type 'done' to exit: ")
        if option == "1":
            webbrowser.open("https://www.youtube.com/results?search_query=investment+tutorials")
        elif option == "2":
            webbrowser.open("https://www.google.com/search?q=beginner%27s+guide+to+investing")
        elif option.lower() == "done":
            break
        else:
            print("Invalid option. Please choose a valid option.")

def calculate_investments(investment_type, amount, time_horizon, months=0, days=0, years=0):
    total_days = years * 365 + months * 30 + days
    
    if investment_type == 'stocks':
        # Stock investment
        stock_symbol = input("Enter the stock symbol: ")
        stock_data = yf.Ticker(stock_symbol)
        stock_price = stock_data.history().tail(1)['Close'].values[0]
        shares = amount / stock_price
        
        return {stock_symbol: shares * (total_days / 365)}
    
    elif investment_type == 'bonds':
        # Bond investment
        bond_symbol = input("Enter the bond symbol: ")
        bond_data = yf.Ticker(bond_symbol)
        bond_price = bond_data.history().tail(1)['Close'].values[0]
        bonds = amount / bond_price
        
        return {bond_symbol: bonds * (total_days / 365)}
    
    elif investment_type == 'funds':
        # Mutual fund investment
        fund_symbol = input("Enter the fund symbol: ")
        fund_data = yf.Ticker(fund_symbol)
        fund_price = fund_data.history().tail(1)['Close'].values[0]
        units = amount / fund_price
        
        return {fund_symbol: units * (total_days / 365)}
    
    elif investment_type == 'etf':
        # ETF investment
        etf_symbol = input("Enter the ETF symbol: ")
        etf_data = yf.Ticker(etf_symbol)
        etf_price = etf_data.history().tail(1)['Close'].values[0]
        shares = amount / etf_price
        
        return {etf_symbol: shares * (total_days / 365)}
    
    else:
        return None  # Handle invalid investment types

# Example usage
investment_type = input("Enter the investment type (stocks, bonds, funds, etf, strategies): ")
if investment_type.lower() == "strategies":
    investment_strategies()
else:
    amount = float(input("Enter the investment amount: "))
    time_horizon = int(input("Enter the time horizon (in months): "))
    months = int(input("Enter the additional months: "))
    days = int(input("Enter the additional days: "))
    years = int(input("Enter the additional years: "))

    investments = calculate_investments(investment_type, amount, time_horizon, months, days, years)
    print(investments)

# Say hello
print("Hello, my name is Calypso. How can I help you?")

# Keep running until user indicates query is finished
while True:
    # Get the user's input
    user_input = input("What would you like to know? (Type 'finished' to exit) ")

    # Check if user wants to exit
    if user_input.lower() == "finished":
        print("Thank you for using Calypso. Goodbye!")
        break

    # Process user query
    processed_input = process_query(user_input)

    # Get the data
    if processed_input.startswith("price"):
        # Extract the stock symbol from user input
        symbol = processed_input.split()[1]

        # Fetch the current price
        current_price = yf.Ticker(symbol).history(period='1d').iloc[-1]['Close']
        print("The current price of " + symbol + " is " + str(current_price))
    elif processed_input.startswith("predict"):
        # Extract the number of future days from user input
        future_days = int(processed_input.split()[1])

        # Predict future prices
        predicted_prices = predict_prices(future_days)
        print("The predicted prices for the next " + str(future_days) + " days are:")
        for i, price in enumerate(predicted_prices, start=1):
            print("Day " + str(i) + ": " + str(price))
    elif processed_input.startswith("financial planning"):
        financial_planning()
    elif processed_input.startswith("debt management"):
        debt_management()
    elif processed_input.startswith("budgeting"):
        budgeting()
    elif processed_input.startswith("investment strategies"):
        investment_strategies()
    else:
        # Search the web for user's query
        search_results = search_web(processed_input)
        if search_results:
            # Scrape relevant information from the first search result
            relevant_info = scrape_web_page(search_results[0])
            print("Here is some relevant information:")
            print(relevant_info)
        else:
            print("I don't understand your request.")

# Function to scrape financial news articles
def scrape_financial_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Example scraping logic:
    # Find all news article elements
    articles = soup.find_all('article')

    # Extract relevant information from each article
    news_articles = []
    for article in articles:
        # Extract the title of the article
        title = article.find('h2').text.strip()

        # Extract the publication date of the article
        date = article.find('time').text.strip()

        # Extract the summary of the article
        summary = article.find('p').text.strip()

        # Extract the link to the full article
        link = article.find('a')['href']

        # Create a dictionary to store the article information
        news_article = {
            'title': title,
            'date': date,
            'summary': summary,
            'link': link
        }

        news_articles.append(news_article)

    return news_articles

# Scrape financial news
news_articles = scrape_financial_news('https://example.com/financial-news')

# Display the scraped news articles
print("Scraped Financial News Articles:")
for article in news_articles:
    print(f"\nTitle: {article['title']}")
    print(f"Date: {article['date']}")
    print(f"Summary: {article['summary']}")
    print(f"Link: {article['link']}")


# Ask if the user has any other queries
user_input = input("\nDo you have any other queries? (yes/no): ")
if user_input.lower() == "no":
    print("Thank you for using Calypso. Have a great day!")
