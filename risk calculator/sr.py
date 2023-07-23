import yfinance as yf
import numpy as np

# Ask the user to enter a stock symbol
stock_symbol = input("Enter a stock symbol: ")

# Get the historical data for the stock
data = yf.download(stock_symbol, period="max")

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
probability_distribution = np.zeros(100)
probability_distribution[int(data["Close"][-2])] = 1

# Generate a prediction
prediction = np.random.choice(len(probability_distribution), p=probability_distribution)

# Print the prediction
print("The predicted price of the stock is {}.".format(prediction))
