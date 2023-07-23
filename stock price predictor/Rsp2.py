from flask import Flask, request, jsonify
import numpy as np
import yfinance as yf
from collections import defaultdict

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    data = request.get_json()
    
    currency = data['currency']
    stock_symbol = data['stock_symbol']
    investment_amount = float(data['investment_amount'])
    start_date = data['start_date']
    end_date = data['end_date']
    num_predictions = int(data['num_predictions'])
    
    # Fetch historical stock data using yfinance
    stock_data = yf.Ticker(stock_symbol)
    hist = stock_data.history(start=start_date, end=end_date)

    # Convert stock data to the specified currency if necessary
    if currency == 'INR':
        conversion_rate = 81.69  # Example conversion rate from USD to INR
        hist['Open'] *= conversion_rate
        hist['High'] *= conversion_rate
        hist['Low'] *= conversion_rate
        hist['Close'] *= conversion_rate

    # Calculate the percentage change in stock price from one day to the next
    price_changes = hist['Close'].pct_change().dropna()

    # Discretize the price changes into a finite number of states
    num_states = 10
    states = np.digitize(price_changes, np.linspace(price_changes.min(), price_changes.max(), num_states))

    # Initialize a dictionary to store the transition counts
    transitions = defaultdict(lambda: defaultdict(int))

    # Count the number of times each state transition occurs
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transitions[current_state][next_state] += 1

    # Calculate the transition probabilities
    transition_probabilities = defaultdict(dict)
    for current_state, next_states in transitions.items():
        total_transitions = sum(next_states.values())
        for next_state, count in next_states.items():
            transition_probabilities[current_state][next_state] = count / total_transitions

    def predict_price(current_price, transition_probabilities, num_predictions):
        # Find the current state
        current_state = np.digitize(current_price, np.linspace(price_changes.min(), price_changes.max(), num_states))

        # Generate predictions
        predictions = []
        for _ in range(num_predictions):
            # Sample the next state based on the transition probabilities
            next_state = np.random.choice(list(transition_probabilities[current_state].keys()), p=list(transition_probabilities[current_state].values()))

            # Convert the state back to a price change
            price_change = np.interp(next_state, range(num_states), np.linspace(price_changes.min(), price_changes.max(), num_states))

            # Calculate the predicted price
            predicted_price = current_price * (1 + price_change)
            predictions.append(predicted_price)

            # Update the current price and state
            current_price = predicted_price
            current_state = next_state

        return predictions

    # Make predictions using the Markov chain model
    current_price = hist['Close'][-1]
    predictions = predict_price(current_price, transition_probabilities, num_predictions)

    # Calculate the standard deviation of the historical stock price changes
    price_changes_std = price_changes.std()

    # Prepare the response data
    response = {
        'estimated_risk': round(price_changes_std, 4),
        'predicted_prices': [round(price, 2) for price in predictions]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

