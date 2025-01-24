from flask import Flask, request
import yfinance as yf
import numpy as np
from prettytable import PrettyTable

app = Flask(__name__)

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    currency = data['currency']
    investment_amount = float(data['investment_amount'])
    stock_name = data['stock_name']
    num_simulations = int(data['num_simulations'])

    stock_data = yf.Ticker(stock_name)
    info = stock_data.info
    hist = stock_data.history(period="max")
    returns = hist['Close'].pct_change().dropna()

    mean_return = returns.mean()
    std_return = returns.std()

    simulated_investment_value = np.zeros(num_simulations)

    for i in range(num_simulations):
        simulated_return = np.random.normal(mean_return, std_return)
        simulated_investment_value[i] = investment_amount * (1 + simulated_return)

        risk = np.percentile(simulated_investment_value[:i+1], 1)
        potential_rewards = np.percentile(simulated_investment_value[:i+1], 99)
        volatility = simulated_investment_value[:i+1].std()

        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["Simulation", i+1])
        table.add_row(["Risk", f"{risk:.2f}"])
        table.add_row(["Potential Rewards", f"{potential_rewards:.2f}"])
        table.add_row(["Volatility", f"{volatility:.2f}"])
        table.add_row(["Stock Name", info["longName"]])
        table.add_row(["Market Value", info["marketCap"]])
        table.add_row(["Shares Outstanding", info["sharesOutstanding"]])

    return str(table)

if __name__ == '__main__':
    app.run()
