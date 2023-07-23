
from flask import Flask, request
import yfinance as yf
from prettytable import PrettyTable

app = Flask(__name__)

@app.route('/stock', methods=['POST'])
def stock():
    data = request.get_json()
    currency = data['currency']
    investment_amount = data['investment_amount']
    stock_name = data['stock_name']

    stock_data = yf.Ticker(stock_name)
    info = stock_data.info

    if 'beta' in info:
        risk = int(info['beta'] * 100)
    else:
        print('Beta value not available for this stock')
        risk = 'N/A'

    if 'trailingAnnualDividendYield' in info:
        potential_rewards = int(info['trailingAnnualDividendYield'] * 100)
    else:
        print('Trailing Annual Dividend Yield not available for this stock')
        potential_rewards = 'N/A'

    hist = stock_data.history(period="max")
    volatility = int(hist['Close'].pct_change().std() * 100)

    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Risk", f"{risk}%"])
    table.add_row(["Potential Rewards", f"{potential_rewards}%"])
    table.add_row(["Volatility", f"{volatility}%"])
    table.add_row(["Stock Name", info["longName"]])
    table.add_row(["Market Value", info["marketCap"]])
    table.add_row(["Shares Outstanding", info["sharesOutstanding"]])

    return str(table)

if __name__ == '__main__':
    app.run()
