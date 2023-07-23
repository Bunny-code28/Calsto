import yfinance as yf
from prettytable import PrettyTable

currency = input('Enter currency (INR or USD): ')
investment_amount = input(f'Enter investment amount in {currency}: ')
stock_name = input('Enter stock name: ')

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

print(table)
print('The risk value shows the Beta value of the stock, Beta (β) is a measure of how much a stock’s price is expected to change compared to the market as a whole (usually the S&P 500). It is used to help investors understand how risky a stock is compared to other stocks. A stock with a beta higher than 1.0 is considered more volatile than the market, which means its price is expected to change more than the market’s price. A stock with a beta lower than 1.0 is considered less volatile than the market, which means its price is expected to change less than the market’s price. A stock with a beta of 1.0 is considered to have volatility equal to that of the market. In summary, beta is a measure of how much a stock’s price is expected to change compared to the market as a whole, and it is used to help investors understand how risky a stock is compared to other stocks.')
print('*The potential rewards is the dividends or the annual profits of company are given to the investors*')
