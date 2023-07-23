import matplotlib.pyplot as plt

def plot_graphs(data, title):
    plt.plot(data)
    plt.title(title)
    plt.show()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get the stock symbol from the request body
    stock_symbol = data['stock_symbol']

    # Get the historical data for the stock
    data = yf.download(stock_symbol)

    # Calculate the moving averages
    short_ma = data["Close"].rolling(window=20).mean()
    long_ma = data["Close"].rolling(window=50).mean()

    # Calculate the MACD
    macd = MACD(data["Close"])

    # Calculate the RSI
    rsi = RSIIndicator(data["Close"], window=14)

    # Access the RSI values
    rsi_values = rsi.rsi()

    # Calculate the volume
    volume = data["Volume"]

    # Calculate the volatility
    volatility = data["Close"].std()

    # Calculate the trend
    trend = np.sign(data["Close"].diff().dropna())

    # Access the MACD components
    macd_line = macd.macd()  # MACD line

    signal_line = macd.macd_signal()  # Signal line

    histogram = macd.macd_diff()  # MACD histogram

    # Make a prediction
    if macd_line[-1] > signal_line[-1] and rsi_values[-1] < 70 and volume[-1] > volume.mean() and volatility < data["Close"].std() and trend[-1] == 1:
        prediction = "The stock is likely to go up."
    elif macd_line[-1] < signal_line[-1] and rsi_values[-1] > 30 and volume[-1] < volume.mean() and volatility > data["Close"].std() and trend[-1] == -1:
        prediction = "The stock is likely to go down."
    else:
        prediction = "The stock is likely to stay flat."

    # Prepare the response data
    response = {
        'prediction': prediction
    }

    # Plot the graphs
    if data['show_graphs'] == True:
        plot_graphs(data["Close"], "Historical Data")
        plot_graphs(data["Close"].tail(5), "Predicted Data")

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
