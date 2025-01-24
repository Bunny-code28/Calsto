from flask import Flask, request, jsonify
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator

app = Flask(__name__)

@app.route('/analyzee', methods=['POST'])
def analyzee_stock():
    data = request.get_json()

    symbol = data['symbol']

    # Get the historical data
    data = yf.download(symbol)

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
        'moving_averages': {
            'short_ma': short_ma.tolist(),
            'long_ma': long_ma.tolist()
        },
        'macd': {
            'macd': macd_line.tolist(),
            'signal': signal_line.tolist(),
            'histogram': histogram.tolist()
        },
        'rsi': rsi_values.tolist(),
        'volume': volume.tolist(),
        'volatility': volatility,
        'trend': trend.tolist(),
        'prediction': prediction
    }

    # Plot the moving averages
    plt.plot(short_ma, label="Short MA")
    plt.plot(long_ma, label="Long MA")
    plt.legend()
    plt.show()

    # Plot the MACD
    plt.plot(macd_line, label="MACD")
    plt.plot(signal_line, label="Signal line")
    plt.plot(histogram, label="Histogram")
    plt.legend()
    plt.show()

    # Plot the RSI
    plt.plot(rsi_values, label="RSI")
    plt.axhline(70, color="red", label="Overbought")
    plt.axhline(30, color="green", label="Oversold")
    plt.legend()
    plt.show()

    # Plot the volume
    plt.plot(volume, label="Volume")
    plt.show()

    # Plot the volatility
    plt.plot(volatility, label="Volatility")
    plt.show()

    # Plot the trend
    plt.plot(trend, label="Trend")
    plt.show()

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)




