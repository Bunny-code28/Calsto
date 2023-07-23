import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def MACD(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def RSI(data, period=14):
    delta = data.diff().dropna()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.ewm(com=period-1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Get the stock symbol
symbol = input("Enter the stock symbol: ")

# Get the historical data
data = yf.download(symbol)

# Plot the data
plt.plot(data["Close"])
plt.show()

# Calculate the moving averages
short_ma = data["Close"].rolling(window=20).mean()
long_ma = data["Close"].rolling(window=50).mean()

# Plot the moving averages
plt.plot(short_ma, label="Short MA")
plt.plot(long_ma, label="Long MA")
plt.legend()
plt.show()

# Calculate the MACD
macd, signal, histogram = MACD(data["Close"])

# Plot the MACD
plt.plot(macd, label="MACD")
plt.plot(signal, label="Signal")
plt.plot(histogram, label="Histogram")
plt.legend()
plt.show()

# Calculate the RSI
rsi = RSI(data["Close"])

# Plot the RSI
plt.plot(rsi, label="RSI")
plt.axhline(70, label="Overbought")
plt.axhline(30, label="Oversold")
plt.legend()
plt.show()

# Calculate the volume
volume = data["Volume"]

# Plot the volume
plt.plot(volume, label="Volume")
plt.legend()
plt.show()

# Calculate the volatility
volatility = data["Close"].std()

# Print the volatility
print("The volatility is: ", volatility)

# Calculate the trend
trend = np.sign(data["Close"].diff().dropna())

# Print the trend
print("The trend is: ", trend)

# Make a prediction
if macd[-1] > signal[-1] and rsi[-1] < 70 and volume[-1] > volume.mean() and volatility < data["Close"].std() and trend[-1] == 1:
    print("The stock is likely to go up.")
elif macd[-1] < signal[-1] and rsi[-1] > 30 and volume[-1] < volume.mean() and volatility > data["Close"].std() and trend[-1] == -1:
    print("The stock is likely to go down.")
else:
    print("The stock is likely to stay flat.")

# Predict the future price
future_price = data["Close"].iloc[-1] + (data["Close"].iloc[-1] - data["Close"].iloc[-2]) * 1.1
print("The predicted future price is: ", future_price)

# Plot the predicted graph
plt.plot(data["Close"])
plt.plot([data["Close"].iloc[-1], future_price], label="Predicted Price")
plt.legend()
plt.show()
