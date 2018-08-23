# Uses Bollinger Band strategy to anticipate price movements

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import datetime as dt

# Return CSV file path given ticker symbol
def symbol_to_path(symbol, base_dir="Data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

# Read stock data for given symbols from CSV files
def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df = df.join(df_temp)
    df = df.dropna()
    return df

# Return rolling mean of given values using specified window size
def get_rolling_mean(values, window):
    return values.rolling(window).mean()

# Return rolling standard deviation of given values using specified window size
def get_rolling_std(values, window):
    return values.rolling(window).std()

# Return upper and lower Bollinger bands
def get_bollinger_bands(rmean, rstdev):
    upper_band = rmean + rstdev * 2
    lower_band = rmean - rstdev * 2
    return upper_band, lower_band

# Begin main
symbol = 'KO'
start = dt.datetime(2017,6,30)
end = dt.datetime(2018,6,30)

dates = pd.date_range(start, end)
stock = get_data([symbol], dates)

rm = get_rolling_mean(stock['Adj Close'], window=20) # Compute rolling mean
rstd = get_rolling_std(stock['Adj Close'], window=20) # Compute rolling standard deviation
upper_band, lower_band = get_bollinger_bands(rm, rstd) # Compute upper and lower bands
ax = stock['Adj Close'].plot(title="Bollinger Bands", label=symbol)
rm.plot(label='Rolling mean', ax=ax)
upper_band.plot(label='upper band', ax=ax)
lower_band.plot(label='lower band', ax=ax)

print "Market Return: ", stock['Adj Close'][-1] - stock['Adj Close'][0], "\n"

bought = False
below_band = False
strategy_ret = 0.0
begin_price = 0.0

for i in range(19, stock.shape[0]):
    if bought == False:
        if below_band == False and stock['Adj Close'][i] < lower_band[i]:
            below_band = True
        elif below_band == True and stock['Adj Close'][i] > lower_band[i]:
            below_band = False
            bought = True
            begin_price = stock['Adj Close'][i]
            print "BUY: ", stock['Adj Close'][i], " on ", stock.index[i]
    else:
        if stock['Adj Close'][i] > rm[i]:
            strategy_ret = strategy_ret + stock['Adj Close'][i] - begin_price
            bought = False
            print "SELL: ", stock['Adj Close'][i], " on ", stock.index[i]
print "Strategy 1 (Sell at Moving Average) Return: ", strategy_ret, "\n"

bought = False
below_band = False
above_band = False
strategy_ret = 0.0
begin_price = 0.0

for i in range(19, stock.shape[0]):
    if bought == False:
        if below_band == False and stock['Adj Close'][i] < lower_band[i]:
            below_band = True
        elif below_band == True and stock['Adj Close'][i] > lower_band[i]:
            below_band = False
            bought = True
            begin_price = stock['Adj Close'][i]
            print "BUY: ", stock['Adj Close'][i], " on ", stock.index[i]
    else:
        if above_band == False and stock['Adj Close'][i] > upper_band[i]:
            above_band = True
        elif above_band == True and stock['Adj Close'][i] < upper_band[i]:
            strategy_ret = strategy_ret + stock['Adj Close'][i] - begin_price
            bought = False
            print "SELL: ", stock['Adj Close'][i], " on ", stock.index[i]
print "Strategy 2 (Sell above Upper Band) Return: ", strategy_ret, "\n"


# Add axis labels and legend
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend(loc='upper left')

plt.show()
