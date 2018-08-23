# Implements On Balance Volume (OBV) momentum indicator - uses volume to predict price changes
# If price increasing and OBV decreasing, sell signal (v.v.)

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
                parse_dates=True, usecols=['Date', 'Adj Close', 'Volume'], na_values=['nan'])
        df = df.join(df_temp)
    df = df.dropna()
    return df

# Begin main
symbol = 'FB'
start = dt.datetime(2017,6,30)
end = dt.datetime(2018,6,30)

dates = pd.date_range(start, end)
stock = get_data([symbol], dates)

obv = [None] * (stock.size/2)
volume = 0
for i in range (1, stock.size/2):
    prev = stock['Adj Close'][i-1]
    curr = stock['Adj Close'][i]
    if curr > prev:
        volume = volume + stock['Volume'][i]
    elif curr < prev:
        volume = volume - stock['Volume'][i]
    obv[i] = volume


stock['OBV'] = obv
stock['OBV7'] = stock['OBV'].rolling(window=7).mean()
stock['OBVNorm'] = stock['OBV'].pct_change()

plt.plot(stock['OBV7'])
plt.figure(2)
plt.plot(stock['Adj Close'])
plt.show()
