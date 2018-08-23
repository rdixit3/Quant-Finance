# Implements Stochastic Oscillator - momentum indicator
# %K = 100(C - L)/(H - L)
# > 80 indicates trading near top of its high-low range, < 20 indicates is trading near the bottom of high-low range
# When increasing, market momentum is increasing (v.v.)

# *Need to implement crossover signal

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
                parse_dates=True, usecols=['Date', 'Low', 'High', 'Adj Close'], na_values=['nan'])
        df = df.join(df_temp)
    df = df.dropna()
    return df

# Begin main
symbol = 'GOOG'
start = dt.datetime(2017,6,30)
end = dt.datetime(2018,6,30)
trading_days = 252

dates = pd.date_range(start, end)
stock = get_data([symbol], dates)

stock['L14'] = stock['Low'].rolling(window=14).min()
stock['H14'] = stock['High'].rolling(window=14).max()
stock['%K14'] = 100*((stock['Adj Close'] - stock['L14']) / (stock['H14'] - stock['L14']) )
stock['%D14'] = stock['%K14'].rolling(window=3).mean()

stock['L5'] = stock['Low'].rolling(window=5).min()
stock['H5'] = stock['High'].rolling(window=5).max()
stock['%K5'] = 100*((stock['Adj Close'] - stock['L5']) / (stock['H5'] - stock['L5']) )
stock['%D5'] = stock['%K5'].rolling(window=3).mean()

print "Market Return: ", stock['Adj Close'][-1] - stock['Adj Close'][0], "\n"

bought = False
strategy_ret = 0.0
begin_price = 0.0

for i in range(17, stock.shape[0]):
    if bought == False:        
        if stock['%K14'][i] < 20 and stock['%D14'][i] < 20:
            print "BUY: ", stock['Adj Close'][i], " on ", stock.index[i]
            begin_price = stock['Adj Close'][i]
            bought = True
    else:        
        if stock['%K14'][i] > 80 and stock['%D14'][i] > 80:
            print "SELL: ", stock['Adj Close'][i], " on ", stock.index[i]
            strategy_ret = strategy_ret + stock['Adj Close'][i] - begin_price
            bought = False
print "Strategy 1 (14 day) Return: ", strategy_ret, "\n"

bought = False
strategy_ret = 0.0
begin_price = 0.0

for i in range(8, stock.shape[0]):
    if bought == False:        
        if stock['%K5'][i] < 20 and stock['%D5'][i] < 20:
            print "BUY: ", stock['Adj Close'][i], " on ", stock.index[i]
            begin_price = stock['Adj Close'][i]
            bought = True
    else:        
        if stock['%K5'][i] > 80 and stock['%D5'][i] > 80:
            print "SELL: ", stock['Adj Close'][i], " on ", stock.index[i]
            strategy_ret = strategy_ret + stock['Adj Close'][i] - begin_price
            bought = False

print "Strategy 2 (5 day) Return: ", strategy_ret
    
plt.plot(stock['Adj Close'])
plt.show()

plt.figure(2)
plt.plot(stock['%K14'])
plt.plot(stock['%D14'])
plt.show()
