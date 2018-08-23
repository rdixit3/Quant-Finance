# Runs Monte Carlo Simulations to model possible stock movements to show potential uncertainty

# TODO: combine both figures on same plot to show overall trend

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
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
    return df

# Compute the daily returns of a stock given a dataframe
def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns = (df/df.shift(1)) - 1
    daily_returns[0] = 0
    return daily_returns

# Compute cumulative return
def cumulativeReturn(df):
    return df[-1]/df[0] - 1

#Begin Main
symbol = 'KO'
start = dt.datetime(2017,6,30)
end = dt.datetime(2018,6,30)
trading_days = 252
trials = 30

dates = pd.date_range(start, end)
stock = get_data([symbol], dates)
stock = stock.dropna()
days = dates.size

cagr = (stock[symbol][-1]/stock[symbol][1]) ** (1.0/ (days/365.0)) - 1.0
vol = stock.pct_change().std() * math.sqrt(trading_days)
mu = cagr

#Create list of daily returns using random normal distribution
start_price = stock[symbol][-1]

futures = np.zeros((trials, trading_days + 1))

for i in range(trials):
    # Create random normal distribution of daily returns
    daily_returns = np.random.normal(mu/trading_days,vol/math.sqrt(trading_days),trading_days)+1
    price_list = [start_price]
    
    for x in daily_returns:
        price_list.append(price_list[-1]*x)

    futures[i-1] = price_list
    plt.plot(price_list)

avg = np.mean(futures, axis=0)
print("Projected value on "), (end + dt.timedelta(days=365)).strftime("%m/%d/%y"), " $", avg[-1]
plt.xlabel("Days from End")
plt.ylabel("Value ($)")
plt.plot(avg, color='yellow', linewidth=4)
plt.figure(2)
plt.plot(stock[symbol])
plt.show()
 
