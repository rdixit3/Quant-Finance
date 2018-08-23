# Analyze a portfolio - Based off GT CSEC 7644 Project 1

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
import math

# Return CSV file path given ticker symbol
def symbol_to_path(symbol, base_dir="Data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

# Read stock data for given symbols and S&P 500 (reference) from CSV files
def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    symbols.remove('SPY')
    return df

# Plot stock prices
def plot_data(df, title="Stock prices", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.show()

# Normalize prices according to first day
def normalize_data(df):
    return df/df.ix[0,::]

# Compute the daily returns of a stock given a dataframe
def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns = (df/df.shift(1)) - 1
    daily_returns[0] = 0
    return daily_returns

# Compute cumulative return
def cumulativeReturn(df):
    return df[-1]/df[0] - 1

# Compute average daily returns
def avgDailyReturn(df):
    daily_returns = compute_daily_returns(df)
    return daily_returns[1:].mean()

# Compute standard deviation of daily returns (volatility)
def stdevDailyReturn(df):
    daily_returns = compute_daily_returns(df)
    return daily_returns[1:].std()

# Compute Sharpe ratio to determine risk-adjusted return
def sharpeRatio(df, rfr, sf):
    return math.sqrt(sf) * (avgDailyReturn(df) - rfr) / stdevDailyReturn(df)

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), syms = ['GOOG','AAPL','GLD','XOM'], allocs=[0.1,0.2,0.3,0.4], sv=1000000, rfr=0.0, sf=252.0):

    # Read in adjusted closing prices for given symbols and S&P 500 (reference) in date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices = prices_all.drop('SPY', axis=1)
    prices_SPY = prices_all['SPY']

    # Normalize data and generate given portfolio
    prices_norm = normalize_data(prices)
    prices_norm = prices_norm * allocs * sv
    portfolio = prices_norm.sum(axis=1)
    portfolio.name = "Portfolio"

    # Compute statistics
    cr = cumulativeReturn(portfolio)
    adr = avgDailyReturn(portfolio)
    sddr = stdevDailyReturn(portfolio)
    sr = sharpeRatio(portfolio, rfr, sf)

    df = pd.concat([normalize_data(portfolio), normalize_data(prices_SPY)], axis=1)
    plot_data(df, ylabel="Normalized price")

    ev = portfolio[-1]

    return cr, adr, sddr, sr, ev

def prompt():
    print("1: Compute statistics/Portfolio analysis")
    print("2: Rolling statistics with Bollinger bands")
    print("3: Volume")
    print("4: Individual stock statistics")
    print("999: Exit")
    inp = int(input("Enter a number: "))
    return inp
    
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

def rolling_plot(start_date, end_date, symbols=['SPY'], window=20):
    dates = pd.date_range(start_date, end_date)
    df = get_data(symbols, dates)

    symbols.append('SPY')

    count = 1
    
    for stock in symbols:
        plt.figure(count)
        rm = get_rolling_mean(df[stock], window=20) # Compute rolling mean
        rstd = get_rolling_std(df[stock], window=20) # Compute rolling standard deviation
        upper_band, lower_band = get_bollinger_bands(rm, rstd) # Compute upper and lower bands
        ax = df[stock].plot(title="Bollinger Bands", label=stock)
        rm.plot(label='Rolling mean', ax=ax)
        upper_band.plot(label='upper band', ax=ax)
        lower_band.plot(label='lower band', ax=ax)

        # Add axis labels and legend
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left')

        count = count + 1

    plt.show()

#Begin Main
start_date = dt.datetime(2012,1,1)
end_date = dt.datetime(2012,12,31)
symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
allocations = [0.25, 0.25, 0.25, 0.25]
start_val = 1000000 
risk_free_rate = 0.0
sample_freq = 252

window = 20

inp = prompt()

while(inp != 999):
    if(inp == 1):
        # Assess the portfolio
        cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date, syms = symbols, allocs = allocations, sv = start_val)

        print "Start Date:", start_date
        print "End Date:", end_date
        print "Symbols:", symbols
        print "Allocations:", allocations, "\n"
        print "Sharpe Ratio:", sr
        print "Volatility (stdev of daily returns):", sddr
        print "Average Daily Return:", adr
        print "Cumulative Return:", cr, "\n"
        print "Start Value:", start_val
        print "End Value:", ev
        print "Net Profit:", ev-start_val
        
    elif(inp == 2):
        rolling_plot(start_date, end_date)
    elif(inp == 3):
        print("1")#Volume stats
    elif(inp == 4):
        print("1")#Individual stock stats
        
    inp = prompt()
