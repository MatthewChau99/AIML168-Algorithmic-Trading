import datetime

import pandas as pd
import pandas_datareader as pdr

# Importing data from datareader
aapl = pdr.get_data_yahoo('AAPL',
                          start=datetime.datetime(2006, 10, 1),
                          end=datetime.datetime(2012, 1, 1))
#
# TIME SERIES AND ITS VISUALIZATION
#

# Inspecting the data 1
aapl.head()
aapl.tail()
aapl.describe()

# Inspecting the data 2
print(aapl.index)
print(aapl.columns)
ts = aapl['Close'][-10:]
type(ts)

# Locating using loc and iloc
print(aapl.loc[
      pd.Timestamp('2006-11-01'):pd.Timestamp('2006-12-31')].head())  # Inspect the first rows of November-December 2006
print(aapl.loc['2007'].head())  # Inspect the first rows of 2007
print(aapl.iloc[22:43])  # Inspect November 2006
print(aapl.iloc[[22, 43], [0, 3]])  # Inspect the 'Open' and 'Close' values at 2006-11-01 and 2006-12-01

# Sampling
sample = aapl.sample(20)  # Sample 20 rows
print(sample)  # Print `sample`
monthly_aapl = aapl.resample('M').mean()  # Resample to monthly level
print(monthly_aapl)  # Print `monthly_aapl`

# Visualization
import matplotlib.pyplot as plt  # Import Matplotlib's `pyplot` module as `plt`

aapl['Close'].plot(grid=True)  # Plot the closing prices for `aapl`
plt.savefig('Time_Series.png')
plt.show()  # Show the plot


#
# COMMON FINANCIAL ANALYSIS
#

# Returns
import numpy as np  # Import `numpy` as `np`

daily_close = aapl[['Adj Close']]
daily_pct_change = daily_close.pct_change()
daily_pct_change.fillna(0, inplace=True)
print(daily_pct_change)  # Inspect daily returns
daily_log_returns = np.log(daily_close.pct_change() + 1)
print(daily_log_returns)  # Print daily log returns

# Visualizing daily percent changes
daily_pct_change.hist(bins=50)  # Plot the distribution of `daily_pct_c`
plt.savefig('Daily_Percent_Changes.png')
plt.show()
print(daily_pct_change.describe())  # Pull up summary statistics

# Cumulative daily returns
cum_daily_return = (1 + daily_pct_change).cumprod()  # Calculate the cumulative daily returns
print(cum_daily_return)

# Plot the cumulative daily returns
cum_daily_return.plot(figsize=(12, 8))
plt.savefig('Cumlative_daily_returns.png')
plt.show()

# Resample the cumulative daily return to cumulative monthly return
cum_monthly_return = cum_daily_return.resample("M").mean()
print(cum_monthly_return)


#
# COLLECTING OTHER STOCK DATA
#

def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))

    datas = map(data, tickers)
    return (pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))


tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))

# Making plots
# Isolate the `Adj Close` values and transform the DataFrame
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

# Calculate the daily percentage change for `daily_close_px`
daily_pct_change = daily_close_px.pct_change()
daily_pct_change.hist(bins=50, sharex=True, figsize=(12, 8))  # Histogram
plt.savefig('Daily_Percentage_Change_Histogram.png')
plt.show()

pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1, figsize=(12, 12))  # Scatter Matrix
plt.savefig('Daily_Percentage_Change_Scatter.png')
plt.show()

# Moving Windows
adj_close_px = aapl['Adj Close']  # Isolate the adjusted closing prices
moving_avg = adj_close_px.rolling(window=40).mean()  # Calculate the moving average
print(moving_avg[-10:])  # Inspect the result

# Moving window rolling
aapl['42'] = adj_close_px.rolling(window=40).mean()  # Short moving window rolling mean
aapl['252'] = adj_close_px.rolling(window=252).mean()  # Long moving window rolling mean
aapl[['Adj Close', '42', '252']].plot()  # Plot the adjusted closing price, the short and long windows of rolling means
plt.savefig('Moving_Window_Rolling.png')
plt.show()

# Volatility Calculation
min_periods = 75  # Define the minimum of periods to consider
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)  # Calculate the volatility
vol.plot(figsize=(10, 8))  # Plot the volatility
plt.savefig('Volatility.png')
plt.show()

#
# ORDINARY LEAST SQUARE REGRESSION
#
import statsmodels.api as sm

all_adj_close = all_data[['Adj Close']]  # Isolate the adjusted closing price
all_returns = np.log(all_adj_close / all_adj_close.shift(1))  # Calculate the returns

aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']  # Isolate the AAPL returns
aapl_returns.index = aapl_returns.index.droplevel('Ticker')

msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']  # Isolate the MSFT returns
msft_returns.index = msft_returns.index.droplevel('Ticker')

# Build up a new DataFrame with AAPL and MSFT returns
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']

X = sm.add_constant(return_data['AAPL'])  # Add a constant

model = sm.OLS(return_data['MSFT'], X).fit()  # Construct the model

print(model.summary())

# Plotting the OLS Regression
plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')
ax = plt.axis()  # Add an axis to the plot
x = np.linspace(ax[0], ax[1] + 0.01)  # Initialize `x`

# Plot the regression line
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)


# Customize the plot
plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft returns')
plt.savefig('Ordinary_Least_Square_Regression.png')
plt.show()

# Plot the rolling correlation
return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()
plt.savefig('Rolling_Correlation.png')
plt.show()

#
#   MOVING AVERAGE CROSSOVER
#
short_window = 40
long_window = 100

# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=aapl.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)

signals['positions'] = signals['signal'].diff()  # Generate trading orders

print(signals)

# Plotting
fig = plt.figure()  # Initialize the plot figure
ax1 = fig.add_subplot(111, ylabel='Price in $')  # Add a subplot and label for y-axis
aapl['Close'].plot(ax=ax1, color='r', lw=2.)  # Plot the closing price
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)  # Plot the short and long moving averages

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')

# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')

plt.savefig('Moving_Average_Crossover.png')
plt.show()

#
#   Backtesting
#
# Set the initial capital
initial_capital = float(100000.0)

# Create a DataFrame `positions`
positions = pd.DataFrame(index=signals.index).fillna(0.0)

# Buy a 100 shares
positions['AAPL'] = 100 * signals['signal']

# Initialize the portfolio with value owned
portfolio = positions.multiply(aapl['Adj Close'], axis=0)

# Store the difference in shares owned
pos_diff = positions.diff()

portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

print(portfolio.head())

# Plotting the backtest
# Create a figure
fig = plt.figure()

ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2.)

ax1.plot(portfolio.loc[signals.positions == 1.0].index,
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(portfolio.loc[signals.positions == -1.0].index,
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='k')

plt.savefig('Backtesting.png')
plt.show()
