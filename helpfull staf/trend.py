import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

# Colecting data
market = 'INTC'
end = dt.datetime(2016, 12, 31)
start = dt.date(end.year - 10, end.month, end.day)
market_data = web.DataReader(market, 'google', start, end)

#Calculating EMA and difference
market_data['ema'] = market_data['Close'].ewm(200).mean()
market_data['diff_pc'] = (market_data['Close'] / market_data['ema']) - 1

#Defining bull/bear signal
TH = 0
market_data['Signal'] = np.where(market_data['diff_pc'] > TH, 1, 0)
market_data['Signal'] = np.where(market_data['diff_pc'] < -TH, -1, market_data['Signal'])

# Plot data and fits

import seaborn as sns  # This is just to get nicer plots

signal = market_data['Signal']

# How many consecutive signals are needed to change trend
min_signal = 2

# Find segments bounds
bounds = (np.diff(signal) != 0) & (signal[1:] != 0)
bounds = np.concatenate(([signal[0] != 0], bounds))
bounds_idx = np.where(bounds)[0]
# Keep only significant bounds
relevant_bounds_idx = np.array([idx for idx in bounds_idx if np.all(signal[idx] == signal[idx:idx + min_signal])])
# Make sure start and end are included
if relevant_bounds_idx[0] != 0:
    relevant_bounds_idx = np.concatenate(([0], relevant_bounds_idx))
if relevant_bounds_idx[-1] != len(signal) - 1:
    relevant_bounds_idx = np.concatenate((relevant_bounds_idx, [len(signal) - 1]))

# Iterate segments
for start_idx, end_idx in zip(relevant_bounds_idx[:-1], relevant_bounds_idx[1:]):
    # Slice segment
    segment = market_data.iloc[start_idx:end_idx + 1, :]
    x = np.array(mdates.date2num(segment.index.to_pydatetime()))
    print(x)
    # Plot data
    data_color = 'green' if signal[start_idx] > 0 else 'red'
    plt.plot(segment.index, segment['Close'], color=data_color)
    # Plot fit
    coef, intercept = np.polyfit(x, segment['Close'], 1)
    fit_val = coef * x + intercept
    print(fit_val, coef, intercept)
    fit_color = 'yellow' if coef > 0 else 'blue'
    plt.plot(segment.index, fit_val, color=fit_color)
    plt.show()