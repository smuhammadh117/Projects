import datetime as dt
import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo

def author():

    # Returns
    #     The GT username of the student
    #
    # Return type
    #     str

    return "shusainie3"


######################## SMA Indicator#1 ##########################
# Function to determine the Rolling Mean
def RollingMean(prices,window=20):

    sma = prices.rolling(window,center=False).mean()

    return sma

#################################################################




# Function to determine the Rolling standard deviation
def RollingStd(prices, window=20):

    rolling_std = prices.rolling(window).std()

    return rolling_std





######################## Bollinger Bands Indicator#2 ##########################
# Function to determine the Upper Bollinger Band
def BollingerBands(prices, window=20):

    # Get the rolling mean
    rolling_mean = RollingMean(prices,window=20)
    # Get the rolling std
    rolling_std = RollingStd(prices,window=20)
    #Get bollinger band
    upper_band = rolling_mean + rolling_std*2
    lower_band = rolling_mean - rolling_std*2

    bb_value = (prices - rolling_mean)/(2 * rolling_std)

    combined_bands = bb_value * (upper_band-lower_band)

    return combined_bands


# Function to determine the Bollinger Band Value
def BollingerBandValue(prices, window=20):

    # Get the rolling mean
    rolling_mean = RollingMean(prices,window=20)
    # Get the rolling std
    rolling_std = RollingStd(prices,window=20)

    bb_value = (prices - rolling_mean)/(2 * rolling_std)

    return bb_value.dropna()

########################################################################







######################## Momentum Indicator#3 ##########################
# Function to determine the Moment
def Moment(prices,window=15):

    momentum = prices/prices.shift(window) - 1

    return momentum.dropna()

#######################################################################








######################## Stochastic Indicator#4 ##########################
# Function to determine the Stochastic Oscillator
def StochasticOscillator(prices, N=14, d_smooth = 3):

    low = prices.rolling(window=N).min()
    high = prices.rolling(window=N).max()

    k = ((prices - low) / (high - low)) * 100
    k_smoothed = k.rolling(window=d_smooth).mean()


    return k_smoothed.dropna()

#################################################################







######################## MACD Indicator#5 ##########################
def MACD(prices):

    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26= prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    combined_macd = (macd_line/macd_hist * signal_line/macd_hist)

    return combined_macd
#################################################################


# Run the indicators
def run():
    syms = ["JPM"]
    sd = dt.datetime(2007, 10, 1)
    actual_sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009,12,31)
    date_range = pd.date_range(actual_sd,ed)
    prices_all = get_data(syms, date_range)  # automatically adds SPY
    prices = prices_all[syms]
    norm_prices = prices/prices.iloc[0]
    wind = 12

    # Bollinger Bands
    bb = BollingerBands(norm_prices,window=wind)
    print(bb)
    ub = bb.iloc[:len(norm_prices)]
    # print(ub)
    lb = bb.iloc[len(norm_prices):]
    bb_value = BollingerBandValue(norm_prices, window=wind)


    # SMA
    sma = RollingMean(norm_prices, window = wind)

    # Momentum
    momentum = Moment(norm_prices, window = wind)

    # Stochastic Oscillator
    k_smooth = StochasticOscillator(norm_prices, N = 14, d_smooth = 3)

    # MACD
    macd = MACD(norm_prices)
    macd_hist = macd[len(norm_prices)*2:]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(ub, color="blue", linestyle="--", label="Upper Bollinger Band")
    plt.plot(lb, color="red", linestyle="--", label="Lower Bollinger Band")
    plt.plot(sma, color="gold", label="SMA")
    plt.plot(norm_prices, color="grey", label="Price")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Portfolio Value", fontsize=12)
    plt.title("Bollinger Bands for a " +str(wind)+ " day window", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(actual_sd,ed)
    plt.legend(loc="lower left")
    plt.savefig("Indicator1_BollingerBands.png")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    plt.axhline(y=1, color="red", linestyle='--')
    plt.axhline(y=-1, color="red", linestyle='--')
    plt.plot(bb_value, color="black", label="BB Value")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Bollinger Band Value", fontsize=12)
    plt.title("Bollinger Band value over time for a " +str(wind)+ " day window", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(actual_sd,ed)
    plt.legend(loc="lower left")
    plt.savefig("Indicator1_BollingerBandValue")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    plt.plot(norm_prices, color="grey", label="Price")
    plt.plot(sma, color="orange", label="SMA")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Portfolio Value", fontsize=12)
    plt.title("Simple Moving Average (SMA) and Price comparison for a " +str(wind)+ " day window", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(actual_sd,ed)
    plt.legend(loc="lower left")
    plt.savefig("Indicator2_SMA.png")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    plt.plot(momentum, color="Orange", label="Momentum")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Momentum", fontsize=12)
    plt.title("Momentum for " +str(wind)+ " day window", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(actual_sd,ed)
    plt.legend(loc="lower left")
    plt.savefig("Indicator3_Momentum.png")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    plt.axhline(80, color='red', linestyle='--')
    plt.axhline(20, color='red', linestyle='--')
    plt.plot(k_smooth, color="blue", label="%K")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Stochastic Value", fontsize=12)
    plt.title("Stochastic Oscillator %K and %D (14 day window, 3 smoothing for D)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(actual_sd,ed)
    plt.legend(loc="lower left")
    plt.savefig("Indicator4_StochasticOscillator.png")
    plt.clf()

    macd_hist_array = macd_hist.to_numpy()
    macd_hist_array = macd_hist_array.squeeze()

    fig = plt.figure(figsize=(10, 6))
    plt.plot(macd[:len(norm_prices)], color="Violet", label="MACD Line")
    plt.plot(macd[len(norm_prices):2*len(norm_prices)], color="Indigo", label="MACD Signal")
    plt.bar(macd_hist.index,macd_hist_array,width=1, color = "grey", label = "MACD Histogram")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized MACD Value", fontsize=12)
    plt.title("MACD", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(actual_sd,ed)
    plt.legend(loc="lower left")
    plt.savefig("Indicator5_MACD.png")
    plt.clf()


if __name__ == "__main__":
    run()