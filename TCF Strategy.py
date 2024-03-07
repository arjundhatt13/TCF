import warnings
warnings.filterwarnings("ignore", message="The 'unit' keyword in TimedeltaIndex construction is deprecated.*")
warnings.filterwarnings("ignore", message="Setting an item of incompatible dtype is deprecated")
warnings.filterwarnings("ignore", message="The default fill_method='pad' in DataFrame.pct_change is deprecated")
import math
import yfinance as yf
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import timedelta
from scipy.stats import norm

# This function scrapes historical data to find the weekly return mean and volatility
def dataScraper(tickers):
    endDate = pd.Timestamp.now()
    startDate = endDate - timedelta(weeks = 52 * 5)
    historicalData = yf.download(tickers, start=startDate, end=endDate, interval="1wk")['Close']
    return historicalData

def portfolioConstructor(metricsData, availableCapital): 
    
    # Convert to Log Returns
    stockMetrics = metricsData
    stockMetrics = np.log(stockMetrics.pct_change() + 1)
    stockMetrics = stockMetrics.tail(len(stockMetrics)-1)
    
    # Calculate Mean & Volatility
    stockMetrics.loc['Mean'] = stockMetrics.mean()
    stockMetrics.loc['Volatility'] = stockMetrics.std()
    
    # Reformat
    stockMetrics = stockMetrics.tail(3)
    stockMetrics.index = ['Recent Movement', 'Mean', 'Volatility']
    
    # Z-Score
    stockMetrics.loc['Z-Score'] = (stockMetrics.loc['Recent Movement'] - 2*stockMetrics.loc['Mean']) / stockMetrics.loc['Volatility']
    
    # Select Z-Scores that meet the threshold
    threshold = norm.ppf(0.90)
    stockMetrics.loc['Threshold Z-Score'] = (abs(stockMetrics.loc['Z-Score']) > threshold) * stockMetrics.loc['Z-Score']
    # Asymmetries Adjustment
    weeklyLoanFee = (1 + 0.15) ** (1/52) - 1
    stockMetrics.loc['Z-Score'] = stockMetrics.loc['Z-Score'].apply(lambda x: (1 - weeklyLoanFee) * x if x > 0 else x)

    # Weight Calculation
    sumZ = sum(abs(stockMetrics.loc['Threshold Z-Score']))
    sumNZ = stockMetrics.loc['Threshold Z-Score'][stockMetrics.loc['Threshold Z-Score'] > 0].sum()

    if math.isnan(sumZ):
        sumZ = 1
    
    # Weightings
    stockMetrics.loc['Portfolio Weight'] = -stockMetrics.loc['Threshold Z-Score'] / (sumZ - sumNZ/2)
    
    stockMetrics['Cash'] = 0
    stockMetrics.loc['Portfolio Weight', 'Cash (Margin)'] = (1 - sum(stockMetrics.loc['Portfolio Weight']))

    # Positions
    stockMetrics.loc['Opening Position'] = availableCapital * stockMetrics.loc['Portfolio Weight']
    
    return stockMetrics

############## Call Function

# S&P 100 Tickers
tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOG", "BRK-B", "LLY", "AVGO", "JPM", "TSLA", "UNH", "V", "XOM", "MA", "JNJ", "PG", "HD", "MRK", "COST", "ABBV", "ADBE", "AMD", "CRM", "CVX", "NFLX", "WMT", "PEP", "KO", "ACN", "BAC", "TMO", "MCD", "CSCO", "LIN", "ABT", "ORCL", "CMCSA", "INTC", "INTU", "DIS", "WFC", "VZ", "AMGN", "IBM", "CAT", "DHR", "QCOM", "NOW", "UNP", "PFE", "GE", "SPGI", "TXN", "AMAT", "PM", "ISRG", "RTX", "COP", "HON", "T", "BKNG", "LOW", "GS", "NKE", "AXP", "BA", "PLD", "SYK", "MDT", "ELV", "NEE", "LRCX", "TJX", "VRTX", "BLK", "MS", "ETN", "PANW", "PGR", "SBUX", "C", "DE", "MDLZ", "ADP", "CB", "UPS", "REGN", "BMY", "ADI", "GILD", "MU", "MMC", "BSX", "CI", "LMT", "CVS", "SCHW"]

# Backtest Details
availableCapital = 5000000

# Grab Historical Data
metricsData = dataScraper(tickers)

# Compute the Portfolio
stockMetrics = portfolioConstructor(metricsData, availableCapital)
positions = stockMetrics.tail(2).loc[:, (stockMetrics.tail(2) != 0).any()]
positions.loc['Opening Position'] = positions.loc['Opening Position'].apply(lambda x: '${:,.2f}'.format(x))
positions.loc['Portfolio Weight'] = positions.loc['Portfolio Weight'].apply(lambda x: '{:,.2f}%'.format(x * 100))
print(positions)
