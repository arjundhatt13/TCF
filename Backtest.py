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
def dataScraper(tickers, backtestLength):
    endDate = pd.Timestamp.now()
    startDate = endDate - timedelta(weeks = backtestLength * 52 + 52 * 5)
    historicalData = yf.download(tickers, start=startDate, end=endDate, interval="1wk")['Close']
    return historicalData

def portfolioConstructor(metricsData, availableCapital): 
    
    # Convert to Log Returns
    stockMetrics = metricsData
    stockMetrics = np.log(stockMetrics.pct_change() + 1)
    stockMetrics = stockMetrics.tail(len(stockMetrics)-1)
    
    # Remove Future Return
    futureReturn = stockMetrics.iloc[-1]
    futureReturn.loc['Cash'] = 0
    stockMetrics = stockMetrics.drop(stockMetrics.tail(1).index)

    # Calculate Mean & Volatility
    stockMetrics.loc['Mean'] = stockMetrics.mean()
    stockMetrics.loc['Volatility'] = stockMetrics.std()
    
    # Reformat
    stockMetrics = stockMetrics.tail(3)
    stockMetrics.index = ['Recent Movement', 'Mean', 'Volatility']
    
    # Z-Score
    stockMetrics.loc['Z-Score'] = (stockMetrics.loc['Recent Movement'] - 2*stockMetrics.loc['Mean']) / stockMetrics.loc['Volatility']
    sumZ = sum(abs(stockMetrics.loc['Z-Score']))
    sumNZ = stockMetrics.loc['Z-Score'][stockMetrics.loc['Z-Score'] < 0].sum()

    if math.isnan(sumZ):
        sumZ = 1

    # Select Z-Scores that meet the threshold
    threshold = norm.ppf(0.90)
    stockMetrics.loc['Threshold Z-Score'] = (abs(stockMetrics.loc['Z-Score']) > threshold) * stockMetrics.loc['Z-Score']

    # Weightings
    stockMetrics.loc['Portfolio Weight'] = -stockMetrics.loc['Threshold Z-Score'] / (sumZ + sumNZ/2)
    
    stockMetrics['Cash'] = 0
    stockMetrics.loc['Portfolio Weight', 'Cash'] = (1 - sum(stockMetrics.loc['Portfolio Weight']))

    # Positions
    stockMetrics.loc['Opening Position'] = availableCapital * stockMetrics.loc['Portfolio Weight']

    # Actual Movement
    stockMetrics.loc['Actual Return'] = np.exp(futureReturn) - 1 
    stockMetrics = stockMetrics.tail(7)

    # Ending Capital
    stockMetrics.loc['Ending Position'] = stockMetrics.loc['Opening Position'] * (1 + stockMetrics.loc['Actual Return'])
    stockMetrics.loc['P&L'] = stockMetrics.loc['Ending Position'] - stockMetrics.loc['Opening Position']
    return stockMetrics



############## Call Function

# S&P 100 Tickers (Five Years Ago)
tickers = [ "AAPL", "ABBV", "ABT", "ACN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "META", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HAL", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY","PEP", "PFE", "PG", "PM", "QCOM", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
 

# Backtest Details
backtestLength = 5
startingCapital = 5000000
availableCapital = startingCapital

# Grab Historical Data
historicalData = dataScraper(tickers, backtestLength)
dates = historicalData.index.tolist()

# Portfolio Information
returns = pd.DataFrame(columns=["Date", "Return", "Portfolio Balance"])

for i in range(len(dates) - (52 * backtestLength) - 1):
    #Week we are curently on
    currentWeek = dates[i + 5*52]
    nextWeek = dates[i + 5*52 + 1]
    
    # Segment Historical Data
    metricsData = historicalData.loc[(historicalData.index >= dates[i]) & (historicalData.index <= nextWeek)]
    
    # Run Backtest
    stockMetrics = portfolioConstructor(metricsData, availableCapital)
    
    
    # Store Data
    rReturn = sum(stockMetrics.loc['Ending Position'])/availableCapital - 1
    availableCapital += sum(stockMetrics.loc['P&L'])
    newRow = {'Date': nextWeek, 'Return': rReturn, 'Portfolio Balance': availableCapital}

    returns.loc[len(returns)] = newRow

# Calculate Return Data
rf = (1 + 0.0177) ** (1/52) - 1

meanReturn = returns['Return'].mean()
stdReturn = returns['Return'].std()
hpr = (availableCapital/startingCapital)**(1/backtestLength)-1

sharpe  = (meanReturn - rf) / stdReturn * np.sqrt(52)

print(f"{meanReturn * 100:.2f}%", "{:.2f}".format(stdReturn), f"{sharpe * 100:.2f}%", f"{hpr * 100:.2f}%")

    
     
 

