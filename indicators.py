import pandas as pd

def calculate_williams_r(data, period=14):
    """
    Calculate the Williams %R indicator for a given dataset.

    Williams %R is a momentum indicator that measures overbought and oversold levels.
    It compares the current closing price to the highest high and lowest low over a specified period.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' price columns.
    period (int): The look-back period over which to calculate the indicator (default is 14).

    Returns:
    pd.Series: A Pandas Series containing the Williams %R values.
    """
    # Calculate the highest high over the specified period
    highest_high = data['High'].rolling(window=period).max()
    # Calculate the lowest low over the specified period
    lowest_low = data['Low'].rolling(window=period).min()
    # Calculate the Williams %R
    williams_r = (highest_high - data['Close']) / (highest_high - lowest_low) * -100
    return williams_r

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'Close' price column.
    fast (int): The period for the fast exponential moving average (default is 12).
    slow (int): The period for the slow exponential moving average (default is 26).
    signal (int): The period for the signal line (default is 9).

    Returns:
    tuple: A tuple containing three Pandas Series (MACD line, Signal line, Histogram).
    """
    # Calculate the fast exponential moving average
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    # Calculate the slow exponential moving average
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    # Calculate the MACD line
    macd = ema_fast - ema_slow
    # Calculate the Signal line
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    # Calculate the MACD Histogram
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) indicator.

    RSI is a momentum oscillator that measures the speed and change of price movements.
    It is used to identify overbought or oversold conditions in a traded asset.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'Close' price column.
    period (int): The look-back period over which to calculate the RSI (default is 14).

    Returns:
    pd.Series: A Pandas Series containing the RSI values.
    """
    # Calculate the difference in price from the previous step
    delta = data['Close'].diff()
    # Calculate gains (positive changes)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    # Calculate losses (negative changes)
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    # Calculate the Relative Strength (RS)
    rs = gain / loss
    # Calculate the RSI
    return 100 - (100 / (1 + rs))