from tradingview_screener import Query, Column
import pandas as pd
import numpy as np
import time


# Types of Queries
def undervalued_growth(i):
    """
    Screens for undervalued growth stocks with the following criteria:
    - EPS (Earnings Per Share) not between 0-0.24 (filtering out low earnings)
    - PEG ratio < 1 (indicates stock may be undervalued relative to growth)
    - P/E ratio between 0-20 (reasonable valuation)
    - Beta between 1-1.5 (moderate market correlation)
    Returns top 50 results sorted by volume
    """
    query = (Query()
             .select('name', 'close', 'volume')
             .where(
        Column('basic_eps_net_income').not_between(0, .24),
        Column('price_earnings_growth_ttm') < 1,
        Column('price_earnings_ttm').between(0, 20),
        Column('beta_1_year').between(1, 1.5)
    )
             .order_by('volume', ascending=False)
             .offset(i + 1)
             .limit(50 + i)
             .get_scanner_data())
    return query


def undervalued_large_cap(i):
    """
    Screens for undervalued large-cap stocks with:
    - PEG ratio < 1 (potentially undervalued)
    - P/E ratio between 0-20 (reasonable valuation)
    - Beta between 1-1.5 (moderate market correlation)
    - Market cap > $1 billion (large-cap companies)
    Sorted by trading volume
    """
    query = (Query()
             .select('name', 'close', 'volume')
             .where(
        Column('price_earnings_growth_ttm') < 1,
        Column('price_earnings_ttm').between(0, 20),
        Column('beta_1_year').between(1, 1.5),
        Column('market_cap_basic') > 1e9
    )
             .order_by('volume', ascending=False)
             .offset(i + 1)
             .limit(50 + i)
             .get_scanner_data())
    return query


def high_beta(i):
    """
    Screens for volatile stocks with:
    - Beta > 1.2 (higher market sensitivity)
    - Market cap > $100 million (mid to large-cap)
    - Volume > 500,000 (ensures liquidity)
    Useful for traders seeking higher-risk, higher-reward opportunities
    """
    query = (Query()
             .select('name', 'close', 'volume')
             .where(
        Column('beta_1_year') > 1.2,
        Column('market_cap_basic') > 100e6,
        Column('volume') > 5e5
    )
             .order_by('volume', ascending=False)
             .offset(i + 1)
             .limit(50 + i)
             .get_scanner_data())
    return query


def low_beta(i):
    """
    Screens for defensive stocks with:
    - Negative beta (moves opposite to market)
    - Market cap > $100 million (mid to large-cap)
    - Volume > 500,000 (ensures liquidity)
    Useful for portfolio hedging or defensive strategies
    """
    query = (Query()
             .select('name', 'close', 'volume')
             .where(
        Column('beta_1_year') < 0,
        Column('market_cap_basic') > 100e6,
        Column('volume') > 5e5
    )
             .order_by('volume', ascending=False)
             .offset(i + 1)
             .limit(50 + i)
             .get_scanner_data())
    return query


def low_52_week(i):
    """
    Screens for stocks trading near their 52-week lows:
    - Price within 10% of 52-week low
    Potential value opportunities or stocks in downtrends
    """
    query = (Query()
             .select('name', 'close', 'volume')
             .where(
        Column('close').above_pct('price_52_week_low', 0.1)
    )
             .order_by('volume', ascending=False)
             .offset(i + 1)
             .limit(50 + i)
             .get_scanner_data())
    return query


def high_52_week(i):
    """
    Screens for stocks trading near their 52-week highs:
    - Price within 10% of 52-week high
    Potential momentum plays or strong performers
    """
    query = (Query()
             .select('name', 'close', 'volume')
             .where(
        Column('close').below_pct('price_52_week_high', 0.1)
    )
             .order_by('volume', ascending=False)
             .offset(i + 1)
             .limit(50 + i)
             .get_scanner_data())
    return query


def high_peg(i):
    """
    Screens for stocks with high PEG ratios:
    - Minimum volume of 50,000 (ensures liquidity)
    - Market cap > $100 million (mid to large-cap)
    - PEG ratio > 1 (potentially overvalued relative to growth)
    Could indicate overvalued stocks or high growth expectations
    """
    query = (Query()
             .select('name', 'close', 'volume')
             .where(
        Column('volume') > 50000,
        Column('market_cap_basic') > 100e6,
        Column('price_earnings_growth_ttm') > 1
    )
             .order_by('volume', ascending=False)
             .offset(i + 1)
             .limit(50 + i)
             .get_scanner_data())
    return query


def get_symbols(NUM_STOCKS, screeners = ""):
    index = np.arange(0, NUM_STOCKS + 1, 50).tolist()
    symbols = set()  # Using set for unique symbols

    if screeners == "":
        screeners = [
            undervalued_growth,
            undervalued_large_cap,
            high_beta
        ]

    for screener in screeners:
        print(f"Running {screener.__name__}...")
        for i in index:
            try:
                query = screener(i)
                count, output_data = query

                if output_data is None or count == 0:
                    print(f"No more data for {screener.__name__}")
                    break

                results_df = pd.DataFrame(output_data)
                new_symbols = set(results_df['name'].tolist())
                symbols.update(new_symbols)
                print(f"Found {len(new_symbols)} new symbols. Total unique: {len(symbols)}")

            except Exception as e:
                print(f"Error in {screener.__name__}: {str(e)}")
                break

    return list(symbols)

