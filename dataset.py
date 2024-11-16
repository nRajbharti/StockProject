import time
import pandas as pd
import yfinance as yf
import json
from pathlib import Path
import indicators
import json_keys
import screeners
from difflib import get_close_matches
import shutil


class StockDataProcessor:
    """
    A class to process and store stock market data with technical indicators.

    This class downloads stock data, calculates various technical indicators,
    and saves the processed data in a JSON format with a specific folder structure.

    Attributes:
        symbol (str): The stock symbol to process
        period (str): The time period for data download (default: "1y")
        first_index (int): Starting index for data processing
        days_window (int): Window size for calculating metrics
        prediction_window (int): Window size for future prediction
    """

    def __init__(self, symbol, period="max", days_window=20, prediction_window=15, first_index=16):
        """
        Initialize the StockDataProcessor with given parameters.

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period for data download
            days_window (int): Window size for calculating metrics
            prediction_window (int): Window size for future prediction
            first_index (int): Starting index for data processing
        """
        self.symbol = symbol
        self.period = period
        self.days_window = days_window
        self.prediction_window = prediction_window
        self.first_index = first_index
        self.data = None
        self.etf_data = None
        self.stock_info = None
        self.list_of_dicts = []

    def download_data(self):
        """Download stock data and information using yfinance."""
        self.data = yf.download(self.symbol, period=self.period)
        ticker = yf.Ticker(self.symbol)
        self.stock_info = ticker.info

    def get_stock_info(self):
        """
        Extract basic stock information.

        Returns:
            tuple: Contains sector, industry, and beta values
        """
        sector = self.stock_info.get('sector', 'N/A')
        industry = self.stock_info.get('industry', 'N/A')
        beta = self.stock_info.get('beta', 'N/A')
        return sector, industry, beta

    def get_common_indices(self):
        """
        Calculate common indices for data processing.

        Returns:
            list: List of indices for processing
        """
        last_index = len(self.data) - 1
        common_indices = []
        current_index = self.first_index

        while current_index + self.days_window + self.prediction_window <= last_index:
            common_indices.append(current_index)
            current_index += self.days_window

        return common_indices

    def process_data(self):
        """Process stock data and calculate technical indicators."""
        try:
            sector, industry, beta = self.get_stock_info()
            self.etf_data = self.map_sector_etf(sector=sector)
            common_indices = self.get_common_indices()

            # Calculate technical indicators
            williams_r = indicators.calculate_williams_r(self.data)
            macd, signal_line, histogram = indicators.calculate_macd(self.data)
            rsi = indicators.calculate_rsi(self.data)

            for idx in common_indices:
                current_dict = {}

                # Process each column in the data
                for column_name in self.data.columns:
                    simple_column_name = column_name[0] if isinstance(column_name, tuple) else column_name

                    try:
                        col_idx = self.data.columns.get_loc(column_name)
                        values = self.data.iloc[idx: idx + self.days_window, col_idx].tolist()
                        current_dict[simple_column_name] = values
                    except (KeyError, IndexError):
                        continue

                # Calculate growth and other metrics
                try:
                    close_col_idx = self.data.columns.get_loc(('Close', self.symbol))

                    # Check if we have enough data for the calculation
                    if (idx + self.days_window + self.prediction_window >= len(self.data) or
                            idx + self.days_window + self.prediction_window >= len(self.etf_data)):
                        continue

                    starting_value = self.data.iloc[idx + self.days_window - 1, close_col_idx]
                    ending_value = self.data.iloc[idx + self.days_window + self.prediction_window, close_col_idx]

                    etf_starting_value = self.etf_data.iloc[idx + self.days_window - 1, close_col_idx]
                    etf_ending_value = self.etf_data.iloc[
                        idx + self.days_window + self.prediction_window, close_col_idx]

                    # Check for invalid values
                    if (pd.isna(starting_value) or pd.isna(ending_value) or
                            pd.isna(etf_starting_value) or pd.isna(etf_ending_value)):
                        continue

                    etf_growth = json_keys.return_growth(etf_starting_value, etf_ending_value)
                    growth = json_keys.return_growth(starting_value, ending_value)

                    # Get technical indicators
                    williams_r_values = williams_r.iloc[idx: idx + self.days_window].values.flatten().tolist()
                    macd_diff_values = (macd.iloc[idx: idx + self.days_window] -
                                        signal_line.iloc[idx: idx + self.days_window]).values.flatten().tolist()
                    rsi_values = rsi.iloc[idx: idx + self.days_window].values.flatten().tolist()

                    # Check if technical indicators contain any NaN values
                    if (any(pd.isna(williams_r_values)) or any(pd.isna(macd_diff_values)) or
                            any(pd.isna(rsi_values))):
                        continue

                    current_dict.update({
                        'Growth': growth,
                        'market_comp': json_keys.percent_difference(growth, etf_growth),
                        'Williams_R': williams_r_values,
                        'macd_diff': macd_diff_values,
                        'rsi': rsi_values
                    })

                except (KeyError, IndexError, ValueError, ZeroDivisionError):
                    continue

                # Add additional information
                try:
                    current_dict.update({
                        'Sector': json_keys.return_sector(sector),
                        'Industry': json_keys.return_industry(industry),
                        'Symbol': json_keys.return_symbol(self.symbol),
                        'Beta': json_keys.return_beta(beta)
                    })
                except (ValueError, TypeError):
                    continue

                self.list_of_dicts.append(current_dict)

        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return

    def save_to_json(self):
        """Save processed data to JSON file in appropriate folder structure."""
        # Convert non-string keys to strings
        for d in self.list_of_dicts:
            for key in list(d.keys()):
                if not isinstance(key, str):
                    d[str(key)] = d.pop(key)

        # Create folder structure
        sector, industry, _ = self.get_stock_info()
        folder_name = Path("data")
        sector_name = folder_name / sector.replace(' ', '_')
        industry_name = sector_name / industry.replace(' ', '_')

        # Create directories
        for path in [folder_name, sector_name, industry_name]:
            path.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        filename = industry_name / f"{self.symbol}.json"
        with open(filename, 'w') as f:
            json.dump(self.list_of_dicts, f, indent=4)

        print(f"Data saved to {filename}")

    def run(self):
        """Execute the complete data processing pipeline."""
        try:
            self.download_data()
            self.process_data()
            self.save_to_json()
        except Exception as e:
            print(f"Failed download:\n['{self.symbol}']: {str(e)}")
            # Create minimal folder structure for failed downloads
            folder_name = Path("data")
            folder_name.mkdir(parents=True, exist_ok=True)

            folder_name_error = Path("error")
            folder_name_error.mkdir(parents=True, exist_ok=True)

            # Save empty or minimal data to indicate failed download
            filename = folder_name_error / "N/A" / "N/A" / f"{self.symbol}.json"
            filename.parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as f:
                json.dump([], f)
            print(f"Data saved to {filename}")

    def map_sector_etf(self, sector):
        """
        Maps the stock sector to an ETF
        Parameter: Stock sector, string
        Returns an ETF yfinance stock dataframe
        """
        sector_mapping = {
            # Main Sector SPDR ETFs
            'Technology': 'XLK',  # Technology Select Sector SPDR Fund
            'Financial': 'XLF',  # Financial Select Sector SPDR Fund
            'Healthcare': 'XLV',  # Health Care Select Sector SPDR Fund
            'Consumer Cyclical': 'XLY',  # Consumer Discretionary Select Sector SPDR Fund
            'Energy': 'XLE',  # Energy Select Sector SPDR Fund
            'Utilities': 'XLU',  # Utilities Select Sector SPDR Fund
            'Materials': 'XLB',  # Materials Select Sector SPDR Fund
            'Industrial': 'XLI',  # Industrial Select Sector SPDR Fund
            'Consumer Staples': 'XLP',  # Consumer Staples Select Sector SPDR Fund
            'Real Estate': 'XLRE',  # Real Estate Select Sector SPDR Fund
            'Communication Services': 'XLC',  # Communication Services Select Sector SPDR Fund

            # Technology Sub-sectors
            'Semiconductor': 'SOXX',  # iShares Semiconductor ETF
            'Software': 'IGV',  # iShares Expanded Tech-Software ETF
            'Cybersecurity': 'HACK',  # ETFMG Prime Cyber Security ETF
            'Cloud Computing': 'SKYY',  # First Trust Cloud Computing ETF
            'Artificial Intelligence': 'BOTZ',  # Global X Robotics & Artificial Intelligence ETF

            # Healthcare Sub-sectors
            'Biotech': 'XBI',  # SPDR S&P Biotech ETF
            'Medical Devices': 'IHI',  # iShares U.S. Medical Devices ETF
            'Healthcare Providers': 'IHF',  # iShares U.S. Healthcare Providers ETF
            'Pharmaceuticals': 'XPH',  # SPDR S&P Pharmaceuticals ETF

            # Financial Sub-sectors
            'Regional Banks': 'KRE',  # SPDR S&P Regional Banking ETF
            'Insurance': 'KIE',  # SPDR S&P Insurance ETF
            'Capital Markets': 'IAI',  # iShares U.S. Broker-Dealers & Securities Exchanges ETF

            # Industrial Sub-sectors
            'Aerospace & Defense': 'ITA',  # iShares U.S. Aerospace & Defense ETF
            'Transportation': 'IYT',  # iShares Transportation Average ETF
            'Construction': 'ITB',  # iShares U.S. Home Construction ETF

            # Energy Sub-sectors
            'Oil & Gas Exploration': 'XOP',  # SPDR S&P Oil & Gas Exploration & Production ETF
            'Clean Energy': 'ICLN',  # iShares Global Clean Energy ETF
            'Solar': 'TAN',  # Invesco Solar ETF
            'Nuclear': 'NLR',  # VanEck Uranium+Nuclear Energy ETF

            # Materials Sub-sectors
            'Gold Miners': 'GDX',  # VanEck Gold Miners ETF
            'Steel': 'SLX',  # VanEck Steel ETF
            'Timber': 'WOOD',  # iShares Global Timber & Forestry ETF

            # Consumer Sub-sectors
            'Retail': 'XRT',  # SPDR S&P Retail ETF
            'Home Builders': 'XHB',  # SPDR S&P Homebuilders ETF
            'Leisure': 'PEJ',  # Invesco Dynamic Leisure and Entertainment ETF
            'Cannabis': 'MSOS',  # AdvisorShares Pure US Cannabis ETF

            # Alternative Classifications
            'Infrastructure': 'IGF',  # iShares Global Infrastructure ETF
            'Water': 'PHO',  # Invesco Water Resources ETF
            'Agriculture': 'DBA',  # Invesco DB Agriculture Fund
            'Gaming': 'ESPO',  # VanEck Video Gaming and eSports ETF
            'Social Media': 'SOCL',  # Global X Social Media ETF
            'Fintech': 'FINX',  # Global X FinTech ETF
            'E-commerce': 'EBIZ',  # Global X E-commerce ETF
            'Internet': 'FDN',  # First Trust Dow Jones Internet Index Fund
            'Blockchain': 'BLOK',  # Amplify Transformational Data Sharing ETF
            'Metaverse': 'META',  # Roundhill Ball Metaverse ETF

            # International Sectors
            'Emerging Markets': 'EEM',  # iShares MSCI Emerging Markets ETF
            'Developed Markets': 'EFA',  # iShares MSCI EAFE ETF
            'BRIC': 'BKF',  # iShares MSCI BRIC ETF
        }
        # Convert input sector to lowercase for better matching
        sector = sector.lower().replace(' services', '')

        # Create a lowercase version of sector_mapping for matching
        sector_mapping_lower = {k.lower(): v for k, v in sector_mapping.items()}

        try:
            # Try exact match first (case-insensitive)
            if sector in sector_mapping_lower:
                etf_symbol = sector_mapping_lower[sector]
            else:
                # Find closest matching sector using fuzzy matching
                matches = get_close_matches(sector, sector_mapping_lower.keys(), n=1, cutoff=0.6)
                if matches:
                    etf_symbol = sector_mapping_lower[matches[0]]
                    print(f"Matched '{sector}' to '{matches[0]}' -> {etf_symbol}")
                else:
                    print(f"No matching ETF sector found for: {sector}")
                    etf_symbol = self.symbol

        except Exception as e:
            print(f"Error matching ETF sector: {str(e)}")
            etf_symbol = self.symbol

        return yf.download(etf_symbol, period=self.period)


#######################
# Main Execution
#######################
symbols = screeners.get_symbols(250)  # Change this number as desired, use 10 for debug
num_symbols = len(symbols)
iterations = 1
# symbols = ['META', 'AAPL', "BX", "TAN", "TSLA"]  # Uncomment this for a quick test

for symbol in symbols:
    processor = StockDataProcessor(symbol=symbol, period='max')
    processor.run()
    print(f"Completed: {iterations}, Remaining: {num_symbols - iterations}")
    iterations = iterations + 1
    time.sleep(.5)

try:
    folder_name = Path("data")
    shutil.rmtree(folder_name / "N")
except FileNotFoundError:
    print("The folder doesn't exist")
except PermissionError:
    print("Permission denied")
