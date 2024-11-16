import torch
import yfinance as yf
import dataset.indicators as indicators
import dataset.json_keys as json_keys
import dataset.screeners as screeners
import numpy as np
from model_v2 import StockLSTM, get_device


class StockPredictorV2:
    def __init__(self, symbol, model_path="best_stock_binary_lstm_model.pth"):
        self.symbol = symbol
        self.device = get_device()
        self.model = self.load_model(model_path)
        self.days_window = 20  # Same as in dataset.py

    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['model_config']

        model = StockLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            categorical_size=config['categorical_size']
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def get_stock_data(self):
        """Download and process stock data"""
        # Download stock data
        data = yf.download(self.symbol, period="1y")
        ticker = yf.Ticker(self.symbol)
        stock_info = ticker.info

        # Calculate technical indicators
        williams_r = indicators.calculate_williams_r(data)
        macd, signal_line, _ = indicators.calculate_macd(data)
        rsi = indicators.calculate_rsi(data)

        # Get the most recent window of data
        williams_r_values = williams_r.iloc[-self.days_window:].values.flatten().tolist()
        macd_diff_values = (macd.iloc[-self.days_window:] -
                            signal_line.iloc[-self.days_window:]).values.flatten().tolist()
        rsi_values = rsi.iloc[-self.days_window:].values.flatten().tolist()

        # Get categorical features
        sector = stock_info.get('sector', 'N/A')
        industry = stock_info.get('industry', 'N/A')
        beta = stock_info.get('beta', 'N/A')

        return {
            'Williams_R': williams_r_values,
            'macd_diff': macd_diff_values,
            'rsi': rsi_values,
            'Sector': json_keys.return_sector(sector),
            'Industry': json_keys.return_industry(industry),
            'Beta': json_keys.return_beta(beta),
            'market_comp': 1  # This will be estimated or could be calculated if needed
        }

    def standardize(self, data):
        """Standardize numerical data"""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)

    def predict(self):
        """Make prediction for the stock"""
        try:
            # Get processed data
            data = self.get_stock_data()

            # Prepare sequential features
            seq_features = np.array([
                data['Williams_R'],
                data['macd_diff'],
                data['rsi'],
                [data['Beta']] * self.days_window,
                [data['market_comp']] * self.days_window
            ], dtype=np.float32).T

            # Standardize sequential features
            for i in range(seq_features.shape[1]):
                seq_features[:, i] = self.standardize(seq_features[:, i])

            # Convert to tensor and add batch dimension
            seq_tensor = torch.FloatTensor(seq_features).unsqueeze(0).to(self.device)

            # Prepare categorical features (you'll need to implement one-hot encoding similar to the training data)
            # This is a placeholder - you'll need to match the exact categorical encoding used in training
            cat_tensor = torch.zeros(1, self.model.fc1.in_features - self.model.hidden_size).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(seq_tensor, cat_tensor)
                probability = output.item()
                prediction = "Buy" if probability > 0.5 else "Hold/Sell"

            return {
                'recommendation': prediction,
                'confidence': probability if prediction == "Buy" else 1 - probability,
                'growth_probability': probability
            }

        except Exception as e:
            return f"Error making prediction: {str(e)}"


# Example usage
if __name__ == "__main__":
    symbols = screeners.get_symbols(5)
    symbols = symbols[0:10]
    # symbol = "INST"  # Replace with your desired stock symbol
    for symbol in symbols:
        predictor = StockPredictorV2(symbol)
        result = predictor.predict()
        print(f"\nPrediction for {symbol}:")
        print(result)