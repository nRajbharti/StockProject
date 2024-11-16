import torch
from model_v4 import StockTransformer, get_device
import yfinance as yf
import dataset.indicators
import dataset.json_keys
import dataset.screeners as screeners
import numpy as np


class StockPredictor:
    def __init__(self, symbol, model_path="best_stock_transformer_model.pth"):
        self.symbol = symbol
        self.device = get_device()
        self.model = self.load_model(model_path)
        self.days_window = 20  # Same as in dataset.py

    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['model_config']

        model = StockTransformer(
            input_size=5,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            categorical_size=config['categorical_size'],
            num_classes=config['num_classes']
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
        williams_r = dataset.indicators.calculate_williams_r(data)
        macd, signal_line, _ = dataset.indicators.calculate_macd(data)
        rsi = dataset.indicators.calculate_rsi(data)

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
            'Sector': dataset.json_keys.return_sector(sector),
            'Industry': dataset.json_keys.return_industry(industry),
            'Beta': dataset.json_keys.return_beta(beta),
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
            cat_tensor = torch.zeros(1, self.model.fc1.in_features - self.model.d_model).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(seq_tensor, cat_tensor)
                prediction = torch.argmax(outputs, dim=1).item()
                probabilities = torch.softmax(outputs, dim=1)[0]

            # Convert prediction to recommendation
            recommendations = ['Sell', 'Hold', 'Buy']
            recommendation = recommendations[prediction]

            return {
                'recommendation': recommendation,
                'confidence': probabilities[prediction].item(),
                'probabilities': {
                    rec: prob.item() for rec, prob in zip(recommendations, probabilities)
                }
            }

        except Exception as e:
            return f"Error making prediction: {str(e)}"


# Example usage
if __name__ == "__main__":
    symbols = screeners.get_symbols(5)
    for symbol in symbols:
        predictor = StockPredictor(symbol)
        result = predictor.predict()
        print(f"\nPrediction for {symbol}:")
        print(result)