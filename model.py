import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Enable CUDA error debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#######################
# Device Configuration
#######################
def get_device():
    """Determine the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = get_device()
print(f"Using device: {device}")


#######################
# Data Preprocessing
#######################
def create_one_hot_encoder(categories):
    """Create a one-hot encoder dictionary for categorical variables."""
    unique_categories = list(set(categories))
    return {cat: idx for idx, cat in enumerate(unique_categories)}


def one_hot_encode(category, category_to_idx):
    """Convert a category to one-hot encoded vector."""
    encoding = np.zeros(len(category_to_idx))
    encoding[category_to_idx[category]] = 1
    return encoding


def standardize(data):
    """Standardize numerical data using mean and standard deviation."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + 1e-8)


def load_data_recursively(root_dir):
    """
    Load JSON files recursively from nested directories.

    Args:
        root_dir (str): Root directory path containing sector/industry folders
    Returns:
        list: Combined data from all JSON files
    """
    all_data = []
    root_path = Path(root_dir)

    for json_file in root_path.rglob('*.json'):
        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    all_data.extend(file_data)
                else:
                    all_data.append(file_data)
        except Exception as e:
            print(f"Error reading {json_file}: {str(e)}")

    return all_data


#######################
# Model Definition
#######################
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, categorical_size, dropout=0.5):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layers
        combined_size = hidden_size + categorical_size
        self.fc1 = nn.Linear(combined_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, x_cat):
        h0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        c0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)

        out, _ = self.lstm(x_seq, (h0, c0))
        lstm_out = out[:, -1, :]

        combined = torch.cat((lstm_out, x_cat), dim=1)

        out = self.relu(self.fc1(combined))
        out = self.dropout(out)  # Apply dropout
        out = self.sigmoid(self.fc2(out))
        return torch.clamp(out, min=1e-7, max=1 - 1e-7)


#######################
# Dataset Statistics
#######################
def print_dataset_statistics(data, sequences, labels):
    """Print comprehensive dataset statistics including class imbalance."""
    print("\n=== Dataset Statistics ===")

    # Basic counts
    total_entries = len(data)
    valid_entries = len(sequences)
    skipped_entries = total_entries - valid_entries

    print(f"\nGeneral Statistics:")
    print(f"Total entries: {total_entries}")
    print(f"Valid entries (after processing): {valid_entries}")
    print(f"Skipped entries: {skipped_entries} ({(skipped_entries / total_entries) * 100:.2f}%)")

    # Class distribution
    np_labels = np.array(labels)
    positive_samples = np.sum(np_labels)
    negative_samples = valid_entries - positive_samples

    print(f"\nClass Distribution:")
    print(f"Positive samples (Growth > 2): {positive_samples} ({(positive_samples / valid_entries) * 100:.2f}%)")
    print(f"Negative samples (Growth ≤ 2): {negative_samples} ({(negative_samples / valid_entries) * 100:.2f}%)")
    print(f"Imbalance ratio: 1:{negative_samples / positive_samples:.2f}")

    # Sector and Industry distribution
    sectors = {}
    industries = {}
    growth_by_sector = {}
    growth_by_industry = {}

    for entry in data:
        sector = entry['Sector']
        industry = entry['Industry']

        # Count sectors
        sectors[sector] = sectors.get(sector, 0) + 1

        # Count industries
        industries[industry] = industries.get(industry, 0) + 1

        # Track growth distribution by sector/industry
        if entry['Growth'] is not None:
            growth_by_sector.setdefault(sector, {'high': 0, 'low': 0})
            growth_by_industry.setdefault(industry, {'high': 0, 'low': 0})

            if entry['Growth'] > 2:
                growth_by_sector[sector]['high'] += 1
                growth_by_industry[industry]['high'] += 1
            else:
                growth_by_sector[sector]['low'] += 1
                growth_by_industry[industry]['low'] += 1

    print("\nSector Distribution:")
    for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
        total = growth_by_sector[sector]['high'] + growth_by_sector[sector]['low']
        high_growth_pct = (growth_by_sector[sector]['high'] / total * 100) if total > 0 else 0
        print(f"{sector}: {count} entries ({count / total_entries * 100:.2f}%) - High growth: {high_growth_pct:.1f}%")

    print("\nIndustry Distribution (top 10):")
    sorted_industries = sorted(industries.items(), key=lambda x: x[1], reverse=True)
    for industry, count in sorted_industries[:10]:
        total = growth_by_industry[industry]['high'] + growth_by_industry[industry]['low']
        high_growth_pct = (growth_by_industry[industry]['high'] / total * 100) if total > 0 else 0
        print(f"{industry}: {count} entries ({count / total_entries * 100:.2f}%) - High growth: {high_growth_pct:.1f}%")

    return {
        'total_entries': total_entries,
        'valid_entries': valid_entries,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'imbalance_ratio': negative_samples / positive_samples
    }


#######################
# Data Loading and Processing
#######################
def process_data(root_dir):
    """
    Load and process data for training.

    Returns:
        tuple: Processed training and testing data
    """
    # Load data
    data = load_data_recursively(root_dir)

    # Create encoders
    sectors = [entry['Sector'] for entry in data]
    industries = [entry['Industry'] for entry in data]
    sector_encoder = create_one_hot_encoder(sectors)
    industry_encoder = create_one_hot_encoder(industries)

    # Process sequences and features
    sequences = []
    categorical_features = []
    labels = []
    seq_length = len(data[0]['Williams_R'])

    for entry in data:
        if entry['Growth'] is None:
            continue

        try:
            sequence = []
            for i in range(seq_length):
                values = [
                    entry['market_comp'],
                    entry['Williams_R'][i],
                    entry['macd_diff'][i],
                    entry['rsi'][i],
                    entry['Beta']
                ]

                if any(val is None or val == 'N/A' or isinstance(val, str) for val in values):
                    break

                sequence.append([float(val) for val in values])

            if len(sequence) != seq_length:
                continue

            # Process categorical features
            sector_encoded = one_hot_encode(entry['Sector'], sector_encoder)
            industry_encoded = one_hot_encode(entry['Industry'], industry_encoder)
            categorical = np.concatenate([sector_encoded, industry_encoded])

            sequences.append(sequence)
            categorical_features.append(categorical)
            labels.append(1 if entry['Growth'] > 2 else 0)

        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    print_dataset_statistics(data, sequences, labels)
    return prepare_training_data(sequences, categorical_features, labels)


def prepare_training_data(sequences, categorical_features, labels):
    """Prepare and split data for training."""
    # Convert to numpy arrays
    sequences = np.array(sequences, dtype=np.float32)
    categorical_features = np.array(categorical_features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Standardize numerical features
    for i in range(sequences.shape[2]):
        sequences[:, :, i] = standardize(sequences[:, :, i])

    # Split data (80-20)
    split_idx = int(0.8 * len(sequences))

    # Convert to PyTorch tensors and move to device
    return {
        'train': {
            'seq': torch.FloatTensor(sequences[:split_idx]).to(device),
            'cat': torch.FloatTensor(categorical_features[:split_idx]).to(device),
            'labels': torch.FloatTensor(labels[:split_idx]).to(device)
        },
        'test': {
            'seq': torch.FloatTensor(sequences[split_idx:]).to(device),
            'cat': torch.FloatTensor(categorical_features[split_idx:]).to(device),
            'labels': torch.FloatTensor(labels[split_idx:]).to(device)
        }
    }


def prepare_training_data_v2(sequences, categorical_features, labels):
    # Convert to numpy arrays
    sequences = np.array(sequences, dtype=np.float32)
    categorical_features = np.array(categorical_features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Standardize numerical features
    for i in range(sequences.shape[2]):
        sequences[:, :, i] = standardize(sequences[:, :, i])

    # Split data (70-15-15)
    total = len(sequences)
    train_idx = int(0.7 * total)
    valid_idx = int(0.85 * total)

    return {
        'train': {
            'seq': torch.FloatTensor(sequences[:train_idx]).to(device),
            'cat': torch.FloatTensor(categorical_features[:train_idx]).to(device),
            'labels': torch.FloatTensor(labels[:train_idx]).to(device)
        },
        'valid': {
            'seq': torch.FloatTensor(sequences[train_idx:valid_idx]).to(device),
            'cat': torch.FloatTensor(categorical_features[train_idx:valid_idx]).to(device),
            'labels': torch.FloatTensor(labels[train_idx:valid_idx]).to(device)
        },
        'test': {
            'seq': torch.FloatTensor(sequences[valid_idx:]).to(device),
            'cat': torch.FloatTensor(categorical_features[valid_idx:]).to(device),
            'labels': torch.FloatTensor(labels[valid_idx:]).to(device)
        }
    }

#######################
# Training
#######################
def train_model(model, train_data, num_epochs=250, batch_size=32, max_grad_norm=1.0):
    """Train the model with the specified parameters."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        try:
            outputs = model(train_data['seq'], train_data['cat'])
            loss = criterion(outputs.squeeze(), train_data['labels'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step(loss)  # Step the scheduler based on validation loss

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        except RuntimeError as e:
            print(f"Training error: {str(e)}")
            break


def train_model_v2(model, train_data, valid_data, num_epochs=250, batch_size=32, max_grad_norm=1.0, patience=20):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Create DataLoaders
    train_dataset = TensorDataset(train_data['seq'], train_data['cat'], train_data['labels'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for seq_batch, cat_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(seq_batch, cat_batch).squeeze()
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * seq_batch.size(0)

        epoch_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(valid_data['seq'], valid_data['cat']).squeeze()
            val_loss = criterion(val_outputs, valid_data['labels']).item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, "best_stock_lstm_model.pth")
            epochs_no_improve = 0
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f} (New Best)')
        else:
            epochs_no_improve += 1
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics without sklearn.

    Args:
        y_true: Ground truth labels (tensor)
        y_pred: Predicted labels (tensor)

    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert tensors to boolean masks
    y_true = y_true.bool()
    y_pred = y_pred.bool()

    # Calculate metrics
    true_positive = torch.logical_and(y_pred, y_true).sum().item()
    true_negative = torch.logical_and(~y_pred, ~y_true).sum().item()
    false_positive = torch.logical_and(y_pred, ~y_true).sum().item()
    false_negative = torch.logical_and(~y_pred, y_true).sum().item()

    # Calculate derived metrics
    accuracy = (true_positive + true_negative) / len(y_true)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_positive': true_positive,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_model(model, filepath):
    """
    Save model weights to a file.

    Args:
        model: PyTorch model
        filepath: Path to save the model weights
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.lstm.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'categorical_size': model.fc1.in_features - model.hidden_size
        }
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device):
    """
    Load model weights from a file.

    Args:
        filepath: Path to the saved model weights
        device: Device to load the model on

    Returns:
        StockLSTM: Loaded model
    """
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['model_config']

    model = StockLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        categorical_size=config['categorical_size']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    return model


#######################
# Main Execution
#######################
# Initialize parameters
input_size = 5
hidden_size = 64
num_layers = 2
root_dir = "dataset/data"

# Process data
processed_data = process_data(root_dir)

# Initialize model
categorical_size = processed_data['train']['cat'].shape[1]
model = StockLSTM(input_size, hidden_size, num_layers, categorical_size).to(device)

# Train model
train_model_v2(model, processed_data['train'], num_epochs=100)

# Evaluate model
model.eval()
with torch.no_grad():
    outputs = model(processed_data['test']['seq'], processed_data['test']['cat'])
    predicted = (outputs.squeeze() > 0.5).float()

    # Calculate metrics
    metrics = calculate_metrics(processed_data['test']['labels'], predicted)

    print("\nTest Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {metrics['true_positive']}")
    print(f"True Negatives: {metrics['true_negative']}")
    print(f"False Positives: {metrics['false_positive']}")
    print(f"False Negatives: {metrics['false_negative']}")

# Save the model
save_model(model, "stock_lstm_model.pth")