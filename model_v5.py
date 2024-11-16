import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    unique_categories = sorted(list(set(categories)))
    return {cat: idx for idx, cat in enumerate(unique_categories)}


def one_hot_encode(category, category_to_idx):
    """Convert a category to one-hot encoded vector."""
    encoding = np.zeros(len(category_to_idx))
    if category in category_to_idx:
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
# Positional Encoding
#######################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Injects positional information into the input embeddings.

        Args:
            d_model (int): Embedding dimension
            max_len (int): Maximum length of sequences
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Positionally encoded embeddings
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


#######################
# Hybrid LSTM-Transformer Model Definition
#######################
class StockLSTMTransformer(nn.Module):
    """
    Hybrid model combining LSTM and Transformer layers for stock prediction.
    """

    def __init__(self, input_size, hidden_size, num_lstm_layers, d_model, nhead, num_transformer_layers,
                 dim_feedforward, categorical_size, num_classes, dropout=0.5):
        super(StockLSTMTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first=True, dropout=dropout)

        # Transformer components
        self.input_proj = nn.Linear(hidden_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_transformer_layers)

        # Fully connected layers
        combined_size = d_model + categorical_size
        self.fc1 = nn.Linear(combined_size, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x_seq, x_cat):
        # LSTM processing
        h0 = torch.zeros(self.num_lstm_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        c0 = torch.zeros(self.num_lstm_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        lstm_out, _ = self.lstm(x_seq, (h0, c0))  # Shape: (batch, seq_len, hidden_size)

        # Transformer processing
        transformer_input = self.input_proj(lstm_out)  # Shape: (batch, seq_len, d_model)
        transformer_input = self.pos_encoder(transformer_input)
        transformer_input = transformer_input.permute(1, 0, 2)  # (seq_len, batch, d_model)
        transformer_out = self.transformer_encoder(transformer_input)  # (seq_len, batch, d_model)
        transformer_out = transformer_out[-1, :, :]  # (batch, d_model)

        # Combine with categorical features
        combined = torch.cat((transformer_out, x_cat), dim=1)

        # Final predictions
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


#######################
# Label Encoding
#######################
def get_class_label(growth):
    """
    Convert a growth value to a class label.

    Classes:
        2: Buy (Growth > 4)
        1: Hold (-3.75 <= Growth <= 4)
        0: Sell (Growth < -3.75)
    """
    if growth > 4:
        return 2  # Buy
    elif -3.75 <= growth <= 4:
        return 1  # Hold
    else:
        return 0  # Strong Sell


#######################
# Data Loading and Processing
#######################
def process_data(root_dir):
    """
    Load and process data for training.

    Returns:
        dict: Processed training, validation, and testing data
    """
    data = load_data_recursively(root_dir)

    sectors = [entry['Sector'] for entry in data if 'Sector' in entry]
    industries = [entry['Industry'] for entry in data if 'Industry' in entry]
    sector_encoder = create_one_hot_encoder(sectors)
    industry_encoder = create_one_hot_encoder(industries)

    sequences = []
    categorical_features = []
    labels = []
    seq_length = 0
    for entry in data:
        if 'Williams_R' in entry and isinstance(entry['Williams_R'], list):
            seq_length = len(entry['Williams_R'])
            break

    if seq_length == 0:
        raise ValueError("No valid sequence data found.")

    for entry in data:
        growth = entry.get('Growth')
        if growth is None:
            continue

        try:
            sequence = []
            for i in range(seq_length):
                values = [
                    entry.get('market_comp', 0),
                    entry['Williams_R'][i],
                    entry['macd_diff'][i],
                    entry['rsi'][i],
                    entry.get('Beta', 0)
                ]

                if any(val is None or val == 'N/A' or isinstance(val, str) for val in values):
                    raise ValueError("Invalid numerical value.")

                sequence.append([float(val) for val in values])

            if len(sequence) != seq_length:
                continue

            sector = entry.get('Sector', 'Unknown')
            industry = entry.get('Industry', 'Unknown')
            sector_encoded = one_hot_encode(sector, sector_encoder)
            industry_encoded = one_hot_encode(industry, industry_encoder)
            categorical = np.concatenate([sector_encoded, industry_encoded])

            sequences.append(sequence)
            categorical_features.append(categorical)
            labels.append(get_class_label(entry['Growth']))

        except Exception as e:
            print(f"Error processing entry: {e}")
            continue

    print_dataset_statistics(data, sequences, labels)
    return prepare_training_data(sequences, categorical_features, labels)


def print_dataset_statistics(data, sequences, labels):
    """Print comprehensive dataset statistics including class imbalance."""
    print("\n=== Dataset Statistics ===")

    total_entries = len(data)
    valid_entries = len(sequences)
    skipped_entries = total_entries - valid_entries

    print(f"\nGeneral Statistics:")
    print(f"Total entries: {total_entries}")
    print(f"Valid entries (after processing): {valid_entries}")
    print(f"Skipped entries: {skipped_entries} ({(skipped_entries / total_entries) * 100:.2f}%)")

    np_labels = np.array(labels)
    class_counts = np.bincount(np_labels, minlength=3)
    print(f"\nClass Distribution:")
    class_names = ["Sell", "Hold", "Buy"]
    for idx, count in enumerate(class_counts):
        pct = (count / valid_entries) * 100 if valid_entries > 0 else 0
        print(f"{class_names[idx]} ({idx}): {count} ({pct:.2f}%)")

    sectors = {}
    industries = {}
    growth_by_sector = {}
    growth_by_industry = {}

    for entry in data:
        sector = entry.get('Sector', 'Unknown')
        industry = entry.get('Industry', 'Unknown')
        growth = entry.get('Growth')

        sectors[sector] = sectors.get(sector, 0) + 1
        industries[industry] = industries.get(industry, 0) + 1

        if growth is not None:
            growth_by_sector.setdefault(sector, {'high': 0, 'low': 0})
            growth_by_industry.setdefault(industry, {'high': 0, 'low': 0})

            if growth > 4:
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


def prepare_training_data(sequences, categorical_features, labels):
    """Prepare and split data for training, validation, and testing."""
    sequences = np.array(sequences, dtype=np.float32)
    categorical_features = np.array(categorical_features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    for i in range(sequences.shape[2]):
        sequences[:, :, i] = standardize(sequences[:, :, i])

    permutation = np.random.permutation(len(sequences))
    sequences = sequences[permutation]
    categorical_features = categorical_features[permutation]
    labels = labels[permutation]

    total = len(sequences)
    train_idx = int(0.7 * total)
    valid_idx = int(0.85 * total)

    return {
        'train': {
            'seq': torch.FloatTensor(sequences[:train_idx]).to(device),
            'cat': torch.FloatTensor(categorical_features[:train_idx]).to(device),
            'labels': torch.LongTensor(labels[:train_idx]).to(device)
        },
        'valid': {
            'seq': torch.FloatTensor(sequences[train_idx:valid_idx]).to(device),
            'cat': torch.FloatTensor(categorical_features[train_idx:valid_idx]).to(device),
            'labels': torch.LongTensor(labels[train_idx:valid_idx]).to(device)
        },
        'test': {
            'seq': torch.FloatTensor(sequences[valid_idx:]).to(device),
            'cat': torch.FloatTensor(categorical_features[valid_idx:]).to(device),
            'labels': torch.LongTensor(labels[valid_idx:]).to(device)
        }
    }


#######################
# Training
#######################

def train_model(model, train_data, valid_data, num_epochs=250, batch_size=32, max_grad_norm=1.0, patience=20,
               learning_rate=0.001):
    """
    Train the model with the specified parameters, incorporating early stopping and learning rate scheduling.

    Args:
        model (nn.Module): The neural network model to train.
        train_data (dict): Training data containing 'seq', 'cat', and 'labels'.
        valid_data (dict): Validation data containing 'seq', 'cat', and 'labels'.
        num_epochs (int): Maximum number of epochs to train.
        batch_size (int): Number of samples per batch.
        max_grad_norm (float): Maximum norm for gradient clipping.
        patience (int): Number of epochs to wait for improvement before stopping.
        learning_rate (float): Initial learning rate for the optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    train_dataset = TensorDataset(train_data['seq'], train_data['cat'], train_data['labels'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_data['seq'], valid_data['cat'], valid_data['labels'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for seq_batch, cat_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(seq_batch, cat_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * seq_batch.size(0)

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_batch, cat_batch, labels_batch in valid_loader:
                outputs = model(seq_batch, cat_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * seq_batch.size(0)

        val_loss /= len(valid_loader.dataset)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, "best_stock_lstm_transformer_model.pth")
            epochs_no_improve = 0
            print(f'--> New best model saved with validation loss {val_loss:.4f}')
        else:
            epochs_no_improve += 1
            print(f'--> No improvement for {epochs_no_improve} epoch(s)')

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


def calculate_metrics(y_true, y_pred, num_classes=3):
    """
    Calculate classification metrics without sklearn.

    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred (torch.Tensor): Predicted labels.
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary containing various metrics.
    """
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    accuracy = torch.trace(confusion_matrix).item() / confusion_matrix.sum().item()

    precision = []
    recall = []
    f1 = []
    for cls in range(num_classes):
        tp = confusion_matrix[cls, cls].item()
        fp = confusion_matrix[:, cls].sum().item() - tp
        fn = confusion_matrix[cls, :].sum().item() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)

    macro_precision = sum(precision) / num_classes
    macro_recall = sum(recall) / num_classes
    macro_f1 = sum(f1) / num_classes

    return {
        'confusion_matrix': confusion_matrix.tolist(),
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def save_model(model, filepath):
    """Save model weights to a file."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.lstm.input_size,
            'hidden_size': model.hidden_size,
            'num_lstm_layers': model.num_lstm_layers,
            'd_model': model.transformer_encoder.layers[0].linear1.in_features,
            'nhead': model.transformer_encoder.layers[0].self_attn.num_heads,
            'num_transformer_layers': len(model.transformer_encoder.layers),
            'dim_feedforward': model.transformer_encoder.layers[0].linear1.out_features,
            'categorical_size': model.fc1.in_features - model.transformer_encoder.layers[0].linear1.in_features,
            'num_classes': model.fc2.out_features
        }
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device):
    """Load model weights from a file."""
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['model_config']

    model = StockLSTMTransformer(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_lstm_layers=config['num_lstm_layers'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_transformer_layers=config['num_transformer_layers'],
        dim_feedforward=config['dim_feedforward'],
        categorical_size=config['categorical_size'],
        num_classes=config['num_classes']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    return model


#######################
# Main Execution
#######################

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    input_size = 5
    hidden_size = 64
    num_lstm_layers = 2
    d_model = 64
    nhead = 8
    num_transformer_layers = 2
    dim_feedforward = 256
    dropout_rate = 0.5
    num_classes = 3
    root_dir = "dataset/data"

    processed_data = process_data(root_dir)

    categorical_size = processed_data['train']['cat'].shape[1]
    model = StockLSTMTransformer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        dim_feedforward=dim_feedforward,
        categorical_size=categorical_size,
        num_classes=num_classes,
        dropout=dropout_rate
    ).to(device)

    train_model(
        model,
        train_data=processed_data['train'],
        valid_data=processed_data['valid'],
        num_epochs=250,
        batch_size=32,
        max_grad_norm=1.0,
        patience=20,
        learning_rate=0.001
    )

    best_model = load_model("best_stock_lstm_transformer_model.pth", device)

    best_model.eval()
    with torch.no_grad():
        outputs = best_model(processed_data['test']['seq'], processed_data['test']['cat'])
        predicted = torch.argmax(outputs, dim=1)
        true_labels = processed_data['test']['labels']

        metrics = calculate_metrics(true_labels.cpu(), predicted.cpu(), num_classes=num_classes)

        print("\nTest Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix:")
        for idx, row in enumerate(metrics['confusion_matrix']):
            print(f"True {idx}: {row}")
        print("\nPer-Class Precision:")
        for idx, prec in enumerate(metrics['precision_per_class']):
            print(f"Class {idx}: {prec:.4f}")
        print("\nPer-Class Recall:")
        for idx, rec in enumerate(metrics['recall_per_class']):
            print(f"Class {idx}: {rec:.4f}")
        print("\nPer-Class F1 Score:")
        for idx, f1_score in enumerate(metrics['f1_per_class']):
            print(f"Class {idx}: {f1_score:.4f}")
        print(f"\nMacro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")

    save_model(best_model, "best_stock_lstm_transformer_model_final.pth")