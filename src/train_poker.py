import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
from torch import optim

from src.models.coral_head import coral_predict
from src.poker_dataset import PokerHandDataset
from src.models.model_factory import get_poker_dl_model
from src.util import calculate_metrics

# Configuration
DATA_PATH = 'datasets/poker_hand/poker-hand-training-true.data'
NUM_CLASSES = 10  # Poker hand ranks from 0 to 9
SUIT_EMBEDDING_DIM = 10
RANK_EMBEDDING_DIM = 10
BATCH_SIZE = 64
N_EPOCHS = 50
LEARNING_RATE = 0.001


def main():
    """
    Main function to train and evaluate the poker models.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    print("Loading Poker Hand dataset...")
    full_dataset = PokerHandDataset(DATA_PATH)

    # Split dataset into training and testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Collect all data from DataLoader into single tensors
    all_train_X = []
    all_train_y = []
    for batch_X, batch_y in train_loader:
        all_train_X.append(batch_X)
        all_train_y.append(batch_y)
    X_train_tensor_full = torch.cat(all_train_X)
    y_train_tensor_full = torch.cat(all_train_y)

    all_test_X = []
    all_test_y = []
    for batch_X, batch_y in test_loader:
        all_test_X.append(batch_X)
        all_test_y.append(batch_y)
    X_test_tensor_full = torch.cat(all_test_X)
    y_test_tensor_full = torch.cat(all_test_y)

    # Create dummy pandas DataFrames/Series for dl_trainer functions
    # The actual tensor data will be passed through the .values attribute
    # and then converted back to tensor in dl_trainer functions.
    # This is a workaround to fit the existing dl_trainer API.
    X_train_df = pd.DataFrame({'data': X_train_tensor_full.tolist()})
    y_train_series = pd.Series(y_train_tensor_full.numpy())

    X_test_df = pd.DataFrame({'data': X_test_tensor_full.tolist()})
    y_test_series = pd.Series(y_test_tensor_full.numpy())

    print("Initializing PokerCNNModel...")
    model_kwargs = {
        'suit_embedding_dim': SUIT_EMBEDDING_DIM,
        'rank_embedding_dim': RANK_EMBEDDING_DIM,
        'num_classes': NUM_CLASSES
    }
    model = get_poker_dl_model('poker_cnn', **model_kwargs)
    model.to(device)

    print("Training PokerCNNModel...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(N_EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = model.loss_fn(outputs, labels)  # Use the loss_fn defined in PokerCNNModel

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {loss.item():.4f}')

    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_df['data'].tolist(), dtype=torch.long).to(device)
        logits = model(X_test_tensor)
        y_pred = coral_predict(logits).cpu().numpy()

    print("Calculating metrics...")
    metrics = calculate_metrics(y_test_series.values, y_pred)

    print("\n--- PokerCNNModel Evaluation ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nInitializing PokerLSTMModel...")
    model_kwargs = {
        'suit_embedding_dim': SUIT_EMBEDDING_DIM,
        'rank_embedding_dim': RANK_EMBEDDING_DIM,
        'num_classes': NUM_CLASSES
    }
    model_lstm = get_poker_dl_model('poker_lstm', **model_kwargs)
    model_lstm.to(device)

    print("Training PokerLSTMModel...")
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE)

    model_lstm.train()
    for epoch in range(N_EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_lstm(inputs)
            loss = model_lstm.loss_fn(outputs, labels)

            optimizer_lstm.zero_grad()
            loss.backward()
            optimizer_lstm.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {loss.item():.4f}')

    print("Making predictions with PokerLSTMModel...")
    model_lstm.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_df['data'].tolist(), dtype=torch.long).to(device)
        logits = model_lstm(X_test_tensor)
        y_pred_lstm = coral_predict(logits).cpu().numpy()

    print("Calculating metrics for PokerLSTMModel...")
    metrics_lstm = calculate_metrics(y_test_series.values, y_pred_lstm)

    print("\n--- PokerLSTMModel Evaluation ---")
    for metric_name, value in metrics_lstm.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nInitializing PokerTransformerModel...")
    model_kwargs = {
        'suit_embedding_dim': SUIT_EMBEDDING_DIM,
        'rank_embedding_dim': RANK_EMBEDDING_DIM,
        'num_classes': NUM_CLASSES,
        'nhead': 2,
        'nlayers': 2,
        'd_hid': 128,
    }
    model_transformer = get_poker_dl_model('poker_transformer', **model_kwargs)
    model_transformer.to(device)

    print("Training PokerTransformerModel...")
    optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=LEARNING_RATE)

    model_transformer.train()
    for epoch in range(N_EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_transformer(inputs)
            loss = model_transformer.loss_fn(outputs, labels)

            optimizer_transformer.zero_grad()
            loss.backward()
            optimizer_transformer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {loss.item():.4f}')

    print("Making predictions with PokerTransformerModel...")
    model_transformer.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_df['data'].tolist(), dtype=torch.long).to(device)
        logits = model_transformer(X_test_tensor)
        y_pred_transformer = coral_predict(logits).cpu().numpy()

    print("Calculating metrics for PokerTransformerModel...")
    metrics_transformer = calculate_metrics(y_test_series.values, y_pred_transformer)

    print("\n--- PokerTransformerModel Evaluation ---")
    for metric_name, value in metrics_transformer.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
