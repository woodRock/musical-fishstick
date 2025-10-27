"""Module for training and prediction functions for various deep learning models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from coral import coral_loss, coral_predict
from corn_loss import corn_loss
from emd_loss import emd_loss
from pom_scratch import pom_loss, pom_predict
from adjacent_model import adjacent_loss, adjacent_predict

def train_model(model, X_train, y_train, n_epochs=20, batch_size=32, lr=0.001):
    """Handles the training loop for a standard PyTorch classification model.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        nn.Module: The trained PyTorch model.
    """
    # Convert pandas data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train() # Set the model to training mode
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

def predict(model, X_test):
    """Makes predictions with a trained PyTorch model.

    Args:
        model (nn.Module): The trained PyTorch model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Predicted class labels.
    """
    model.eval() # Set the model to evaluation mode
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.numpy()


def train_coral_model(model, X_train, y_train, n_epochs=20, batch_size=32, lr=0.001):
    """Handles the training loop for a CORAL model.

    Args:
        model (nn.Module): The CORAL model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        nn.Module: The trained CORAL model.
    """
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = coral_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

def predict_coral(model, X_test):
    """Makes predictions with a trained CORAL model.

    Args:
        model (nn.Module): The trained CORAL model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Predicted class labels.
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test_tensor)
        predicted = coral_predict(logits)
    return predicted.numpy()


def train_pom_scratch_model(model, X_train, y_train, num_classes, n_epochs=20, batch_size=32, lr=0.01):
    """Handles the training loop for the scratch POM model.

    Args:
        model (nn.Module): The POM model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        num_classes (int): Total number of classes.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.

    Returns:
        nn.Module: The trained POM model.
    """
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            cum_probs = model(inputs)
            loss = pom_loss(cum_probs, labels, num_classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

def predict_pom_scratch(model, X_test):
    """Makes predictions with a trained scratch POM model.

    Args:
        model (nn.Module): The trained POM model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Predicted class labels.
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        cum_probs = model(X_test_tensor)
        predicted = pom_predict(cum_probs)
    return predicted.numpy()


def train_adjacent_model(model, X_train, y_train, n_epochs=20, batch_size=32, lr=0.01):
    """Handles the training loop for the scratch Adjacent Category model.

    Args:
        model (nn.Module): The Adjacent Category model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.

    Returns:
        nn.Module: The trained Adjacent Category model.
    """
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            probs = model(inputs)
            loss = adjacent_loss(probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

def predict_adjacent(model, X_test):
    """Makes predictions with a trained scratch Adjacent Category model.

    Args:
        model (nn.Module): The trained Adjacent Category model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Predicted class labels.
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_test_tensor)
        predicted = adjacent_predict(probs)
    return predicted.numpy()


def train_emd_model(model, X_train, y_train, n_epochs=20, batch_size=32, lr=0.001):
    """Handles the training loop for a model using EMD loss.

    Args:
        model (nn.Module): The model to train with EMD loss.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        nn.Module: The trained model.
    """
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = emd_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model


def train_corn_model(model, X_train, y_train, num_classes, n_epochs=20, batch_size=32, lr=0.001):
    """Handles the training loop for a CORN model.

    Args:
        model (nn.Module): The CORN model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        num_classes (int): Total number of classes.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        nn.Module: The trained CORN model.
    """
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = corn_loss(outputs, labels, num_classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model
