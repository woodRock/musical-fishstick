import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score
)

def get_serializable_params(model):
    """Gets JSON-serializable parameters from a scikit-learn model."""
    params = model.get_params()
    serializable_params = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            serializable_params[key] = value
        else:
            serializable_params[key] = str(value)
    return serializable_params


def calculate_metrics(y_true, y_pred, is_regression=False):
    """Calculates and returns a dictionary of evaluation metrics."""
    if is_regression:
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
        }
    else:
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "QWK": cohen_kappa_score(y_true, y_pred, weights='quadratic'),
            "MSE": mean_squared_error(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Acc.": balanced_accuracy_score(y_true, y_pred, adjusted=True)
        }

def load_car_evaluation():
    """Loads and preprocesses the Car Evaluation dataset."""
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    category_orders = [
        ['low', 'med', 'high', 'vhigh'],
        ['low', 'med', 'high', 'vhigh'],
        ['2', '3', '4', '5more'],
        ['2', '4', 'more'],
        ['small', 'med', 'big'],
        ['low', 'med', 'high'],
        ['unacc', 'acc', 'good', 'vgood']
    ]
    file_path = '/Users/woodj/Desktop/musical-fishstick/datasets/car_evaluation/car.data'
    df = pd.read_csv(file_path, header=None, names=cols)
    encoder = OrdinalEncoder(categories=category_orders)
    df_encoded = pd.DataFrame(encoder.fit_transform(df), columns=cols)
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']
    return X, y, False # Returns X, y, is_regression

def load_wine_quality():
    """Loads and preprocesses the Wine Quality dataset."""
    red_wine_path = '/Users/woodj/Desktop/musical-fishstick/datasets/wine_quality/winequality-red.csv'
    white_wine_path = '/Users/woodj/Desktop/musical-fishstick/datasets/wine_quality/winequality-white.csv'
    df_red = pd.read_csv(red_wine_path, sep=';')
    df_white = pd.read_csv(white_wine_path, sep=';')
    df = pd.concat([df_red, df_white], ignore_index=True)
    X = df.drop('quality', axis=1)
    y = df['quality']
    # Scale features for models that are sensitive to feature scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y, False

def load_boston_housing():
    """Loads the Boston Housing dataset and converts it to an ordinal problem."""
    file_path = '/Users/woodj/Desktop/musical-fishstick/datasets/boston_housing/boston_housing.csv'
    cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=cols)
    
    # Convert regression target to ordinal categories using quartiles
    labels = ['low', 'medium', 'high', 'very high']
    df['MEDV_ORDINAL'] = pd.qcut(df['MEDV'], q=4, labels=labels)

    X = df.drop(['MEDV', 'MEDV_ORDINAL'], axis=1)
    y_categorical = df['MEDV_ORDINAL']

    # Encode the new ordinal target
    encoder = OrdinalEncoder(categories=[labels])
    y = encoder.fit_transform(y_categorical.to_numpy().reshape(-1, 1))
    y = pd.Series(y.flatten(), name='MEDV_ORDINAL').astype(int)

    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y, False # Now a classification problem

def load_poker_hand():
    """Loads and preprocesses the Poker Hand dataset."""
    file_path = '/Users/woodj/Desktop/musical-fishstick/datasets/poker_hand/poker-hand-training-true.data'
    cols = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
    df = pd.read_csv(file_path, header=None, names=cols)
    X = df.drop('CLASS', axis=1)
    y = df['CLASS'].astype(int)
    # Remap labels to be zero-indexed for PyTorch
    y = y - y.min() 
    y = y.astype(int)
    # Scale features for models that are sensitive to feature scaling
    return X, y, False


def load_dataset(name):
    """Dispatcher to load the specified dataset."""
    if name.lower() == 'car':
        return load_car_evaluation()
    elif name.lower() == 'wine':
        return load_wine_quality()
    elif name.lower() == 'boston':
        return load_boston_housing()
    elif name.lower() == 'poker':
        return load_poker_hand()
    else:
        raise ValueError(f"Unknown dataset: {name}")
