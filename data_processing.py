import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path='dataset/Housing.csv', test_size=0.2, random_state=42):
    """
    Loads dataset and prepocesses missing values and categoricals.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Map binary categories
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Map furnishingstatus ordinal
    df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

    # Clean missing or dropna if any
    df = df.dropna()

    X = df.drop('price', axis=1)
    y = df['price'].values
    feature_names = X.columns
    N, D = X.shape

    print(f"Dataset Loaded. Features: {D}, Samples: {N}")

    # Standard scale features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Scale target to avoid huge MSE values and help learning
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, feature_names, scaler_X, scaler_y
