"""
data_loader.py — loads and prepares any dataset for the pipeline.
One function, load_dataset(name), returns ready-to-use train/test data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def load_dataset(name):
    """
    Load one dataset by its short name (e.g. "uavcan").
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    # 1. Find the file path from config
    path = config.DATASETS[name]

    # 2. Read the CSV
    df = pd.read_csv(path)

    # 3. Clean: remove infinite values and rows with missing data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 4. Separate features (X) from the label (y)
    y = df[config.LABEL_COLUMN].values
    X = df.drop(columns=[config.LABEL_COLUMN])
    feature_names = list(X.columns)

    # 5. Scale features (mean 0, std 1)
    X = StandardScaler().fit_transform(X)

    # 6. Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, feature_names