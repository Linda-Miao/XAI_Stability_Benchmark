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
    # ==========================================================
    # Special case: CICIDS2017
    # Paper 1 already created:
    # - full dataset (2.8M rows after cleaning)
    # - 80 features
    # - 13 classes
    # - StandardScaler applied
    # - train/test split = 70/30
    #
    # Load directly to guarantee identical preprocessing.
    # ==========================================================
    if name == "cicids":
        from sklearn.model_selection import train_test_split
        base = config.DATA_ROOT / "CICIDS-2017" / "processed"

        X_train = np.load(base / "X_train.npy")
        X_test = np.load(base / "X_test.npy")

        y_train = (
            pd.read_csv(base / "y_train.csv")
            .squeeze()
            .to_numpy()
        )

        y_test = (
            pd.read_csv(base / "y_test.csv")
            .squeeze()
            .to_numpy()
        )

        feature_names = (
            pd.read_csv(base / "feature_names.csv")
            .squeeze()
            .tolist()
        )
        # CICIDS is 2.8M rows - too large for 8GB RAM to train 4 models.
        # Subsample to memory-safe sizes. Stratified keeps all 13 classes
        # in proportion. Note in methodology: CICIDS uses a stratified
        # subsample due to memory constraints.
        if len(X_train) > config.CICIDS_TRAIN_CAP:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train,
                train_size=config.CICIDS_TRAIN_CAP,
                random_state=config.SEED,
                stratify=y_train,
            )
        if len(X_test) > config.CICIDS_TEST_CAP:
            X_test, _, y_test, _ = train_test_split(
                X_test, y_test,
                train_size=config.CICIDS_TEST_CAP,
                random_state=config.SEED,
                stratify=y_test,
            )

        return X_train, X_test, y_train, y_test, feature_names


    # ==========================================================
    # Normal CSV datasets
    # ==========================================================
    
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