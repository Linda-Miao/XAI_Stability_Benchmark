"""
config.py

Purpose:
This file stores all project settings in one location so they can be
easily updated without changing multiple files.

Contents:
- Dataset folder locations
- Dataset file paths
- Label column name
- Random seed value
- Train/test split size
- Number of top features to select
"""

# Import Path from pathlib.
# Path creates file and folder paths that work across different operating systems.
from pathlib import Path


# Path.home() returns the current user's home directory.
# "/" is used to safely join folders together.
DATA_ROOT = Path.home() / "Desktop" / "replication_studies" / "XAI_Feature_Selection"


# Dictionary containing all dataset file locations.
#
# Key   = short dataset name used in the code
# Value = full path to the processed CSV file
#
# Example:
# DATASETS["uavcan"]
# returns:
# Desktop/replication_studies/XAI_Feature_Selection/UAVCAN/processed_UAVCAN.csv
DATASETS = {
    "uavcan":      DATA_ROOT / "UAVCAN" / "processed_UAVCAN.csv",
    "isot":        DATA_ROOT / "ISOT" / "processed_ISOT.csv",
    "uav_attack":  DATA_ROOT / "UAV_Attack" / "processed_UAV_Attack.csv",
     
    # CICIDS uses processed files, not this path directly
    "cicids": DATA_ROOT / "CICIDS-2017" / "processed",
    "uav_cyber": DATA_ROOT / "UAV-Cyber-Physical" / "processed",

    # UAVIDS-2025
    "uavids": DATA_ROOT / "UAVIDS-2025" / "UAVIDS-2025.csv",
}


# Name of the target column used for machine learning.
# The model learns to predict values in this column.
LABEL_COLUMN = "label"


# Random seed used to make results reproducible.
# Using the same seed helps generate the same train/test split
# and model results each time the code runs.
SEED = 42


# Percentage of data reserved for testing.
# 0.3 = 30% test data
# 0.7 = 70% training data
TEST_SIZE = 0.3


# Number of top-ranked features to keep after feature selection.
# Example:
# If TOP_K = 5, only the 5 most important features are retained.
TOP_K = 5

# Which label number means "normal/benign" in each dataset
# (confirmed from Paper 1 notebooks)
BENIGN_LABEL = {
    "uavcan": 1,       # Normal = 1
    "isot": 0,         # Benign = 0
    "uav_attack": 2,   # benign = 2
    "cicids": 0,       # BENIGN = 0
    "uav_cyber": 2,     # benign = 2
    "uavids": 2,     # "Normal Traffic" encodes to 2 (alphabetical order)
}

# SHAP is too slow to run on full test sets:
# SHAP sample sized (SHAP is slow, so we explain a subset) sizes match paper 1
SHAP_SAMPLES_RF = 500
SHAP_SAMPLES_CNN = 100
SHAP_SAMPLES_AE = 50
SHAP_BACKGROUND = 50

# XAI test-set cap: XAI methods (especially PI, LIME) run on at most this many
# test rows, subsampled reproducibly. Big datasets (CICIDS: 848K test rows)
# would otherwise take hours. ML metrics still use the full test set.
XAI_TEST_CAP = 3000

# CICIDS (2.8M rows) is subsampled to 150K train / 50K test (stratified) because
# an 8GB machine (~3GB available) cannot train 4 models on the full dataset.
# Both train and test are capped; 50K test is still a large, representative sample.
CICIDS_TRAIN_CAP = 150000
CICIDS_TEST_CAP = 50000

# Model training hyperparameters
CNN_EPOCHS = 5
CNN_BATCH = 256
AE_EPOCHS = 10
AE_BATCH = 512