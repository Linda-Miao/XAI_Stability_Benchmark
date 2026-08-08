"""
PURPOSE:
This file creates and trains the four machine learning models used in Paper 2
for Intrusion Detection System (IDS) benchmarking.

MODELS:
1. Random Forest (RF)
2. Convolutional Neural Network (CNN)
3. Autoencoder (AE)
4. Isolation Forest (IF)

RESEARCH GOAL:
Compare how different model types behave when detecting cyber attacks,
and how explainable AI (XAI) methods interpret their decisions.

MODEL CATEGORIES:

SUPERVISED MODELS
- RF
- CNN

These models learn from BOTH:
    normal traffic + attack traffic

UNSUPERVISED MODELS
- AE
- IF

These models learn ONLY:
    normal traffic

They detect anomalies by finding behavior that does NOT look normal.

PAPER 1 SETTINGS:
All hyperparameters match Paper 1 so results remain comparable.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import numpy as np  # NumPy = numerical operations and arrays
import tensorflow as tf  # TensorFlow = deep learning library
import config # shared project settings (must import before using config.SEED)

# Set seeds for reproducible neural network training
# RF/IF use random_state=SEED; CNN/AE need NumPy + TenshorFlow globle seeds
# for reproducible weight initialization, dropout, and batch shuffling.
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

from tensorflow.keras.models import Model  # Build neural network model

# Layers used to build CNN and Autoencoder
from tensorflow.keras.layers import (
    Input,           # Input layer (data enters model here)
    Conv1D,          # Convolution layer (pattern detection)
    MaxPooling1D,    # Reduce feature size
    Flatten,         # Convert multi-dim data to 1D vector
    Dense,           # Fully connected layer (learn patterns)
    Dropout          # Prevent overfitting
)

# Traditional machine learning models
from sklearn.ensemble import RandomForestClassifier, IsolationForest

import config  # shared project settings (like SEED)


# ==========================================================
# 1. RANDOM FOREST
# ==========================================================

# TYPE: Supervised Learning
# LEARNS FROM: Normal traffic + attack traffic
# STRENGTH: Very stable and interpretable model
# PAPER 1 RESULT: One of the strongest performing models
# PURPOSE: Learn decision rules that separate attacks from normal traffic

def build_random_forest(X_train, y_train):

    rf = RandomForestClassifier(
        n_estimators=100,          # number of trees
        random_state=config.SEED,  # reproducible results
        n_jobs=-1                  # use all CPU cores
    )

    rf.fit(X_train, y_train)  # train model using labeled data
    return rf


# ==========================================================
# 2. CNN (1D Convolutional Neural Network)
# ==========================================================

# TYPE: Supervised Learning
# LEARNS FROM: Normal traffic + attack traffic
# STRENGTH: Finds hidden patterns that RF may miss
# PURPOSE: Learn complex relationships among features

def build_cnn(X_train, y_train):

    input_dim = X_train.shape[1]            # number of features
    n_classes = len(np.unique(y_train))     # number of attack classes

    # CNN expects 3D input:
    # (samples, features, channels)
    X_train_cnn = X_train.reshape(-1, input_dim, 1)

    # Input layer
    inp = Input(shape=(input_dim, 1))

    # ======================================================
    # Conv1D explanation (padding example)
    #
    # original: 10 20 30 40 50
    # padding:  0 10 20 30 40 50 0
    #
    # allows CNN to see edges properly
    # ======================================================

    # First convolution layer (pattern detection)
    x = Conv1D(64, 3, activation="relu", padding="same")(inp)

    # Second convolution layer
    x = Conv1D(128, 3, activation="relu", padding="same")(x)

    # Downsample feature maps.
    x = MaxPooling1D(2)(x)

    # Flatten to single vector
    x = Flatten()(x)

    # Dense layer = Learns higher-level feature representations.
    x = Dense(128, activation="relu")(x)

    # Dropout = randomly disables 30% neurons to reduce overfitting
    x = Dropout(0.3)(x)

    # ======================================================
    # OUTPUT LAYER (Softmax)
    #
    # Example raw scores:
    # normal = 2.3
    # DDoS = 7.1
    # spoofing = 1.8
    #
    # Softmax converts them to probabilities:
    # normal = 0.01
    # DDoS = 0.92
    # spoofing = 0.01
    # ======================================================

    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        loss="sparse_categorical_crossentropy",  # multi-class classification
        optimizer="adam",                        # adaptive learning algorithm
        metrics=["accuracy"]
    )

    model.fit(
        X_train_cnn,
        y_train,
        # from config (was 256); number of samples per update step
        batch_size=config.CNN_BATCH, 
        # from config (was 5); number of full passes through dataset
        epochs=config.CNN_EPOCHS,  
        validation_split=0.1, # use 10% of training data for validation to monitor performance during training. it like train-> pretest->test
        shuffle=True, # explicit
        verbose=1          # show training progress
    )

    return model


# ==========================================================
# 3. AUTOENCODER
# ==========================================================

# TYPE: Unsupervised Learning
# LEARNS FROM: ONLY normal traffic
# PURPOSE: Learn what "normal" looks like
# IDEA: High reconstruction error = anomaly
# PAPER 1 FINDING: performance varies across datasets

def build_autoencoder(X_train, y_train, dataset_name):
    input_dim = X_train.shape[1]              # number of features
    benign = config.BENIGN_LABEL[dataset_name]

    # Keep ONLY normal traffic
    X_normal = X_train[y_train == benign]

    # Safety check: if no benign samples found, the benign label is likely
    # wrong (e.g. set to a text value when labels are numeric). This caught
    # a real bug on UAVIDS-2025 where BENIGN_LABEL was "Normal Traffic" (text)
    # instead of the encoded number.
    if len(X_normal) == 0:
        raise ValueError(
            f"No benign samples for '{dataset_name}': BENIGN_LABEL is {benign} "
            f"but training labels are {np.unique(y_train)}. Check BENIGN_LABEL matches the encoded label."
        )

    inp = Input(shape=(input_dim,))
    # Encoder (compress data)
    x = Dense(32, activation="relu")(inp)
    # Bottleneck (compressed representation)
    x = Dense(16, activation="relu")(x)
    # Decoder (reconstruct data)
    x = Dense(32, activation="relu")(x)
    # Output = Reconstruct the original input features.
    out = Dense(input_dim, activation="linear")(x)
    ae = Model(inputs=inp, outputs=out)
    ae.compile(
        optimizer="adam",
        loss="mse"   # reconstruction error
    )
    ae.fit(
        X_normal,
        X_normal,
        batch_size=config.AE_BATCH,
        epochs=config.AE_EPOCHS,
        validation_split=0.1,
        verbose=1
    )
    return ae


# ==========================================================
# 4. ISOLATION FOREST
# ==========================================================

# TYPE: Unsupervised Learning
# LEARNS FROM: ONLY normal traffic
# PURPOSE: Detect anomalies by isolation
# IDEA: anomalies are easier to isolate than normal samples
# WHY IN PAPER 2: compare against Autoencoder

def build_isolation_forest(X_train, y_train, dataset_name):
    benign = config.BENIGN_LABEL[dataset_name]
    X_normal = X_train[y_train == benign]  # train on normal traffic only

    # Safety check: catch empty benign set (e.g. wrong BENIGN_LABEL).
    if len(X_normal) == 0:
        raise ValueError(
            f"No benign samples for '{dataset_name}': BENIGN_LABEL is {benign} "
            f"but training labels are {np.unique(y_train)}. Check BENIGN_LABEL matches the encoded label."
        )

    iso = IsolationForest(
        n_estimators=100,                  # number of trees
        random_state=config.SEED,          # reproducible results
        n_jobs=-1                          # use all CPU cores
    )
    iso.fit(X_normal)                     # learn normal behavior
    return iso