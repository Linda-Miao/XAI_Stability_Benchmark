"""
metrics.py
PURPOSE:
Measure how well each IDS model performs on unseen test data.
Think of his file as the "report card" for the models
Paper 2 has three evaluation layers:
---------------------------------------------------------
1. MACHINE LEARNING (ML) METRICS
---------------------------------------------------------
Question: "how good is the model at detecting attacks?"
Metrics: 
Accuracy = overall percentage of correc predictions
Precision = when model says "attack", how ofen is it correct?
Recall = of all real attacks, how many did he model find?
F1 score = balance between Precision and Recall

---------------------------------------------------------
2. SOFTWARE ENGINEERING (SE) METRICS
---------------------------------------------------------
Question: "How pratical is explanatin method?"
Examples: runtime, memory usage, reproducibility and integration effort; (added later)

---------------------------------------------------------
3. ESS
---------------------------------------------------------
ESS = Explainability stability score
research contribution panned for paper 2
question: if we run multiple XAI methods, do they probudce similar explantions? (added later)
"""
import numpy as np
# function that compute evealuation scores
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import config

# ==========================================================
# SUPERVISED MODELS
# RF and CNN
# ==========================================================

def evaluate_supervised(model,X_test, y_test, is_cnn=False):
    """
    Evaluate RF or CNN(it like grading a student afer studying).
    INPUT: model: trained RF or CNN; X_test unseen test data; y_test correct answers
    OUTPUT: accuracy, precision, recall and f1
    """

    # ------------------------------------------------------
    # CNN needs special input shape
    # ------------------------------------------------------
    if is_cnn:
        # CNN expects: (sample features, channels)
        X_test_cnn = X_test.reshape(
            -1, X_test.shape[1],1
        )
        # output example: sameple 1: [0.01, 0.92, 0.04,0.03]
        # means: class 0 = 1%, class 1 = 92%, class 2 = 4% and class 3 = 3%
        # argmax chooses he largest probability. result: class 1

        y_pred = np.argmax(
            model.predict(X_test_cnn, verbose=0), axis = 1
        )
    else: # RF already predicts classes directly: [0,1,1,0,2,1]
        y_pred = model.predict(X_test) 
    # ------------------------------------------------------
    # Calculate report-card scores
    # ------------------------------------------------------
    return{
        # when model show attack, how often is it righ?
        "accuracy": accuracy_score(y_test,y_pred),
        "precision": precision_score(
            y_test, y_pred, average = "weighted", zero_division=0
        ),

        # of all real atacks, how many did model catach?
        "recall":
            recall_score(y_test,y_pred, average="weighted", zero_division=0),
        # balance between precision and recall
        "f1":
            f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
    }

# ==========================================================
# AUTOENCODER
# ==========================================================
def evaluate_autoencoder(ae,X_test,y_test, dataset_name):
    """
    AE only learn normal traffic, it nver learns attacks. thus, it cannot  show "this is DDoS"
    instead it asks: "can I reconstruct this sample?" 
    if reconstruction error is large: attack; if reconstruction error is small: normal
    """
    benign = config.BENIGN_LABEL[dataset_name]
    # ------------------------------------------------------
    # STEP 1
    # Reconstruct every test sample
    # ------------------------------------------------------
    X_pred = ae.predict(X_test, verbose=0)
    # ------------------------------------------------------
    # STEP 2
    # Calculate reconstruction error
    # ------------------------------------------------------
    # MSE = mean squared error
    # measures: how different is reconstruction form original samples: small MSE: normal; large MSE: attack
    mse = np.mean(np.power(
        X_test - X_pred, 2),
                  axis=1)
    # ------------------------------------------------------
    # STEP 3
    # Find threshold
    # ------------------------------------------------------
    # the goal is set a standard number to check the results:
    # example: 0.01,0.02,0.03...0.10, theshold: 0.08. above threshold: attack, below threshold: normal
    normal_mse = mse[y_test == benign]
    threshold = np.percentile(normal_mse, 95)
    # ------------------------------------------------------
    # STEP 4
    # Convert to attack / normal
    # ------------------------------------------------------
    y_pred = (mse > threshold).astype(int)    # 1 = attack
    y_true = (y_test != benign).astype(int)  # 1 = attack

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": float(threshold),
    }
    
    # ==========================================================
    # ISOLATION FOREST
    # ==========================================================
def evaluate_isolation_forest(iso, X_test, y_test, dataset_name):
    """
    Isolation Forest labels each sample inlier or outlier.
    +1 = inlier(normal), -1 = outlier(atack).
    """
    benign = config.BENIGN_LABEL[dataset_name]
        
    raw = iso.predict(X_test)
    y_pred = (raw == -1).astype(int)        # 1 = attack
    y_true = (y_test != benign).astype(int) # 1 = attack

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
        