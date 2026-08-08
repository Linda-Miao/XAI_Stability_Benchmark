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

evaluate_supervised, evaluate_autoencoder, evaluate_isolation_forest.
---------------------------------------------------------
2. SOFTWARE ENGINEERING (SE) METRICS
---------------------------------------------------------
Question: "How pratical is explanatin method?"
Examples: runtime, memory usage, reproducibility and integration effort; (added later)

Runtime — measured automatically with a timer (easy)
Integration effort — rule-based (count lines of wrapper code)
Wrapper required — a recorded fact (yes/no per method×model)
Reproducibility — measured by re-running and checking agreement (reuses ESS machinery)

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
    evaluate_supervised -> ML 
    """

    # ------------------------------------------------------
    # CNN needs special input shape
    # ------------------------------------------------------
    if is_cnn:
        # CNN expects: (sample, features, channels)
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
    evaluate_autoencoder -> ML 
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
        # ===== COMPNAY / SECURITY VIEW (binary, attack-focused) =====
        # These measure ONLY how well we catch attacks (the attack class alone),
        # not averaged with benign. This is what a real security team cares about:
        # "of all the atacks, how many did we catch?" a low recall here means
        # attacks slip through - a critical, honest signal even if it looks bad.
        # No averaging. The score reflects attack detection only and is not made higher
        # by the many correctly classified normal samples.
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),

        # ===== PAPER 1 COMPARISON VIEW (weishged across both classes) =====
        # These average the score across BOTH benign and attach classes, weighted
        # by class size. This matches paper 1;s method, so our numbers are riectly 
        # comparable to it. Because benign is the majority and easy to classify,
        # weighting pulls the score UP, which is why the weighted F1 looks higher
        # than the binary F1. Same predictions, different way of summaryizing them.
         "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        
        "threshold": float(threshold),
    }
    
# ==========================================================
# ISOLATION FOREST
# ==========================================================
def evaluate_isolation_forest(iso, X_test, y_test, dataset_name):
    """
    Isolation Forest labels each sample inlier or outlier.
    +1 = inlier(normal), -1 = outlier(attack).
    evaluate_isolation_forest -> ML
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

# ==========================================================
# ESS - EXPLAINABILITY STABILITY SCORE  (the contribution)
# ==========================================================
# ESS measures how much the different XAI methods AGREE on which features matter, 
# for a single model on a single dataset.

# Steps: 
# 1. Take each methods importance array (skip any with all_zero = True)
# 2. From each, take the top-k features (K = config.TOP_K, default 5).
# 3. For every PAIR of methods, compute Jaccard similarity of their 
#    top-K sets: J(A,B) = |A ∩ B| / |A ∪ B|
# ESS = average of all parwise, Jaccard scores.

# High ESS = methods agree, one explanation can be trusted.
# Low ESS = methods disagree, a single explanation is unreliable.

# Needs at least 2 usable methods, Fewer than 2 -> ESS is underfined
# (it can not measure agreement with only one method), returns None.

from itertools import combinations

def top_k_set(importance, k):
    # Return the indces of the top-k features by absolute magnitude.
    return set(np.argsort(np.abs(importance))[::-1][:k])
def jaccard(set_a, set_b):
    # Jaccard similarity between two sets: |intersection| / |union|.
    union = set_a | set_b
    if not union:        # both empty (shouldn't happen)
        return 0.0
    return len(set_a & set_b) / len(union)

def compute_ess(method_results, k=None):
    """
    Compute ESS for one model from its XAI method outputs 
    method_results: dict mapping method name -> (importance, all_zero)
        e.g. {"shap": (imp, False), "lime": (imp, False),
               "pi": (imp, True), "ig": (imp,False)}
    Return a dict:
        {
          "ess": float or None,
          "methods_used": [names of methods that contributed],
          "pairwise": {("shap", "lime"): 0.43, ...},
          "n_methods": int,
        }
    """
    if k is None:
        k = config.TOP_K

    # 1. keep only methods that produced a usable ranking
    usable = {
        name: imp
        for name, (imp, all_zero) in method_results.items()
        if not all_zero
    }

    methods_used = sorted(usable.keys())
    n = len(methods_used)

    # 2. need at least 2 methods to measure agreement
    if n < 2:
        return {
            "ess": None,
            "methods_used": methods_used,
            "pairwise": {},
            "n_methods": n,
        }
    # 3. top-k set for each usable method
    top_sets = {name: top_k_set(imp, k) for name, imp in usable.items()}

    # 4. pairwise Jaccard over every pair of methods
    pairwise = {}
    for a, b in combinations(methods_used, 2):
        pairwise[(a,b)] = jaccard(top_sets[a], top_sets[b])
    ess = float(np.mean(list(pairwise.values())))

    return {
        "ess": ess,
        "methods_used": methods_used,
        "pairwise": pairwise,
        "n_methods": n
    }