"""
xai.py
PURPOSE:
Explain WHY each model made its decision.
If metrics.py is the "report card" for the models,
xai.py is the "interview" - it asks each model to justify its answers.

Paper 2 uses four XAI methods. Each one asks the same question
in a different way: "which features mattered most?"
---------------------------------------------------------
1. SHAP (SHapley Additive exPlanations)
---------------------------------------------------------
Question: "How much did each feature contribute, split fairly?"
Idea: from game theory - divide the credit among all features
      the way you would divide a team's prize money.
Note: SHAP needs a different explainer per model type:
      RF  -> TreeExplainer     (fast)
      CNN -> GradientExplainer (neural networks)
      AE  -> KernelExplainer   (works on anything, but slowest)
---------------------------------------------------------
2. LIME (Local Interpretable Model-agnostic Explanations)
---------------------------------------------------------
Question: "For THIS one prediction, what mattered?"
Idea: build a simple model around one sample and read its weights.
Note: LIME is LOCAL (explains one instance);
      the other three are GLOBAL (explain overall behavior).
---------------------------------------------------------
3. PERMUTATION IMPORTANCE (PI)
---------------------------------------------------------
Question: "If I shuffle this feature, how much does performance drop?"
Idea: break one feature at a time and see what the model loses.
      Big drop = important feature. No drop = not used.
---------------------------------------------------------
4. INTEGRATED GRADIENTS (IG)  <- NEW IN PAPER 2
---------------------------------------------------------
Question: "How sensitive is the output to each input?"
Idea: turn each feature slowly from zero up to its real value
      and watch how much the model's answer moves.
Note: gradient-based, so it suits the neural models (CNN, AE).
      Needs no new library - pure TensorFlow.

---------------------------------------------------------
OUTPUT FORMAT (important for ESS)
---------------------------------------------------------
Every function returns a numpy array of importance scores,
one per feature, in the same order as feature_names.
Scores are ABSOLUTE magnitude (direction ignored) so that all
four methods can be compared on the same basis.
ESS then takes the top-5 features from each method and
measures how much the four lists overlap.
"""

import numpy as np
import shap
import config

# ==========================================================
# 1. SHAP
# ==========================================================
# SHAP needs a DIFFERENT explainer depending on the model:
#   RF  -> TreeExplainer      (fast, built for tree-based models)
#   CNN -> GradientExplainer  (for neural networks = deep learning models)
#   AE -> KernelExplainer    (model-agnostic(work with almost any models), but slow); it reconstruction error instead of class predict; slowest mothod beause it repeatedly queries the model.
# IF -> KernelExplanier  (predicts anomaly scores instead of classes.)

"""
SHAP Explainers

1. TreeExplainer (Random Forest)
Use:
• Random Forest
• Decision Tree
• XGBoost
• LightGBM
• CatBoost

Advantages:
• Fastest
• Exact SHAP values
• Designed for tree models

Disadvantages:
• Only works for tree-based models(Random Forest; Decision Tree; XGBoost; LightGBM and CatBoost)
----------------------------------------------------
2. GradientExplainer (CNN)
Use:
• Neural Networks (CNN, DNN, etc.)
Why?
Neural networks are made of neurons.
Gradient tells us which direction changes the prediction the most.

Advantages:
• Designed for neural networks
• Faster than KernelExplainer
• Uses network gradients

Disadvantages:
• Only works for differentiable neural networks
----------------------------------------------------
3. KernelExplainer (AE / IF)
Use:
• Autoencoder
• Isolation Forest
• Almost any machine learning model
Why?
It treats the model as a black box.
It repeatedly changes one feature at a time,
runs the model again,
and measures how the prediction changes.

Advantages:
• Works on almost any model
• Very flexible
• No need to know the model's internal structure

Disadvantages:
• Slowest
• Runs the model thousands of times
• Uses lots of CPU
• Produces approximate (not exact) SHAP values
"""

def shap_random_forest(rf,X_sample, feature_names):
    # SHAP for Random Forest using TreeEplainer (fastest).
    explainer = shap.TreeExplainer(rf)            # initialize TreeExplainer
    shap_values = explainer.shap_values(X_sample) # generate SHAP values that quantify each feature's contribution to each prediction

    # shap is (samples, features, classes) for multi-class, take absolute value, then average over classes, then over samples
    if shap_values.ndim == 3: 
        importance = np.mean(np.abs(shap_values), axis=(0,2)) 
        # ndim = number dimension is 3 (samples, features and classes)
        # axis = (0,2) -> 0 = all samples; 2 = all classes.
        
    else: importance = np.mean(np.abs(shap_values),axis=0) # binary: avarage asolute SHAP values
    all_zero = bool(np.all(importance == 0))
    return importance
    
def shap_cnn(cnn,X_sample, X_background, feature_names):
    # SHAP for CNN usig GradientExplainer (need 3D input).
    n_features = X_sample.shape[1]

    # CNN needs 3d: samples, features, channels
    # -1 = automatically keep the number of samples
    X_sample_3d = X_sample.reshape(-1, n_features, 1) 
    X_bg_3d = X_background.reshape(-1, n_features, 1)
    explainer = shap.GradientExplainer(cnn, X_bg_3d)
    shap_values = np.array(explainer.shap_values(X_sample_3d))

    # shap is (samples, features, 1, classes) - drop the channel axis
    # take absolute value, average over samples and classes
    importance = np.mean(np.abs(shap_values[:, :, 0, :]), axis=(0,2))
    return importance

def shap_autoencoder(ae, X_sample, X_background, feature_names):
    # X_background = normal reference used for comparison
    """
    SHAP for Autoencoder using KernalExplainer. the AE has no class output, so we wrap it: the     'prediction' it explain it the reconstruction error(MSE) per sample.
    WARNING: it is slowest model (paper 1: 380s on 50 samples)
    """
    def ae_error(X):
        X_pred = ae.predict(X, verbose=0)
        return np.mean(np.power(X - X_pred, 2), axis=1)
    explainer = shap.KernelExplainer(ae_error, X_background)
    shap_values = explainer.shap_values(X_sample)

    importance = np.mean(np.abs(shap_values), axis=0)
    return importance

def shap_isolation_forest(iso, X_sample, X_background, feature_names):
    # SHAP for Isolation Forest using KernelExplainer,
    # we explain the anomaly score (lower = more  anomalous)
    def iso_score(X):
        return iso.decision_function(X)
        
    explainer = shap.KernelExplainer(iso_score, X_background)
    shap_values = explainer.shap_values(X_sample)
    importance = np.mean(np.abs(shap_values), axis=0)
    return importance
        
"""
# ==============================================================================
# SHAP SUMMARY: TreeExplainer vs. GradientExplainer
# ==============================================================================

# KEY DIFFERENCES:
# 1. TreeExplainer:
#    - Models: Trees, Random Forests, XGBoost
#    - Structure: Discrete logic (if/else splits)
#    - Baseline: Internal node statistics (sample counts recorded during training)
#    - Core Math: Combinatorics & Weighted Averages

# 2. GradientExplainer:
#    - Models: Neural Networks
#    - Structure: Continuous math curves
#    - Baseline: External background dataset required
#    - Core Math: Calculus & Algebra (Integrated Gradients)

# FORMULAS (Plain ASCII):

# TreeExplainer:
# phi_i = SUM_S [ (|S|! * (|F| - |S| - 1)!) / |F|! ] * [ E[f(x)|S U {i}] - E[f(x)|S] ]

# GradientExplainer:
# phi_i = (x_i - x'_i) * INTEGRAL_0_to_1 [ (df / dx_i)(x' + a*(x - x')) ] da

"""

# ==========================================================
# 2. PERMUTATION IMPORTANCE
# ==========================================================
# idea: shuffle one feature, measure how much performance drops; 
# big drop = important feature, no drop = model doesn't needt it.
# WARNING: if a model scores 100%, shuffling one features may not hurt it at all. 

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score

def pi_random_forest(rf, X_sample, y_sample, feature_names, n_repeats = 10):
    # pi for RF - sklearn hadles this directly.
    result = permutation_importance(
        rf, X_sample, y_sample,
        n_repeats = n_repeats,
        random_state = config.SEED,
        scoring = "f1_weighted",
        n_jobs = -1
    )
    importance = result.importances_mean
    all_zero = bool(np.all(importance == 0))
    return importance, all_zero

def pi_cnn(cnn, X_sample, y_sample, feature_names, n_repeats = 10):
    """
    PI for CNN - implemented manually 
    because sklearn's version cannot hadle keras models (3D input, probability output).
    WARNING: slwest method in the pipeline (paper 1: 508s)
    """
    n_features = X_sample.shape[1]

    # baseline score with all features intact
    X_3d = X_sample.reshape(-1, n_features, 1)
    baseline_pred = np.argmax(cnn.predict(X_3d, verbose = 0), axis = 1)
    baseline = f1_score(y_sample, baseline_pred, average = "weighted", zero_division = 0)
    importance = np.zeros(n_features)

    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_sample.copy()
            np.random.shuffle(X_perm[:, j]) # break one feature
            X_perm_3d = X_perm.reshape(-1, n_features, 1)
            pred = np.argmax(cnn.predict(X_perm_3d, verbose=0), axis=1)
            score = f1_score(y_sample, pred, average="weighted", zero_division=0)
            drops.append(baseline - score)
        importance[j] = np.mean(drops)
    all_zero = bool(np.all(importance == 0))
    return importance, all_zero

def pi_autoencoder(ae, X_sample, y_sample, feature_names, dataset_name, n_repeats = 10):
    """
    PI for autoencoder - shuffle a feature, see how much f1 drops.
    the ae has no class output, so we convert reconstruction error to a binary prediction using th e95th-percentile threshold
    """
    benign = config.BENIGN_LABEL[dataset_name]
    n_features = X_sample.shape[1]
    y_true = (y_sample != benign).astype(int)

    # baseline: reconstruct, get MSE, threshold, score
    X_pred = ae.predict(X_sample, verbose=0)
    mse = np.mean(np.power(X_sample - X_pred, 2), axis=1)
    threshold = np.percentile(mse[y_sample == benign], 95)
    baseline = f1_score(y_true, (mse > threshold).astype(int), zero_division=0)

    importance = np.zeros(n_features)
    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_sample.copy()
            np.random.shuffle(X_perm[:, j])
            X_pred_perm = ae.predict(X_perm, verbose=0)
            mse_perm = np.mean(np.power(X_perm - X_pred_perm, 2), axis=1)
            score = f1_score(y_true, (mse_perm > threshold).astype(int), zero_division=0)
            drops.append(baseline - score)
        importance[j] = np.mean(drops)

    all_zero = bool(np.all(importance == 0))
    return importance, all_zero
    
def pi_isolation_forest(iso, X_sample, y_sample, feature_names, dataset_name, n_repeats = 10):
    # pi for isolation forest - same idea, using its +1/-1 output
    benign = config.BENIGN_LABEL[dataset_name]
    n_features = X_sample.shape[1]
    y_true = (y_sample != benign).astype(int)

    baseline = f1_score(y_true, (iso.predict(X_sample) == -1).astype(int), zero_division=0)
    importance = np.zeros(n_features)
    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_sample.copy()
            np.random.shuffle(X_perm[:, j])
            score = f1_score(y_true, (iso.predict(X_sample) == -1).astype(int), zero_division=0)
            drops.append(baseline - score)
        importance[j] = np.mean(drops)
    all_zero = bool(np.all(importance == 0))
    return importance, all_zero

# ==========================================================
# 3. LIME
# ==========================================================
# LIME is LOCAL, it explains one prediction at a time by perturbing that sample and fitting a simple linear model to the results.

# ESS needs a GLOBAL ranking comparable to SHAP and PI, so we run LIME on many samples and average the asolute feature weights.
# Paper 1 used LIME illustratively on 2 instances; averaging over n_samples gives a stable global picture.

# NOTE: LIME returns feature DESCRPTIONS like "-0.51 < lat_y <= -0.27", not plain names, so we map each description back to a feature index.

import lime
import lime.lime_tabular
def _lime_global(explainer, predict_fn, X_sample, feature_names, n_samples, n_perturb=1000):
    """
    shared helper: run LIME on n_samples instances and average absolute feature weigts 
    into one global importance array. 
    Returns the running average after every instance, so we can compare(say) 
    50 vs 100 sample without re-running LIME.
    """
    n_features = len(feature_names)
    running = np.zeros(n_features)
    snapshots = {}  # {n_used: importance array}

    for i in range (n_samples):
        exp = explainer.explain_instance(
            X_sample[i],
            predict_fn,
            num_features = n_features, # get weights for all features
            num_samples = n_perturb
        )

        # exp.at-map() gives {class:[(feature_index, weight),...]} -> returns features 
        #indices directly which sidesteps the string-parsing problem entirely.
        # take the first available class's explanation
        weights = list(exp.as_map().values())[0]
        for  feat_idx, w in weights:
            running[feat_idx] += abs(w)
        if (i+1) in (50, n_samples):
            snapshots[i+1] = running.copy()/(i+1)
    return snapshots
    # snapshots solves 50 vs 100 question in one pass and record 
    # running 50 and 100 results for me to comparison.

def lime_random_forest(rf, X_train, X_sample, feature_names, n_samples = 100):
    """LIME for Random Forest - works directly with predict_proba."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="classification",
        random_state=config.SEED
    )
    return _lime_global(explainer, rf.predict_proba, X_sample, feature_names, n_samples)
    # _lime_global is shared helper ,which use it for 4 times but only write it once.
    
def lime_cnn(cnn, X_train, X_sample, feature_names, n_samples = 100):
    # LIME for CNN - needs a wrapper to reshape 2D input to 3D.
    n_features = len(feature_names)

    def cnn_predict(X):
        X_3d = X.reshape(X.shape[0], n_features, 1)
        return cnn.predict(X_3d, verbose=0)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="classification",
        random_state=config.SEED
    )
    return _lime_global(explainer, cnn_predict, X_sample, feature_names, n_samples)

def lime_autoencoder(ae, X_train, X_sample, feature_names, threshold, n_samples = 100):
    """
    LIME for Autoencoder - needs a wrapper because the AE outputs
    reconstruction error, not class probabilities. 
    we convert MSE to [P(normal), P(attack)] using th threshold.
    """
    def ae_predict_proba(X):
        X_pred = ae.predict(X, verbose=0)
        mse = np.mean(np.power(X- X_pred, 2), axis=1)
        attack_prob = np.clip(mse / (threshold * 2), 0, 1)
        return np.column_stack([1 - attack_prob, attack_prob])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names = feature_names,
        mode = "classification",
        random_state=config.SEED
    )
    return _lime_global(explainer, ae_predict_proba, X_sample, feature_names, n_samples)

def lime_isolation_forest(iso, X_train, X_sample, feature_names, n_samples = 100):
    """
    LIME for Isolation Forest - convert the anomaly score to pseudo-probabilities. 
    decision_function is negatve for outliers
    """
    def iso_predict_proba(X):
        scores = iso.decision_function(X)
        attack_prob = np.clip(-scores + 0.5, 0, 1)
        return np.column_stack([1 - attack_prob, attack_prob])
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, 
        feature_names = feature_names,
        mode="classification",
        random_state = config.SEED   
    )
    return _lime_global(explainer, iso_predict_proba, X_sample, feature_names, n_samples)

# ==========================================================
# 4. INTEGRATED GRADIENTS  (NEW IN PAPER 2)
# ==========================================================
# Idea: start from a baseline (all zeros), step gradually towars the 
# real input, and measure the gradient at each step. Sum those gradients along the path.

# ONLY works on neural models (CNN, AE) - tree have no gradients.
# Needs no new library: pure TensorFlow

import tensorflow as tf

def ig_cnn(cnn, X_sample, feature_names, n_steps=50):
    """
    Integrated Gradients for the CNN.
    Attributes the predicted class score back to each to each input feature.
    """
    n_features = X_sample.shape[1]
    X = X_sample.reshape(-1, n_features, 1).astype("float32")
    baseline = np.zeros_like(X)

    total = np.zeros(n_features)

    for i in range(X.shape[0]):
        x = X[i:i+1]
        base = baseline[i:i+1]

        # build the path from baseline to input in n_steps
        alphas = np.linspace(0, 1, n_steps).reshape(-1, 1, 1).astype("float32")
        path = base + alphas * (x - base)  # (n_steps, features, 1)
        path_tf = tf.convert_to_tensor(path)

        with tf.GradientTape() as tape:
            tape.watch(path_tf)
            preds = cnn(path_tf)
            target = tf.reduce_max(preds, axis=1) # the winning clas score

        grads = tape.gradient(target, path_tf).numpy() # (n_steps, features, 1)

        # average the gradients along the path, scale by the input different
        avg_grads = np.mean(grads, axis=0) # (features, 1)
        attribution = (x[0] - base[0]) * avg_grads   # (features, 1)
        total += np.abs(attribution).flatten()

    importance = total / X.shape[0]
    all_zero = bool(np.all(importance == 0))
    return importance, all_zero

def ig_autoencoder(ae, X_sample, feature_names, n_steps=50):
    """
    integreate gradients for the autoencoder. 
    The AE has no class score, so we attribute reconstruction error
    (MSE) back to each input feature.
    """
    n_features = X_sample.shape[1]
    X = X_sample.astype("float32")
    baseline = np.zeros_like(X)

    total = np.zeros(n_features)
    for i in range(X.shape[0]):
        x = X[i:i+1]
        base = baseline[i:i+1]
        alphas = np.linspace(0, 1, n_steps).reshape(-1,1).astype("float32")
        path = base + alphas * (x - base)        # (n_steps, features)
        path_tf = tf.convert_to_tensor(path)

        with tf.GradientTape() as tape:
            tape.watch(path_tf)
            recon = ae(path_tf)
            mse = tf.reduce_mean(tf.square(path_tf - recon), axis = 1)

        grads = tape.gradient(mse, path_tf).numpy()

        avg_grads = np.mean(grads, axis = 0)
        attribution = (x[0] - base[0]) * avg_grads
        total += np.abs(attribution)

    importance = total / X.shape[0]
    all_zero = bool(np.all(importance == 0))
    return importance, all_zero
        