"""
run.py — the conductor.
Runs the full pipeline for one or more datasets:
  load data -> train 4 models -> run 4 XAI methods -> compute ML metrics + ESS
  -> collect everything into a results structure.

This is the only file you execute to produce results. It calls the other
modules; it contains no analysis logic of its own.

run.py
│
├── Load dataset
│
├── Train 4 models
│     ├── RF
│     ├── CNN
│     ├── AE
│     └── Isolation Forest
│
├── Evaluate ML performance
│     ├── Accuracy
│     ├── Precision
│     ├── Recall
│     └── F1
│
├── Run XAI
│     ├── SHAP
│     ├── LIME
│     ├── PI
│     └── IG
│
├── Compute ESS
│
└── Save all results
"""
import time
import numpy as np

import config
import data_loader
import models
import metrics
import xai


# ----------------------------------------------------------
# Run all 4 XAI methods for one model, return {method: (importance, all_zero)}
# ----------------------------------------------------------
def run_xai_for_model(model_name, model, Xtr, Xte, ytr, yte, feat, dataset_name, threshold=None):
    """
    Returns a dict {method_name: (importance, all_zero)} for the given model,
    plus a dict of runtimes {method_name: seconds}.
    Only runs the methods that apply to this model type.
    """
    X_bg = Xtr[:config.SHAP_BACKGROUND] 
    # X_bg = Xtr[:config.SHAP_BACKGROUND]  -> background samples,
    # Only SHAP KernelExplainer and GradientExplainer need these.
    """
    SHAPE Eexplaier   | model | background? |  why?
    TreeExplaier      | RF    | NO          | It can read the tree structure directly.
    GradientExplainer | CNN   | YES          | It can read the tree structure directly.
    KernelExplainer   | AE, Isolation Forest| YES | It can read the tree structure directly.
    """
    results = {}  # each xai importance arrays
    runtimes = {} # get each xai run time

    def timed(name, func):
        start = time.time()
        out = func()
        runtimes[name] = round(time.time() - start, 2)
        results[name] = out

    if model_name == "rf":
        timed("shap", lambda: xai.shap_random_forest(model, Xte[:config.SHAP_SAMPLES_RF], feat))
        timed("lime", lambda: xai.lime_random_forest(model, Xtr, Xte, feat))
        timed("pi",   lambda: xai.pi_random_forest(model, Xte, yte, feat))
        # TreeSHAP does not require background samples because it uses the tree structure directly.
        # IG is not used because tree models do not provide gradients.

    elif model_name == "cnn":
        timed("shap", lambda: xai.shap_cnn(model, Xte[:config.SHAP_SAMPLES_CNN], X_bg, feat))
        timed("lime", lambda: xai.lime_cnn(model, Xtr, Xte, feat))
        timed("pi",   lambda: xai.pi_cnn(model, Xte, yte, feat))
        timed("ig",   lambda: xai.ig_cnn(model, Xte[:100], feat))
        # Deep SHAP/GradientExplainer requires background samples because they define the reference distribution for explanations.
        # LIME and PI do not need background samples because they use perturbations and evaluation data respectively.
        # IG uses gradients and a baseline input rather than SHAP background samples.
        
    elif model_name == "ae":
        timed("shap", lambda: xai.shap_autoencoder(model, Xte[:config.SHAP_SAMPLES_AE], X_bg, feat))
        timed("lime", lambda: xai.lime_autoencoder(model, Xtr, Xte, feat, threshold))
        timed("pi",   lambda: xai.pi_autoencoder(model, Xte, yte, feat, dataset_name))
        timed("ig",   lambda: xai.ig_autoencoder(model, Xte[:100], feat))
         # KernelSHAP requires background samples because it estimates feature contributions relative to a reference dataset.
        # LIME and PI do not require background samples because they rely on local perturbations and feature shuffling.
        # IG uses gradients and a baseline input, not SHAP background data.

    elif model_name == "iso":
        timed("shap", lambda: xai.shap_isolation_forest(model, Xte[:config.SHAP_SAMPLES_AE], X_bg, feat))
        timed("lime", lambda: xai.lime_isolation_forest(model, Xtr, Xte, feat))
        timed("pi",   lambda: xai.pi_isolation_forest(model, Xte, yte, feat, dataset_name))
        # Only SHAP may require background samples depending on the SHAP explainer used.
        # LIME uses training data to create local perturbations; PI uses test data to measure feature impact.
        # IG is not used because Isolation Forest is a tree-based model without gradients.

    return results, runtimes
    


# ----------------------------------------------------------
# Run the full pipeline for ONE dataset
# ----------------------------------------------------------
def run_dataset(dataset_name, verbose=True): 
    # data_name = which dataset to run; verbose = whether to print progress information
    """
    Train all 4 models on one dataset, run XAI, compute ML metrics + ESS.
    Returns a nested dict of all results for this dataset.
    """
    if verbose:
        print(f"\n{'='*55}\nDATASET: {dataset_name}\n{'='*55}")

    Xtr, Xte, ytr, yte, feat = data_loader.load_dataset(dataset_name)
    # Xtr = training features; Xte = testing features; ytr= training labels; yte = testing labels

    # 1. train the 4 models
    if verbose: print("training models...")
    rf  = models.build_random_forest(Xtr, ytr)
    cnn = models.build_cnn(Xtr, ytr)
    ae  = models.build_autoencoder(Xtr, ytr, dataset_name)
    iso = models.build_isolation_forest(Xtr, ytr, dataset_name)

    # 2. ML metrics (calulate: AE threshold; RF/CNN/AE/IF f1)
    if verbose: print("computing ML metrics...")
    ae_ml = metrics.evaluate_autoencoder(ae, Xte, yte, dataset_name)
    threshold = ae_ml["threshold"]     # AE, LIME needs this
    ml_metrics = {
        "rf":  metrics.evaluate_supervised(rf, Xte, yte),
        "cnn": metrics.evaluate_supervised(cnn, Xte, yte, is_cnn=True),
        "ae":  ae_ml,
        "iso": metrics.evaluate_isolation_forest(iso, Xte, yte, dataset_name),
    }

    # 3. XAI + ESS per model
    model_objs = {"rf": rf, "cnn": cnn, "ae": ae, "iso": iso}
    ess_results = {}
    xai_runtimes = {}

    for mname, mobj in model_objs.items():
        if verbose: print(f"running XAI for {mname}...")
        xai_out, runtimes = run_xai_for_model(
            mname, mobj, Xtr, Xte, ytr, yte, feat, dataset_name, threshold
        )
        ess_results[mname] = metrics.compute_ess(xai_out)
        xai_runtimes[mname] = runtimes

    # 4. assemble
    dataset_results = {
        "dataset": dataset_name,
        "n_features": len(feat),
        "ml_metrics": ml_metrics,
        "ess": ess_results,
        "runtimes": xai_runtimes,
    }

    if verbose:
        print(f"\n--- {dataset_name} summary ---")
        for mname in model_objs:
            ess = ess_results[mname]["ess"]
            f1 = ml_metrics[mname]["f1"]
            n = ess_results[mname]["n_methods"]
            ess_str = f"{ess:.3f}" if ess is not None else "N/A"
            print(f"  {mname:4}  F1={f1:.3f}  ESS={ess_str} ({n} methods)")

    return dataset_results


# ----------------------------------------------------------
# Run several datasets
# ----------------------------------------------------------
def run_all(dataset_names=None):
    """Run the pipeline for a list of datasets (default: all in config)."""
    if dataset_names is None:
        dataset_names = list(config.DATASETS.keys())

    all_results = {}
    for name in dataset_names:
        all_results[name] = run_dataset(name)
    return all_results