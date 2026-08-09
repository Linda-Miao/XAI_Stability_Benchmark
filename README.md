# XAI Stability Benchmark

Extends a benchmark for explainable intrusion detection on autonomous systems with additional models, XAI methods, and a new metric — the **Explainability Stability Score (ESS)** — that measures how much different explanation methods agree on which features matter.

Builds on [TCSS499_Research_Benchmark](https://github.com/Linda-Miao/TCSS499_Research_Benchmark).

**Interactive results explorer:** https://xai-app-eta.vercel.app

## Scope
- **6 datasets** (UAVCAN, UAVIDS-2025, UAV-Cyber, ISOT, UAV-Attack, CICIDS2017)
- **4 models** — Random Forest, 1D-CNN, Autoencoder, Isolation Forest
- **4 XAI methods** — SHAP, LIME, Permutation Importance, Integrated Gradients
- **3 evaluation layers** — ML metrics, software-engineering metrics, and ESS

## Structure
- `config.py` — settings and dataset paths
- `data_loader.py` — loads, cleans, and splits datasets
- `models.py` — RF, CNN, Autoencoder, Isolation Forest
- `metrics.py` — ML metrics and ESS (SE metrics in progress)
- `xai.py` — SHAP, LIME, Permutation Importance, Integrated Gradients
- `run.py` — runs the full pipeline across all datasets

## Status
Implementation complete; results finalized. Paper in progress.