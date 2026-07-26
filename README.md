# XAI Stability Benchmark

Extends a benchmark for explainable intrusion detection on autonomous
systems with additional models, XAI methods, and a new metric — the
Explainability Stability Score (ESS) — that measures how much different
explanation methods agree.

Builds on TCSS499_Research_Benchmark.

## Structure
- config.py — settings and dataset paths
- data_loader.py — loads, cleans, and splits datasets
- models.py — RF, CNN, Autoencoder, Isolation Forest
- metrics.py — ML metrics (SE metrics and ESS in progress)
- xai.py — SHAP, LIME, Permutation Importance, Integrated Gradients
