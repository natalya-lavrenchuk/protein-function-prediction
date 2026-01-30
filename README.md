Protein Function Prediction Project

Milestone 2: Logistic Regression with SAGA and F-max results

1. Methodology
Algorithm: Multi-label classification using Logistic Regression wrapped in a OneVsRestClassifier strategy.
Feature Space: Proteins were represented using ProtT5 transformer embeddings (1024-dimensional vectors).
Computational Optimization: For Cross-Validation - liblinear solver (to rapidly iterate and calculate the Fmax). For the Final Model - SAGA solver (superior handling of large-scale protein data).
Validation Strategy: 3-fold cross-validation was performed on a representative 1,000-protein subset.

2. Evaluation & Baseline Comparison
Fmax scores (calculated via cross-validation on a 1,000-protein subset) against the provided Naive and InterPro baselines.

Molecular Function (MFO)
Our SAGA Model: Fmax=0.3283 | Baseline Naive Fmax=0.4503 | InterPro Fmax=0.6181

Cellular Component (CCO)
Our SAGA Model: Fmax=0.4619 | Baseline Naive Fmax=0.5976 | InterPro Fmax=0.2563

Biological Process (BPO)
Our SAGA Model: Fmax=0.2768 | Baseline Naive Fmax=0.3439 | InterPro Fmax=0.3505

Performance Summary: While our current scores are based on only 1% of the training data (1,000 proteins), the strong predictive power is achived. 
