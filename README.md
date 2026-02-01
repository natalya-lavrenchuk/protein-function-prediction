Protein Function Prediction (GO Term Assignment)

This repository implements a multi-label protein function prediction pipeline for Gene Ontology (GO) terms, using a combination of:

Protein sequence embeddings

InterPro (IPR) domain features

Optional BLAST-based annotation transfer

Linear classifiers with ensemble blending

CAFA-style evaluation (Fmax, PR curves)

The project supports training, validation, testing, evaluation, and submission generation, with all steps orchestrated through a master notebook.

Project Structure
protein-func-pred/
│
├── artifacts/              # Saved GO term lists per aspect
├── configs/
│   └── config.yaml         # Central configuration file
│
├── data/
│   ├── raw/                # Original provided datasets
│   └── splits/             # Train/val splits, BLAST results
│
├── figures/                # Generated plots for the report
├── metrics/                # JSON metrics (Fmax, thresholds, etc.)
│
├── models/                 # Trained models (one per GO aspect)
│   ├── model_biological_process.pkl
│   ├── model_cellular_component.pkl
│   └── model_molecular_function.pkl
│
├── outputs/
│   ├── preds_raw/          # Raw prediction matrices
│   ├── submission/         # Earlier submissions
│   ├── submission_v4/      # Final embedding + IPR ensemble
│   └── submission_v4_blast # Final ensemble + BLAST
│
├── results/                # CAFA-style intermediate outputs
│
├── scripts/
│   ├── model_train_v4.py           # Final training pipeline
│   ├── predict_test_v4.py          # Test-set prediction
│   ├── eval_model_plus_blast_val.py# Validation with BLAST
│   ├── eval_blast_transfer.py      # BLAST-only baseline
│   ├── plots.py                    # Plot utilities
│   ├── master_file.ipynb           # Main execution notebook
│   └── (older versions kept for reference)
│
├── requirements.txt
├── README.md
└── report/                  # Final report and figures

Environment Setup
1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

2. Install dependencies
pip install -r requirements.txt


Note: BLAST evaluation requires BLAST results to already be generated and placed in data/splits/. BLAST is not run automatically inside this repo.

Configuration

All paths and hyperparameters are defined in:

configs/config.yaml


Key options include:

Training/test file paths

GO aspects (molecular_function, biological_process, cellular_component)

Subsampling for hyperparameter tuning

Minimum positives per GO term

Random seed and CV folds

🧠 Model Overview
Features

Embeddings: Precomputed protein embeddings (.h5)

InterPro: Hashed domain features using HashingVectorizer

BLAST (optional): Annotation transfer baseline

Models

One-vs-Rest SGDClassifier (logistic loss)

Separate model per GO aspect

Probabilistic outputs

Ensemble

Final prediction scores are computed as:

P_final = λ · P_embeddings + (1 − λ) · P_interpro (+ BLAST if enabled)


λ is selected via validation to maximize Fmax.

🚀 Running the Pipeline
Option 1 (Recommended): Run everything via the master notebook

Open:

scripts/master_file.ipynb


This notebook runs, in order:

Training (model_train_v4.py)

Validation & threshold selection

BLAST validation (optional)

Figure generation (PR curves, F1 vs threshold)

Test-set prediction

Submission file creation

Option 2: Run scripts manually
Train the model
python scripts/model_train_v4.py


Outputs:

Models → models/

Metrics → metrics/

Artifacts → artifacts/

Evaluate BLAST transfer (optional)
python scripts/eval_blast_transfer.py

Validate model + BLAST ensemble
python scripts/eval_model_plus_blast_val.py

Predict on the test set
python scripts/predict_test_v4.py


Outputs:

outputs/submission_v4/
└── submission_final.tsv

Evaluation Metrics

The project uses CAFA-style evaluation, including:

Fmax (maximum micro-averaged F1 across thresholds)

Precision–Recall curves (micro-weighted)

F1 vs threshold plots

GO term prediction counts per protein

Metrics are saved as JSON files in:

metrics/


Figures are saved to:

figures/

Figures Generated

Key plots used in the report:

F1 vs threshold (per GO aspect)

Precision–Recall curves (micro-weighted)

Model comparison (embeddings vs IPR vs ensemble)

GO term filtering statistics

Distribution of predicted GO terms per protein

BLAST Integration (Optional)

BLAST is treated as an annotation transfer baseline and can be:

Evaluated alone

Blended with the embedding + IPR ensemble

BLAST results must be provided as TSV files with:

query   target   bits   evalue   score


BLAST is not required to run the core pipeline.

Final Output

The final submission file follows the CAFA format:

ProteinID   GO:XXXXXXX   score


Located at:

outputs/submission_v4/submission_final.tsv

Notes for Reviewers

The full training set is large; hyperparameter tuning is performed on a subsample for efficiency.

Thresholds are selected based on validation Fmax, not arbitrarily.

Older scripts are retained for transparency and comparison.

Authors

Natalya Lavrenchuk (ID 2141882)
Christina Caporale (ID 2141881)
Iuliia Osipova (ID 2148937)
Master’s Students — Data Science and Computational Chemistry
Protein Function Prediction Project