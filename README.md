# Protein Function Prediction (GO Term Assignment)

This repository contains a protein function prediction pipeline for assigning
**Gene Ontology (GO) terms** to proteins.  
The project combines multiple sources of biological information and evaluates
performance using **CAFA-style metrics**.

The final system uses an **ensemble model** based on:
- protein sequence embeddings
- InterPro (IPR) domain annotations
- BLAST-based annotation transfer

## Project Structure

```text
protein-func-pred/
├── artifacts/                  # Saved GO term lists per aspect
├── configs/
│   └── config.yaml             # Central configuration file
├── data/
│   ├── raw/                    # Original provided datasets
│   └── splits/                 # Train/val splits, BLAST results
├── figures/                    # Generated plots for the report
├── metrics/                    # JSON metrics (Fmax, thresholds, etc.)
├── models/                     # Trained models (one per GO aspect)
├── outputs/
│   ├── preds_raw/              # Raw prediction matrices
│   ├── submission_v4/          # Final embedding + IPR ensemble
│   └── submission_v4_blast/    # Final ensemble + BLAST
├── results/                    # CAFA-style intermediate outputs
├── scripts/
│   ├── model_train_v4.py       # Final training pipeline
│   ├── predict_test_v4.py      # Test-set prediction
│   ├── eval_model_plus_blast_val.py  # Validation with BLAST
│   ├── eval_blast_transfer.py  # BLAST-only baseline
│   ├── plots.py                # Plot utilities
│   └── master_file.ipynb       # Main execution notebook
├── requirements.txt
├── README.md
└── report/                     # Final report and figures
```

---

## Environment Setup

1. Create and activate a virtual environment:
```text
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```text
pip install -r requirements.txt
```

## Configuration

All file paths and hyperparameters are defined in:
```text
configs/config.yaml
```

This includes:
- data paths (train/test/BLAST)
- random seed
- cross-validation folds
- GO aspects
- subsampling size
- minimum number of positives per GO term

The pipeline can be reconfigured **without modifying code**.

---
## Model Overview
### Features
1. Embeddings: Precomputed protein embeddings (.h5)
2. InterPro: Hashed domain features using HashingVectorizer
3. BLAST: Annotation transfer baseline

### Models
- One-vs-Rest SGDClassifier (logistic loss)
- Separate model per GO aspect
- Probabilistic outputs

**Note on BLAST usage:**  
BLAST was treated as a post-training component. It was evaluated only on a held-out validation set to select optimal ensemble weights, which were then fixed and applied at test time alongside the trained embedding + InterPro model.

BLAST is **not executed automatically within this repository** due to storage and runtime constraints. Users who wish to run BLAST-based evaluation or testing must generate BLAST results externally and place them in `data/splits/` before running the relevant scripts.

### Ensemble

Training and final prediction scores are computed as:
```text
P_train = α · P_emb + (1-α)·P_ipr,
P_final = λ · P_train + (1 − λ) · P_blast

where,
α is the training ensemble weight between embeddings and InterPro domains
λ is determined in validation to maximize Fmax while incorporating BLAST
```

## Running the Pipeline

To train the models, validate, and test most probabilistic GO terms, run the master file:
```text
scripts/master_file.ipynb
```
This notebook runs, in order:
1. Training (model_train_v4.py) on embeddings and InterPro
2. Validation & threshold selection
3. BLAST validation (optional)
4. Figure generation (PR curves, F1 vs threshold)
5. Test-set prediction
6. Submission file creation

**Note:** Full training on the complete dataset can take approximately **1–2 hours**
depending on hardware. Hyperparameter tuning is performed on a subsampled training
set to reduce runtime.


## Outputs:

### Model Training and Validation
Models
```text
models/
└── model_biological_process.pkl
└── model_cellular_component.pkl
└── model_molecular_function.pkl
```

Metrics
```text
metrics/
└── metrics_biological_process.json
└── metrics_blast_val.json
└── metrics_cellular_components.json
└── metrics_model_plus_blast_val.json
└── metrics_molecular_funtion.json
```

Evaluate BLAST transfer:
```text
python scripts/eval_blast_transfer.py
```

Validate model + BLAST ensemble
```text
python scripts/eval_model_plus_blast_val.py
```

### Predict on Test Set
```text
python scripts/predict_test_v4.py

outputs/submission_v4/
└── submission_final.tsv
```

## Evaluation Metrics
- The project uses CAFA-style evaluation, including:
- Fmax (maximum micro-averaged F1 across thresholds)
- Precision–Recall curves (micro-weighted)
- F1 vs threshold plots
- GO term prediction counts per protein

Metrics are saved as JSON files in: metrics/
Figures are saved to: figures/

## Figures Generated
Key plots used in the report:
- F1 vs threshold (per GO aspect)
- Precision–Recall curves (micro-weighted)
- Model comparison (embeddings vs IPR vs ensemble)
- GO term filtering statistics

BLAST Integration
BLAST is treated as an annotation transfer baseline and can be:
- Evaluated alone
- Blended with the embedding + IPR ensemble
- BLAST results must be provided as TSV files with:

```text
query   target   bits   evalue   score
```
BLAST is not required to run the core pipeline.

## Final Output

The final submission file follows the CAFA format:
```text
ProteinID   GO:XXXXXXX   score
```
Located at:
```text
outputs/submission_v4/submission_final.tsv
```

## Notes for Reviewers
- The full training set is large; hyperparameter tuning is performed on a subsample for efficiency.
- Thresholds are selected based on validation Fmax, not arbitrarily.
- Older scripts are retained for transparency and comparison.

## Authors

Natalya Lavrenchuk (ID 2141882)
Christina Caporale (ID 2141881)
Iuliia Osipova (ID 2148937)
Master’s Students — Data Science and Computational Chemistry
University of Padova



