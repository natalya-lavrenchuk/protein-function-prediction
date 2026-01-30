from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

def plot_go_term_frequency_hist(
    Y,
    aspect: str,
    bins: int = 50,
    log_y: bool = True
) -> None:
    # Convert sparse to dense if needed
    if hasattr(Y, "toarray"):
        Y = Y.toarray()

    # Compute GO term frequencies
    term_freq = np.asarray(Y.sum(axis=0)).ravel()

    # Summary statistics
    n_terms = term_freq.size
    min_f = int(term_freq.min())
    med_f = int(np.median(term_freq))
    max_f = int(term_freq.max())

    plt.figure(figsize=(7, 4))
    plt.hist(term_freq, bins=bins)

    if log_y:
        plt.yscale("log")

    plt.xlabel("Number of proteins annotated with GO term")
    plt.ylabel("Number of GO terms" + (" (log scale)" if log_y else ""))
    plt.title(f"GO term frequency distribution: {aspect}")

    # Annotate stats in upper-right corner
    stats_text = (
        f"Terms: {n_terms}\n"
        f"Min: {min_f}\n"
        f"Median: {med_f}\n"
        f"Max: {max_f}" )

    plt.text(
        0.98, 0.95,
        stats_text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85) )

    plt.tight_layout()
    plt.show()
