# Series-Parallel Ensembles

This repository contains the code, data and analyses that accompany the paper
**"Series-parallel ensembles"** (J. Divasón, F. J. Martinez-de-Pison, E. Sáenz-de-Cabezón).

Series-parallel ensembles are a reliability-theoretic framework for combining a fixed pool of
**pretrained** base classifiers. Each base learner is treated as a component of a system, and the
learners are combined through a **series-parallel structure function**: series connections behave as
restrictive filters (logical AND for binary tasks / product of confidences for multiclass), while
parallel connections provide alternative decision paths (logical OR / co-product). The ensemble is
trained by searching for the structure that optimises a validation metric, **without ever refitting the
base learners**. The framework also provides reliability-inspired importance measures that indicate
which base learner is most worth improving.

Everything needed to fully reproduce the experiments and the statistical analysis of the paper is
included here.

---

## Repository structure

```
├───experiments
│       datasets.7z
│       Experiments_binary.ipynb
│       Experiments_multiclass.ipynb
│       Experiments_multiclass_autogluon.ipynb
│       importance_experiments_binary.ipynb
│       importance_results_level1.csv
│       importance_results_level2.csv
│       possible_trees.pickle
│       possible_trees_compact.gz
│       results_binary_2026_05_08_09_35_16.csv
│       results_multiclass2026_04_24_12_37_04.csv
│       results_multiclass_autogluon_2026_05_05_10_17_58.csv
│
├───generate_all_trees
│       generate_sp_trees.ipynb
│
├───series_parallel
│       compilation.py
│       ensemble.py
│       importance.py
│       io.py
│       scoring.py
│       search_greedy.py
│       search_random.py
│       series_parallel_random.py
│       trees.py
│       __init__.py
│
└───statistical_analyses
        importance_statistical_analysis.ipynb
        statistical_analysis_binary.ipynb
        statistical_analysis_multiclass.ipynb
        statistical_analysis_multiclass_autogluon.ipynb
```

---

## What each folder contains

### `series_parallel/` — the library

This is the Python package that implements the whole method: tree data structures, compilation and
evaluation of series-parallel structures, exhaustive / greedy / random search strategies, vectorised
scoring for binary and multiclass tasks, loading of the precomputed tree catalogues, and the
importance measures. The public entry point is the `SeriesParallelEnsemble` class. Importing the
package with `from series_parallel import *` exposes all the classes and helper functions used by the
experiment notebooks.

### `experiments/` — experiment notebooks, data and raw results

This folder holds the notebooks that run the experiments of the paper, the preprocessed datasets, the
precomputed catalogue of all series-parallel structures, and the CSV files with the raw per-run
results. Each notebook corresponds to one of the four experiment blocks of the paper (see the mapping
table below).

- `datasets.7z` — the preprocessed OpenML datasets (one-hot encoded, standardised, missing values
  imputed), so the results can be reproduced exactly.
- `possible_trees_compact.gz` / `possible_trees.pickle` — the catalogue of **all** unique
  series-parallel structures for a given number of base models, used by the exhaustive search strategy.
  The compact gzip file is the one loaded by default; the pickle is kept as a legacy alternative. Both
  are produced by the notebook in `generate_all_trees/`.
- `results_*.csv` and `importance_results_*.csv` — the raw outputs consumed by the notebooks in
  `statistical_analyses/`.

### `generate_all_trees/` — catalogue generation

`generate_sp_trees.ipynb` enumerates every unique series-parallel structure with *N* labelled leaves
and serialises them to the compact binary format (`possible_trees_compact.gz`). This only needs to be
run once; the resulting file is already provided in `experiments/`.

### `statistical_analyses/` — statistical tests and figures

The notebooks here take the raw `*.csv` result files and produce the statistical comparisons reported
in the paper (Friedman omnibus tests, Holm-corrected pairwise Wilcoxon and Nemenyi tests, Cliff's
δ effect sizes, win-count matrices, predictive-validity correlations, hit rate / regret, etc.).

---

## Mapping between files and experiments in the paper

| Paper experiment | Experiment notebook | Raw results | Statistical analysis |
|---|---|---|---|
| **Exp. 1** — Binary datasets (43 datasets, exhaustive SP vs. Voting / Stacking / RF / XGBoost) | `experiments/Experiments_binary.ipynb` | `results_binary_2026_05_08_09_35_16.csv` | `statistical_analyses/statistical_analysis_binary.ipynb` |
| **Exp. 2** — Multiclass datasets (20 datasets, exhaustive SP vs. baselines) | `experiments/Experiments_multiclass.ipynb` | `results_multiclass2026_04_24_12_37_04.csv` | `statistical_analyses/statistical_analysis_multiclass.ipynb` |
| **Exp. 3** — Greedy / random SP on the AutoGluon L1 pool | `experiments/Experiments_multiclass_autogluon.ipynb` | `results_multiclass_autogluon_2026_05_05_10_17_58.csv` | `statistical_analyses/statistical_analysis_multiclass_autogluon.ipynb` |
| **Exp. 4** — Importance measures (validity, decision utility and robustness) | `experiments/importance_experiments_binary.ipynb` | `importance_results_level1.csv`, `importance_results_level2.csv` | `statistical_analyses/importance_statistical_analysis.ipynb` |

---

## Quick usage example

The `series_parallel` package must be importable (e.g. run from the repository root, or add it to your
`PYTHONPATH`). A minimal binary example:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
import pandas as pd
from series_parallel import SeriesParallelEnsemble

# X, y : your data (numpy arrays, here a scikit-learn dataset as an example)
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=0)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=0)

estimators = [
    ("knn",   KNeighborsClassifier()),
    ("dt",    DecisionTreeClassifier(random_state=0)),
    ("logit", LogisticRegression(max_iter=4000)),
    ("gnb",   GaussianNB()),
]

# The 'exhaustive' strategy loads possible_trees_compact.gz from the working directory
# (or pass a path via trees="/path/to/possible_trees_compact.gz").
ens = SeriesParallelEnsemble(estimators, metric=f1_score, strategy="exhaustive")
ens.fit_estimators(X_tr, y_tr)   # 1) train the base learners on the train split
ens.fit(X_val, y_val)            # 2) search the best SP structure on the validation split
y_pred = ens.predict(X_te)       # 3) predict on the held-out test split

# --- Evaluation ---
def scores(y_true, y_hat):
    return {
        "accuracy":  accuracy_score(y_true, y_hat),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall":    recall_score(y_true, y_hat, zero_division=0),
        "f1":        f1_score(y_true, y_hat, zero_division=0),
    }

rows = {name: scores(y_te, model.predict(X_te)) for name, model in ens.estimators}
rows["SP-ensemble"] = scores(y_te, y_pred)

results = pd.DataFrame(rows).T.round(4)
print(results.to_string())
print("\nBest structure:", ens.best_tree_as_string())
```

Notes:

- **Multiclass tasks:** create the ensemble with `mode="multiclass"` and predict with
  `ens.predict_multiclass(X_te)`.
- **Larger pools:** when exhaustive enumeration becomes impractical (roughly `N > 8`), use
  `strategy="greedy"` or `strategy="random"`; these do **not** require the precomputed tree catalogue.
- **Importance measures:** see `series_parallel/importance.py` (`flip_importance`, `oracle_importance`,
  `degradation_importance`, `birnbaum_importance`, `compute_all_importances`).

---

## Requirements

The experiments were run with **Python 3.12.4**. Main libraries and the versions used:

| Library | Version |
|---|---|
| numpy | 2.0.0 |
| pandas | 2.2.2 |
| scikit-learn | 1.7.2 |
| xgboost | 3.2.0 |

Additional dependencies used by the notebooks / package: `scipy` and `joblib` (scoring, statistical
tests and parallelism), `mlxtend` (voting baselines), `bayes_opt` / `bayesian-optimization`
(hyperparameter tuning of base learners and baselines), and `matplotlib` (figures). `autogluon` is only
needed for Experiment 3. Optionally, `PrettyPrint` and `colorama` enable pretty-printing of the best
tree.

Experiments were carried out on a server with 2× AMD EPYC 7452 and 256 GB RAM, running Ubuntu 22.04.

---

## Reproducing the results

1. Unpack `experiments/datasets.7z`.
2. (Optional) Regenerate the tree catalogue with `generate_all_trees/generate_sp_trees.ipynb`; the
   catalogue `experiments/possible_trees_compact.gz` is already provided.
3. Run the experiment notebooks in `experiments/` to produce the `*.csv` result files (already provided
   as well).
4. Run the corresponding notebooks in `statistical_analyses/` to obtain the tables, tests and figures
   reported in the paper.

## Data availability

The original datasets were retrieved automatically from OpenML. The preprocessed versions
(one-hot encoded, scaled, missing values handled and so on) are provided in `experiments/datasets.7z`
to guarantee full reproducibility.
