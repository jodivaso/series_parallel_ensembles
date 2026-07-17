"""
search_random.py — Time-limited parallel random search for SP trees.

Each worker generates random SP trees (by repeatedly merging two
randomly chosen components) and evaluates them against the metric until
its individual time budget expires.  Workers compute their own deadline
from the moment they start, so joblib process-spawning overhead does
not consume the search budget.

Covers:
  - _random_search_worker_binary   / _random_search_binary
  - _random_search_worker_multiclass / _random_search_multiclass
"""

import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score

from .trees import _generate_random_tree
from .compilation import compile_tree, execute_binary
from .scoring import (
    _vectorized_f1_binary,
    _vectorized_accuracy,
    _eval_probs_recursive,
)


# =====================================================================
# Random search — binary
# =====================================================================

def _random_search_worker_binary(predictions_bool, Y_bool, metric,
                                  time_limit_seconds, N, seed,
                                  use_vec_f1, use_vec_acc):
    """Worker: generate and evaluate random SP trees for time_limit_seconds.

    Each worker computes its own deadline from its own start time, so
    joblib process-spawning overhead does not eat into the search budget.

    Returns (best_tree, best_score, n_evaluated).
    """
    rng        = np.random.RandomState(seed)
    best_score = -np.inf
    best_tree  = None
    n_evaluated = 0
    deadline   = time.time() + time_limit_seconds

    while True:
        tree  = _generate_random_tree(N, rng)
        preds = execute_binary(compile_tree(tree), predictions_bool)

        if use_vec_f1:
            score = _vectorized_f1_binary(preds.reshape(1, -1), Y_bool)[0]
        elif use_vec_acc:
            score = _vectorized_accuracy(preds.reshape(1, -1), Y_bool)[0]
        else:
            score = metric(Y_bool, preds)

        n_evaluated += 1
        if score > best_score:
            best_score = score
            best_tree  = tree

        # Check deadline every 64 iterations to reduce time.time() overhead
        if n_evaluated % 64 == 0 and time.time() >= deadline:
            break

    return best_tree, best_score, n_evaluated


def _random_search_binary(predictions_bool, Y, metric, time_limit_seconds,
                           n_jobs=-1, random_state=123):
    """Time-limited parallel random search for binary classification.

    Parameters
    ----------
    predictions_bool : list of np.ndarray, each shape (m,)
        Hard predictions from each base model.
    Y : np.ndarray, shape (m,)
        True labels.
    metric : callable(y_true, y_pred) -> float
        Scoring function (higher is better).
    time_limit_seconds : float
        Per-worker search budget in seconds.
    n_jobs : int
        Number of parallel workers (-1 = all cores).
    random_state : int
        Base seed (each worker gets a derived seed).

    Returns
    -------
    best_tree : tuple
    best_score : float
    total_evaluated : int
    """
    N      = len(predictions_bool)
    Y_bool = np.asarray(Y, dtype=np.bool_)
    preds  = [np.asarray(p, dtype=np.bool_) for p in predictions_bool]

    use_vec_f1  = (metric is f1_score)
    use_vec_acc = (metric is accuracy_score)

    n_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    n_cores = max(1, n_cores)
    # Cap workers so each gets at least ~5 s of productive work.
    max_workers = max(1, int(time_limit_seconds // 5))
    n_workers   = min(n_cores, max_workers)

    seeds = [random_state + i * 997 for i in range(n_workers)]

    if n_workers == 1:
        results = [_random_search_worker_binary(
            preds, Y_bool, metric, time_limit_seconds, N, seeds[0],
            use_vec_f1, use_vec_acc)]
    else:
        results = Parallel(n_jobs=n_workers)(
            delayed(_random_search_worker_binary)(
                preds, Y_bool, metric, time_limit_seconds, N, seed,
                use_vec_f1, use_vec_acc)
            for seed in seeds)

    best_tree, best_score, _ = max(results, key=lambda x: x[1])
    total_evaluated = sum(r[2] for r in results)
    return best_tree, best_score, total_evaluated


# =====================================================================
# Random search — multiclass
# =====================================================================

def _random_search_worker_multiclass(preds_proba, Y, metric,
                                      time_limit_seconds, N, seed):
    """Worker: generate and evaluate random SP trees for time_limit_seconds.

    Returns (best_tree, best_score, n_evaluated).
    """
    rng         = np.random.RandomState(seed)
    best_score  = -np.inf
    best_tree   = None
    n_evaluated = 0
    deadline    = time.time() + time_limit_seconds

    while True:
        tree        = _generate_random_tree(N, rng)
        probs       = _eval_probs_recursive(tree, preds_proba)
        pred_classes = np.argmax(probs, axis=1)
        score       = metric(Y, pred_classes)

        n_evaluated += 1
        if score > best_score:
            best_score = score
            best_tree  = tree

        # Check deadline every 16 iterations (multiclass is slower per tree)
        if n_evaluated % 16 == 0 and time.time() >= deadline:
            break

    return best_tree, best_score, n_evaluated


def _random_search_multiclass(preds_proba, Y, metric, time_limit_seconds,
                               n_jobs=-1, random_state=123):
    """Time-limited parallel random search for multiclass classification.

    Parameters
    ----------
    preds_proba : np.ndarray, shape (N, m, C)
        Cached predict_proba outputs for all base models.
    Y : np.ndarray, shape (m,)
        True labels.
    metric : callable(y_true, y_pred) -> float
        Scoring function (higher is better).
    time_limit_seconds : float
        Per-worker search budget in seconds.
    n_jobs : int
        Number of parallel workers (-1 = all cores).
    random_state : int
        Base seed.

    Returns
    -------
    best_tree : tuple
    best_score : float
    total_evaluated : int
    """
    N = preds_proba.shape[0]

    n_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    n_cores = max(1, n_cores)
    max_workers = max(1, int(time_limit_seconds // 5))
    n_workers   = min(n_cores, max_workers)

    seeds = [random_state + i * 997 for i in range(n_workers)]

    if n_workers == 1:
        results = [_random_search_worker_multiclass(
            preds_proba, Y, metric, time_limit_seconds, N, seeds[0])]
    else:
        results = Parallel(n_jobs=n_workers)(
            delayed(_random_search_worker_multiclass)(
                preds_proba, Y, metric, time_limit_seconds, N, seed)
            for seed in seeds)

    best_tree, best_score, _ = max(results, key=lambda x: x[1])
    total_evaluated = sum(r[2] for r in results)
    return best_tree, best_score, total_evaluated
