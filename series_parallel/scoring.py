"""
scoring.py — Vectorised scoring helpers for binary and multiclass modes.

Binary:
  - Truth-table vectorisation: evaluate every SP tree on the 2^N possible
    input patterns (tiny), then map the M real samples via a fancy-index
    lookup.  Histogram-based F1 and accuracy avoid materialising the
    (n_trees × M) prediction matrix.
  - Compiled fallback for custom metrics or N > _MAX_N_FOR_TRUTH_TABLE.

Multiclass:
  - Recursive probability composition (_eval_probs_recursive)
  - Batch scorer for joblib workers (_score_batch_multiclass)
"""

import numpy as np

from .compilation import compile_tree, execute_binary


# Maximum N for which truth-table vectorisation is used.
# 2^20 ≈ 1M patterns — beyond that we fall back to compiled evaluation.
_MAX_N_FOR_TRUTH_TABLE = 20


# =====================================================================
# Vectorised metric helpers (binary)
# =====================================================================

def _vectorized_f1_binary(all_preds_bool, Y_bool):
    TP = np.logical_and(all_preds_bool, Y_bool).sum(axis=1)
    FP = np.logical_and(all_preds_bool, ~Y_bool).sum(axis=1)
    FN = np.logical_and(~all_preds_bool, Y_bool).sum(axis=1)
    denom = (2 * TP + FP + FN).astype(np.float64)
    return np.where(denom > 0, 2.0 * TP / denom, 0.0)


def _vectorized_accuracy(all_preds_bool, Y_bool):
    return (all_preds_bool == Y_bool).mean(axis=1)


# =====================================================================
# Truth-table vectorisation (binary only)
# =====================================================================
# For N binary models there are only 2^N distinct input patterns.
# We evaluate every SP tree on these 2^N patterns (tiny arrays),
# producing a truth table per tree.  Then we map each of the M samples
# to its pattern index and do a single numpy fancy-index lookup to get
# all (n_trees × M) predictions at once.

def _build_truth_tables(trees, N):
    """Return bool array (n_trees, 2^N): truth table for each tree."""
    n_patterns = 1 << N
    pattern_preds = [
        np.array([(i >> j) & 1 for i in range(n_patterns)], dtype=np.bool_)
        for j in range(N)
    ]
    tables = np.empty((len(trees), n_patterns), dtype=np.bool_)
    for t, tree in enumerate(trees):
        tables[t] = execute_binary(compile_tree(tree), pattern_preds)
    return tables


def _predictions_to_pattern_indices(predictions, N):
    """Map M boolean prediction vectors → integer pattern indices in [0, 2^N)."""
    indices = np.zeros(len(predictions[0]), dtype=np.intp)
    for j in range(N):
        indices |= np.asarray(predictions[j], dtype=np.intp) << j
    return indices


def _score_f1_from_truth_tables(truth_tables, pattern_indices, Y_bool):
    """Compute F1 for all trees via pattern histogram — O(n_trees × 2^N).

    Instead of expanding to (n_trees, M) and summing over M samples,
    we count how many positive/negative samples fall into each of the
    2^N patterns (a tiny histogram), then compute TP/FP/FN as dot
    products of (n_trees, 2^N) × (2^N,).
    """
    n_patterns = truth_tables.shape[1]

    pos_hist = np.bincount(
        pattern_indices[Y_bool], minlength=n_patterns).astype(np.float64)
    neg_hist = np.bincount(
        pattern_indices[~Y_bool], minlength=n_patterns).astype(np.float64)

    tt = truth_tables.astype(np.float64)

    TP = tt @ pos_hist       # (n_trees,)
    FP = tt @ neg_hist       # (n_trees,)
    FN = pos_hist.sum() - TP

    denom = 2.0 * TP + FP + FN
    return np.where(denom > 0, 2.0 * TP / denom, 0.0)


def _score_accuracy_from_truth_tables(truth_tables, pattern_indices, Y_bool):
    """Compute accuracy for all trees via pattern histogram."""
    n_patterns = truth_tables.shape[1]
    M = len(Y_bool)

    pos_hist = np.bincount(
        pattern_indices[Y_bool], minlength=n_patterns).astype(np.float64)
    neg_hist = np.bincount(
        pattern_indices[~Y_bool], minlength=n_patterns).astype(np.float64)

    tt = truth_tables.astype(np.float64)

    # Correct = predicted_positive ∩ actual_positive
    #         + predicted_negative ∩ actual_negative
    correct = tt @ pos_hist + (1.0 - tt) @ neg_hist
    return correct / M


# =====================================================================
# Recursive multiclass helpers (tuple-based, top-level for joblib)
# =====================================================================

def _eval_probs_recursive(tree, preds_proba):
    """Evaluate a single tuple-tree recursively using pre-computed probabilities."""
    if isinstance(tree, int):
        return preds_proba[tree]
    op, left, right = tree
    p_l = _eval_probs_recursive(left, preds_proba)
    p_r = _eval_probs_recursive(right, preds_proba)
    if op == 'P':
        return 1 - (1 - p_l) * (1 - p_r)
    else:
        return p_l * p_r


def _score_batch_multiclass(trees_batch, preds_proba, Y, metric):
    """Score a batch of tuple-trees (for joblib workers)."""
    scores = []
    for tree in trees_batch:
        probs = _eval_probs_recursive(tree, preds_proba)
        preds = np.argmax(probs, axis=1)
        scores.append(metric(Y, preds))
    return scores
