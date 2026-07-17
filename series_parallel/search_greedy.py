"""
search_greedy.py — Greedy / beam-search construction of SP trees.

Covers:
  - _greedy_restart_worker_binary   (single restart, binary)
  - _greedy_build_tree_binary       (driver with parallel restarts)
  - _greedy_restart_worker_multiclass (single restart, multiclass)
  - _greedy_build_tree_multiclass   (driver with parallel restarts)

At each merge step the two components whose combination yields the
highest metric score are fused with either a series (AND) or parallel
(OR) operator.  beam_width > 1 keeps multiple candidate states alive
to escape local optima.  n_restarts > 1 runs independent searches
with randomised tie-breaking and returns the best result.

All N-1 merge steps are always executed (no early stopping).  The
global best (tree, score) is tracked across every step, including
individual leaves.  The ``allow_partial`` flag controls whether the
returned tree may use fewer than N models (global winner) or must be
the best fully-merged tree from the final step.
"""

import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score

from .trees import tree_to_string
from .scoring import (
    _vectorized_f1_binary,
    _vectorized_accuracy,
    _eval_probs_recursive,
)


# =====================================================================
# Greedy tree construction — multiclass
# =====================================================================

def _greedy_restart_worker_multiclass(preds_proba, Y, metric, beam_width,
                                       seed, add_noise, allow_partial):
    """Single restart of greedy beam search for multiclass.

    Always runs all N-1 merge steps and tracks the global best
    (tree, score) seen at any point, including individual leaves.

    Parameters
    ----------
    allow_partial : bool
        If True, return the global best tree (which may use fewer than
        N models).  If False, return only the best fully-merged tree
        from the final step.

    Returns (best_tree, best_score).
    """
    rng = np.random.RandomState(seed) if add_noise else None
    N = preds_proba.shape[0]

    initial_probas = {i: preds_proba[i] for i in range(N)}
    initial_trees  = {i: i              for i in range(N)}
    beams   = [(initial_probas, initial_trees)]
    next_id = N

    # Track global best across all steps (including individual leaves)
    global_best_score = -np.inf
    global_best_tree  = None

    for i in range(N):
        preds = np.argmax(preds_proba[i], axis=1)
        score = metric(Y, preds)
        if score > global_best_score:
            global_best_score = score
            global_best_tree  = i

    for _ in range(N - 1):
        candidates = []

        for pool_probas, pool_trees in beams:
            ids    = list(pool_probas.keys())
            n_pool = len(ids)

            for idx_a in range(n_pool):
                for idx_b in range(idx_a + 1, n_pool):
                    a, b   = ids[idx_a], ids[idx_b]
                    pa, pb = pool_probas[a], pool_probas[b]

                    for op in ('S', 'P'):
                        merged_p = pa * pb if op == 'S' else 1.0 - (1.0 - pa) * (1.0 - pb)
                        preds    = np.argmax(merged_p, axis=1)
                        score    = metric(Y, preds)
                        score_key = (score + rng.uniform(0, 1e-10)
                                     if add_noise else score)
                        candidates.append((
                            score_key, score, op, a, b,
                            merged_p, pool_probas, pool_trees
                        ))

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:beam_width]

        new_beams = []
        seen = set()
        for (score_key, score, op, a, b,
             merged_p, pool_probas, pool_trees) in candidates:

            new_probas = {k: v for k, v in pool_probas.items() if k not in (a, b)}
            new_trees  = {k: v for k, v in pool_trees.items()  if k not in (a, b)}

            merged_tree = (op, pool_trees[a], pool_trees[b])
            new_probas[next_id] = merged_p
            new_trees[next_id]  = merged_tree

            state_key = (tree_to_string(merged_tree), tuple(sorted(new_trees.keys())))
            if state_key not in seen:
                seen.add(state_key)
                new_beams.append((new_probas, new_trees))

                if score > global_best_score:
                    global_best_score = score
                    global_best_tree  = merged_tree

            next_id += 1

        beams = new_beams

    # Evaluate final trees from all surviving beams (score without noise)
    best_score = -np.inf
    best_tree  = None
    for pool_probas, pool_trees in beams:
        final_tree = list(pool_trees.values())[0]
        probs      = _eval_probs_recursive(final_tree, preds_proba)
        score      = metric(Y, np.argmax(probs, axis=1))
        if score > best_score:
            best_score = score
            best_tree  = final_tree

    if allow_partial and global_best_tree is not None and global_best_score > best_score:
        best_score = global_best_score
        best_tree  = global_best_tree

    return best_tree, best_score


def _greedy_build_tree_multiclass(preds_proba, Y, metric, beam_width=10,
                                   n_restarts=1, random_state=123,
                                   n_jobs=1, allow_partial=True):
    """Build an SP tree greedily by merging the best pair at each step.

    Always runs all N-1 merge steps and tracks the global best tree
    seen at any point.

    Restarts are run in parallel when n_jobs != 1 and n_restarts > 1.

    Parameters
    ----------
    preds_proba : np.ndarray, shape (N, m, C)
        Cached predict_proba outputs for all base models.
    Y : np.ndarray, shape (m,)
        True labels.
    metric : callable(y_true, y_pred) -> float
        Scoring function (higher is better).
    beam_width : int
        Number of candidate states kept at each merge step.
    n_restarts : int
        Number of independent greedy runs with randomised tie-breaking.
    random_state : int
        Seed for the random tie-breaking in restarts.
    n_jobs : int
        Number of parallel workers for restarts (-1 = all cores).
    allow_partial : bool
        If True, the returned tree may use fewer than N models (the
        global best across all merge steps).  If False, only the best
        fully-merged tree (using all N models) is returned.

    Returns
    -------
    best_tree : tuple
    best_score : float
    """
    N = preds_proba.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 models to build an SP tree.")

    seeds = [random_state + i * 997 for i in range(n_restarts)]

    if n_restarts == 1 or n_jobs == 1:
        results = [
            _greedy_restart_worker_multiclass(
                preds_proba, Y, metric, beam_width, seed,
                add_noise=(i > 0), allow_partial=allow_partial)
            for i, seed in enumerate(seeds)]
    else:
        n_workers = min(max(1, os.cpu_count() if n_jobs == -1 else n_jobs),
                        n_restarts)
        results = Parallel(n_jobs=n_workers)(
            delayed(_greedy_restart_worker_multiclass)(
                preds_proba, Y, metric, beam_width, seed,
                add_noise=(i > 0), allow_partial=allow_partial)
            for i, seed in enumerate(seeds))

    return max(results, key=lambda x: x[1])


# =====================================================================
# Greedy tree construction — binary
# =====================================================================

def _greedy_restart_worker_binary(predictions_bool, Y_bool, metric,
                                   beam_width, seed, allow_partial,
                                   add_noise, use_vec_f1, use_vec_acc):
    """Single restart of greedy beam search for binary.

    Always runs all N-1 merge steps and tracks the global best
    (tree, score) seen at any point, including individual leaves.

    Parameters
    ----------
    allow_partial : bool
        If True, return the global best tree (which may use fewer than
        N models).  If False, return only the best fully-merged tree
        from the final step.

    Returns (best_tree, best_score).
    """
    rng  = np.random.RandomState(seed) if add_noise else None
    N    = len(predictions_bool)
    preds = predictions_bool

    def _batch_score(merged_matrix):
        if use_vec_f1:
            return _vectorized_f1_binary(merged_matrix, Y_bool)
        elif use_vec_acc:
            return _vectorized_accuracy(merged_matrix, Y_bool)
        else:
            return np.array([metric(Y_bool, merged_matrix[i])
                             for i in range(merged_matrix.shape[0])])

    initial_preds = {i: preds[i] for i in range(N)}
    initial_trees = {i: i        for i in range(N)}
    beams = [(initial_preds, initial_trees)]

    # Track global best across all steps (including individual leaves)
    global_best_score = -np.inf
    global_best_tree  = None

    leaf_matrix = np.stack(preds)
    leaf_scores = _batch_score(leaf_matrix)
    best_leaf   = int(np.argmax(leaf_scores))
    global_best_score = leaf_scores[best_leaf]
    global_best_tree  = best_leaf

    next_id    = N
    best_score = -np.inf
    best_tree  = None

    for _ in range(N - 1):
        candidates_meta = []
        merged_list     = []

        for beam_idx, (pool_preds, pool_trees) in enumerate(beams):
            ids    = list(pool_preds.keys())
            n_pool = len(ids)

            for idx_a in range(n_pool):
                for idx_b in range(idx_a + 1, n_pool):
                    a, b   = ids[idx_a], ids[idx_b]
                    pa, pb = pool_preds[a], pool_preds[b]

                    for op in ('S', 'P'):
                        merged = (pa & pb) if op == 'S' else (pa | pb)
                        candidates_meta.append((op, a, b, beam_idx))
                        merged_list.append(merged)

        if not candidates_meta:
            break

        all_merged = np.stack(merged_list)
        scores     = _batch_score(all_merged)
        score_keys = (scores + rng.uniform(0, 1e-10, size=len(scores))
                      if add_noise else scores)

        order = np.argsort(-score_keys)

        new_beams = []
        seen      = set()

        for rank in order:
            if len(new_beams) >= beam_width:
                break

            op, a, b, beam_idx = candidates_meta[rank]
            score  = scores[rank]
            merged = merged_list[rank]
            pool_preds, pool_trees = beams[beam_idx]

            new_preds = {k: v for k, v in pool_preds.items() if k not in (a, b)}
            new_trees = {k: v for k, v in pool_trees.items() if k not in (a, b)}

            merged_tree = (op, pool_trees[a], pool_trees[b])
            new_preds[next_id] = merged
            new_trees[next_id] = merged_tree

            state_key = tuple(
                hash(np.packbits(new_preds[k]).data.tobytes())
                for k in sorted(new_preds.keys())
            )

            if state_key not in seen:
                seen.add(state_key)
                new_beams.append((new_preds, new_trees))

                if score > global_best_score:
                    global_best_score = score
                    global_best_tree  = merged_tree

            next_id += 1

        beams = new_beams

    # Evaluate final trees from surviving beams
    final_trees      = []
    final_preds_list = []
    for pool_preds, pool_trees in beams:
        if len(pool_trees) == 1:
            final_trees.append(list(pool_trees.values())[0])
            final_preds_list.append(list(pool_preds.values())[0])

    if final_preds_list:
        final_matrix  = np.stack(final_preds_list)
        final_scores  = _batch_score(final_matrix)
        best_final    = int(np.argmax(final_scores))
        if final_scores[best_final] > best_score:
            best_score = final_scores[best_final]
            best_tree  = final_trees[best_final]

    if allow_partial and global_best_tree is not None and global_best_score > best_score:
        best_score = global_best_score
        best_tree  = global_best_tree

    return best_tree, best_score


def _greedy_build_tree_binary(predictions_bool, Y, metric, beam_width=10,
                               n_restarts=1, random_state=123,
                               allow_partial=True, n_jobs=1):
    """Build an SP tree greedily for binary classification.

    Always runs all N-1 merge steps and tracks the global best tree
    seen at any point.

    Uses boolean AND (series) / OR (parallel) on hard predictions,
    with vectorised batch scoring (numpy) for F1 and accuracy.
    Functional deduplication ensures that two beam states whose pool
    components all produce the same output vectors are treated as
    equivalent, even if their tree structures differ.

    Parameters
    ----------
    predictions_bool : list of np.ndarray, each shape (m,)
        Hard predictions from each base model (castable to bool).
    Y : np.ndarray, shape (m,)
        True labels (castable to bool).
    metric : callable(y_true, y_pred) -> float
        Scoring function (higher is better).
    beam_width : int
        Number of candidate states kept at each merge step.
    n_restarts : int
        Independent runs with randomised tie-breaking.
    random_state : int
        Seed for the random tie-breaking in restarts.
    allow_partial : bool
        If True, the returned tree may use fewer than N models (the
        global best across all merge steps).  If False, only the best
        fully-merged tree (using all N models) is returned.
    n_jobs : int
        Number of parallel workers (-1 = all cores).

    Returns
    -------
    best_tree : tuple or int
    best_score : float
    """
    N = len(predictions_bool)
    if N < 2:
        raise ValueError("Need at least 2 models to build an SP tree.")

    Y_bool = np.asarray(Y, dtype=np.bool_)
    preds  = [np.asarray(p, dtype=np.bool_) for p in predictions_bool]

    use_vec_f1  = (metric is f1_score)
    use_vec_acc = (metric is accuracy_score)

    seeds = [random_state + i * 997 for i in range(n_restarts)]

    if n_restarts == 1 or n_jobs == 1:
        results = [
            _greedy_restart_worker_binary(
                preds, Y_bool, metric, beam_width, seed,
                allow_partial, add_noise=(i > 0),
                use_vec_f1=use_vec_f1, use_vec_acc=use_vec_acc)
            for i, seed in enumerate(seeds)]
    else:
        n_workers = min(max(1, os.cpu_count() if n_jobs == -1 else n_jobs),
                        n_restarts)
        results = Parallel(n_jobs=n_workers)(
            delayed(_greedy_restart_worker_binary)(
                preds, Y_bool, metric, beam_width, seed,
                allow_partial, add_noise=(i > 0),
                use_vec_f1=use_vec_f1, use_vec_acc=use_vec_acc)
            for i, seed in enumerate(seeds))

    return max(results, key=lambda x: x[1])