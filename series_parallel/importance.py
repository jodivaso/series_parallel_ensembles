"""
importance.py — Structural importance measures for Series-Parallel ensembles.

Provides empirical importance measures that quantify how much each base
model contributes to the ensemble's predictions, operating directly on
cached predictions without probabilistic independence assumptions.

Measures implemented
--------------------

flip_importance(predictions, tree)
    Metric-agnostic structural sensitivity.  For each model i and each
    sample, checks whether flipping model i's prediction changes the
    ensemble output.  Returns the fraction of samples where model i is
    "critical" (i.e., the ensemble output depends on it).

    Strengths:
      - Pure structural measure: does not depend on any target metric.
      - No assumptions about error distributions or independence.
      - Cheap: O(N × m) using compiled evaluation.

    Limitations:
      - Treats all flips equally: a flip that corrects a false positive
        and one that introduces a false negative have the same weight.
      - Does not tell you the *direction* of the effect (beneficial or
        harmful), only that the model matters.

oracle_importance(predictions, Y, tree, metric)
    Measures how much the ensemble metric improves when model i is
    replaced by a perfect predictor (ground truth).  Answers the
    question: "if I could make model i perfect, how much would the
    ensemble gain?"

    Strengths:
      - Directly aligned with the target metric (F1, accuracy, etc.).
      - Naturally identifies the model with the most *room for useful
        improvement* within the ensemble structure.

    Limitations:
      - Biased towards models that currently make many errors: a model
        with 60% accuracy in a structurally key position will show very
        high oracle importance simply because there is more to gain,
        even if improving it in practice is hard.
      - The "perfect predictor" scenario is unrealistic — it gives an
        upper bound on the benefit, not an expected gain.

degradation_importance(predictions, Y, tree, metric)
    Measures how much the ensemble metric drops when model i is replaced
    by a trivial predictor (all zeros / all negatives).  Answers the
    question: "how much does the ensemble depend on model i working well?"

    Strengths:
      - Identifies models whose failure is catastrophic for the ensemble.
      - Complementary to oracle_importance: high oracle + high degradation
        means the model is both improvable and critical.

    Limitations:
      - Biased towards models that are already good: a high-accuracy
        model in series will show high degradation importance because
        breaking it breaks the chain, but it may already be near its
        ceiling.
      - The degradation scenario (all zeros) is somewhat arbitrary;
        "all ones" or "random" would give different rankings.


birnbaum_importance(tree_tuple, reliabilities)
    Analytical Birnbaum importance from reliability theory.  For each
    model i, computes h(1_i, p) - h(0_i, p), where h is the reliability
    polynomial of the SP system and p_i is the "reliability" of model i
    (typically its accuracy on the validation set).

    Following Kuo & Zhu (2012, Definition 4.1.1), the total importance
    decomposes into two components:

        I_Bs(i) = h(1_i, p) - h(p)   — importance for system success
        I_Bf(i) = h(p) - h(0_i, p)   — importance for system failure
        I_B(i)  = I_Bs(i) + I_Bf(i)  — total

    I_Bs is the analytical analogue of oracle_importance: how much does
    the system reliability improve when component i is made perfect.
    I_Bf is the analytical analogue of degradation_importance: how much
    does the system reliability drop when component i fails completely.

    Strengths:
      - Closed-form, O(N²) computation — no data needed beyond accuracy.
      - Directly rooted in reliability theory: this is the classical
        structural importance measure for coherent systems.
      - Provides insight into the *structural position* of each model
        independent of the specific samples.

    Limitations:
      - Assumes component failures are independent Bernoulli events.
        In practice, ML models fail on correlated regions of the input
        space, violating this assumption.
      - Uses accuracy as a proxy for reliability, but accuracy is not a
        probability in the reliability-theoretic sense: a model with
        accuracy 0.90 does not "fail at random with probability 0.10"
        — it fails deterministically on specific inputs.
      - Cannot use F1 or other non-probabilistic metrics as reliability
        values, since the series/parallel composition formulas assume
        probabilities of independent events.
      - Despite these limitations, the analytical Birnbaum provides a
        useful baseline for comparison with the empirical measures.


Practical guidance
------------------
For the use case of "which model should I invest in optimizing":

- High flip_importance + high oracle_importance → best candidate.
  The model is structurally influential AND has room to improve.

- High flip_importance + low oracle_importance → already good, protect it.
  It's in a sensitive position but already contributing well.

- Low flip_importance + high oracle_importance → structurally shielded.
  Even if improved, the ensemble structure limits its impact.

The importance values depend on the fixed SP tree. After optimizing a
model, the optimal tree may change — so importance should be recomputed
after each round of optimization.
"""

import numpy as np

from .compilation import compile_tree, execute_binary


# =====================================================================
# Flip Importance (metric-agnostic structural sensitivity)
# =====================================================================

def flip_importance(predictions, compiled_tree):
    """Compute flip importance for each model in a binary SP ensemble.

    For each model i, measures the fraction of samples where flipping
    model i's prediction changes the ensemble output.

    Parameters
    ----------
    predictions : list of np.ndarray, each shape (m,)
        Hard predictions (0/1 or bool) from each base model, indexed
        by model position in the tree (0, 1, ..., N-1).
    compiled_tree : tuple of (opcode, operand) pairs
        Compiled SP tree (from compile_tree).

    Returns
    -------
    importances : np.ndarray, shape (N,)
        Fraction of samples where each model is critical.
        Values in [0, 1].
    """
    N = len(predictions)
    m = len(predictions[0])
    preds = [np.asarray(p, dtype=np.bool_) for p in predictions]

    # Baseline ensemble output
    baseline = execute_binary(compiled_tree, preds)

    importances = np.empty(N, dtype=np.float64)

    for i in range(N):
        # Create a copy with model i flipped
        flipped = list(preds)
        flipped[i] = ~preds[i]

        flipped_output = execute_binary(compiled_tree, flipped)
        importances[i] = np.mean(baseline != flipped_output)

    return importances


# =====================================================================
# Oracle Importance (metric gain from perfect model)
# =====================================================================

def oracle_importance(predictions, Y, compiled_tree, metric):
    """Compute oracle importance for each model in a binary SP ensemble.

    For each model i, replaces its predictions with the ground truth
    and measures the metric improvement.

    Parameters
    ----------
    predictions : list of np.ndarray, each shape (m,)
        Hard predictions (0/1 or bool) from each base model.
    Y : np.ndarray, shape (m,)
        True labels.
    compiled_tree : tuple of (opcode, operand) pairs
        Compiled SP tree (from compile_tree).
    metric : callable(y_true, y_pred) -> float
        Scoring function (e.g., f1_score, accuracy_score).

    Returns
    -------
    importances : np.ndarray, shape (N,)
        Metric improvement when each model becomes perfect.
        Non-negative values; higher means more room for useful improvement.
    """
    N = len(predictions)
    preds = [np.asarray(p, dtype=np.bool_) for p in predictions]
    Y_bool = np.asarray(Y, dtype=np.bool_)

    # Baseline ensemble score
    baseline_output = execute_binary(compiled_tree, preds)
    baseline_score = metric(Y, baseline_output)

    importances = np.empty(N, dtype=np.float64)

    for i in range(N):
        oracle_preds = list(preds)
        oracle_preds[i] = Y_bool  # Perfect predictions

        oracle_output = execute_binary(compiled_tree, oracle_preds)
        oracle_score = metric(Y, oracle_output)
        importances[i] = oracle_score - baseline_score

    return importances


# =====================================================================
# Degradation Importance (metric loss from broken model)
# =====================================================================

def degradation_importance(predictions, Y, compiled_tree, metric):
    """Compute degradation importance for each model in a binary SP ensemble.

    For each model i, replaces its predictions with all zeros (always
    predicts negative class) and measures how much the metric drops.

    Parameters
    ----------
    predictions : list of np.ndarray, each shape (m,)
        Hard predictions (0/1 or bool) from each base model.
    Y : np.ndarray, shape (m,)
        True labels.
    compiled_tree : tuple of (opcode, operand) pairs
        Compiled SP tree (from compile_tree).
    metric : callable(y_true, y_pred) -> float
        Scoring function (e.g., f1_score, accuracy_score).

    Returns
    -------
    importances : np.ndarray, shape (N,)
        Metric drop when each model is degraded to constant zero.
        Non-negative values; higher means the ensemble depends more on
        this model.
    """
    N = len(predictions)
    m = len(predictions[0])
    preds = [np.asarray(p, dtype=np.bool_) for p in predictions]

    # Baseline ensemble score
    baseline_output = execute_binary(compiled_tree, preds)
    baseline_score = metric(Y, baseline_output)

    importances = np.empty(N, dtype=np.float64)
    zeros = np.zeros(m, dtype=np.bool_)

    for i in range(N):
        degraded_preds = list(preds)
        degraded_preds[i] = zeros

        degraded_output = execute_binary(compiled_tree, degraded_preds)
        degraded_score = metric(Y, degraded_output)
        importances[i] = baseline_score - degraded_score

    return importances


# =====================================================================
# Analytical Birnbaum Importance (reliability-theoretic)
# =====================================================================

def _system_reliability(tree, reliabilities):
    """Compute the reliability polynomial h(p) for an SP system.

    Parameters
    ----------
    tree : int or tuple
        SP tree in tuple format: int for leaf, ('S'/'P', left, right).
    reliabilities : list of float
        Component reliabilities p_i (typically accuracy of each model).

    Returns
    -------
    h : float
        System reliability under the independence assumption.
    """
    if isinstance(tree, int):
        return reliabilities[tree]
    op, left, right = tree
    h_left = _system_reliability(left, reliabilities)
    h_right = _system_reliability(right, reliabilities)
    if op == 'P':
        return 1.0 - (1.0 - h_left) * (1.0 - h_right)
    else:
        return h_left * h_right


def birnbaum_importance(tree_tuple, reliabilities, decompose=False):
    """Compute analytical Birnbaum importance for each model.

    I_i^B = h(1_i, p) - h(0_i, p)

    Following Kuo & Zhu (2012, Definition 4.1.1), the total Birnbaum
    importance decomposes into two components:

        I_Bs(i) = h(1_i, p) - h(p)    (importance for system success)
        I_Bf(i) = h(p) - h(0_i, p)    (importance for system failure)
        I_B(i)  = I_Bs(i) + I_Bf(i)   (total)

    I_Bs measures how much more likely the system is to work given that
    component i works (analogous to oracle importance).  I_Bf measures
    how much more likely the system is to fail given that component i
    has failed (analogous to degradation importance).

    Parameters
    ----------
    tree_tuple : int or tuple
        SP tree in tuple format (as stored in SeriesParallelEnsemble._best_tree_tuple).
    reliabilities : list or np.ndarray of float, length N
        Component reliabilities (typically accuracy of each base model
        on the validation set).
    decompose : bool, default False
        If True, return a dict with 'total', 'success', and 'failure'
        arrays.  If False, return only the total (backward compatible).

    Returns
    -------
    importances : np.ndarray, shape (N,) — if decompose is False
        Birnbaum importance of each model.  Values in [0, 1].
    importances : dict — if decompose is True
        'total'   : np.ndarray (N,) — I_B  = h(1_i,p) - h(0_i,p)
        'success' : np.ndarray (N,) — I_Bs = h(1_i,p) - h(p)
        'failure' : np.ndarray (N,) — I_Bf = h(p) - h(0_i,p)
    """
    N = len(reliabilities)
    reliabilities = list(reliabilities)

    total   = np.empty(N, dtype=np.float64)
    success = np.empty(N, dtype=np.float64)
    failure = np.empty(N, dtype=np.float64)

    # System reliability with actual reliabilities
    h_p = _system_reliability(tree_tuple, reliabilities)

    for i in range(N):
        saved = reliabilities[i]

        reliabilities[i] = 1.0
        h_1 = _system_reliability(tree_tuple, reliabilities)

        reliabilities[i] = 0.0
        h_0 = _system_reliability(tree_tuple, reliabilities)

        total[i]   = h_1 - h_0        # I_B
        success[i] = h_1 - h_p        # I_Bs
        failure[i] = h_p - h_0        # I_Bf

        reliabilities[i] = saved

    if decompose:
        return {'total': total, 'success': success, 'failure': failure}

    return total


# =====================================================================
# Summary: all measures at once
# =====================================================================

def compute_all_importances(predictions, Y, compiled_tree, metric,
                            model_names=None, tree_tuple=None,
                            reliabilities=None):
    """Compute all importance measures and return a summary dict.

    Parameters
    ----------
    predictions : list of np.ndarray, each shape (m,)
        Hard predictions from each base model.
    Y : np.ndarray, shape (m,)
        True labels.
    compiled_tree : tuple of (opcode, operand) pairs
        Compiled SP tree.
    metric : callable(y_true, y_pred) -> float
        Scoring function.
    model_names : list of str or None
        Optional names for each model (for display).
    tree_tuple : tuple or None
        SP tree in tuple format (needed for Birnbaum).  If None,
        Birnbaum importance is not computed.
    reliabilities : list of float or None
        Component reliabilities for Birnbaum (typically accuracy on VAL).
        If None, Birnbaum importance is not computed.

    Returns
    -------
    result : dict with keys:
        'flip' : np.ndarray (N,)
        'oracle' : np.ndarray (N,)
        'degradation' : np.ndarray (N,)
        'birnbaum' : np.ndarray (N,) or None
        'birnbaum_success' : np.ndarray (N,) or None  — I_Bs (Kuo 4.1)
        'birnbaum_failure' : np.ndarray (N,) or None  — I_Bf (Kuo 4.1)
        'model_names' : list of str
        'baseline_score' : float
    """
    N = len(predictions)
    if model_names is None:
        model_names = [f"model_{i}" for i in range(N)]

    baseline_output = execute_binary(
        compiled_tree,
        [np.asarray(p, dtype=np.bool_) for p in predictions]
    )
    baseline_score = metric(Y, baseline_output)

    birn = None
    birn_success = None
    birn_failure = None
    if tree_tuple is not None and reliabilities is not None:
        birn_decomp = birnbaum_importance(tree_tuple, reliabilities,
                                          decompose=True)
        birn = birn_decomp['total']
        birn_success = birn_decomp['success']
        birn_failure = birn_decomp['failure']

    return {
        'flip': flip_importance(predictions, compiled_tree),
        'oracle': oracle_importance(predictions, Y, compiled_tree, metric),
        'degradation': degradation_importance(
            predictions, Y, compiled_tree, metric),
        'birnbaum': birn,
        'birnbaum_success': birn_success,
        'birnbaum_failure': birn_failure,
        'model_names': model_names,
        'baseline_score': baseline_score,
    }