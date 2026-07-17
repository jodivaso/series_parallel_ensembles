"""
ensemble.py — SeriesParallelEnsemble: the main public class.

Wraps all sub-modules and exposes a scikit-learn compatible interface:
fit() / predict() / predict_multiclass().

Strategies
----------
exhaustive : evaluate every SP tree of N models (default).
greedy     : beam search that merges the best pair at each step.
random     : time-limited random sampling of SP trees.
"""

import os

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

from .trees import (
    BinaryTree,
    tuple_to_binarytree,
    tree_to_string_with_names,
    print_tree,
)
from .io import BinaryTree, _get_or_convert, _load_trees_cached
from .compilation import compile_tree, execute_binary
from .scoring import (
    _MAX_N_FOR_TRUTH_TABLE,
    _build_truth_tables,
    _predictions_to_pattern_indices,
    _score_f1_from_truth_tables,
    _score_accuracy_from_truth_tables,
    _vectorized_f1_binary,
    _vectorized_accuracy,
    _eval_probs_recursive,
    _score_batch_multiclass,
)
from .search_greedy import _greedy_build_tree_binary, _greedy_build_tree_multiclass
from .search_random import _random_search_binary, _random_search_multiclass


class SeriesParallelEnsemble(BaseEstimator):
    """Series-Parallel ensemble with compact tuple-based trees.

    Binary:     truth-table vectorisation (default) or compiled stack
                evaluation, both with vectorised scoring.
    Multiclass: recursive tuple evaluation + joblib parallelism.

    Parameters
    ----------
    estimators : list of (name, estimator) pairs
        Base learners.
    mode : str or None
        'multiclass' for multiclass problems; None / anything else for binary.
    metric : callable, default accuracy_score
        Scoring function used during tree selection.
    strategy : {'exhaustive', 'greedy', 'random'}, default 'exhaustive'
        Tree-search strategy.
    vectorized : bool, default True
        If True, binary exhaustive fit uses truth-table lookup (~100x faster
        for typical N).  Falls back to compiled evaluation when N > 20.
    trees : dict, str, or None
        Pre-loaded tree dictionary {N: [tuple_tree, ...]}, a path to a
        .gz or .pickle file, or None (auto-detect from current directory).
        Only used by the exhaustive strategy.
    random_state : int, default 123
        Base random seed.
    """

    def __init__(self, estimators, mode=None, train_ensemble_X=None,
                 metric=accuracy_score, bayesian_optimization=None,
                 trees=None, random_state=123, strategy='exhaustive',
                 vectorized=True):
        self.estimators           = estimators
        self.mode                 = mode
        self.train_ensemble_X     = train_ensemble_X
        self.bayesian_optimization = bayesian_optimization
        self.metric               = metric
        self.random_state         = random_state
        self.strategy             = strategy
        self.vectorized           = vectorized

        if strategy not in ('exhaustive', 'greedy', 'random'):
            raise ValueError(
                f"strategy must be 'exhaustive', 'greedy', or 'random'. "
                f"Got: '{strategy}'")

        self.best_tree        = None   # BinaryTree (for visualisation)
        self._best_tree_tuple = None   # tuple      (for fast inference)
        self._compiled_best   = None   # compiled instructions (binary)

        # Load tree dictionary — only needed for exhaustive strategy.
        if strategy == 'exhaustive':
            if isinstance(trees, dict):
                sample = next(iter(trees.values()))[0]
                if isinstance(sample, BinaryTree):
                    self.dict_of_trees = _get_or_convert(trees)
                else:
                    self.dict_of_trees = trees
            elif isinstance(trees, str):
                self.dict_of_trees = _load_trees_cached(trees)
            elif trees is None:
                self.dict_of_trees = _load_trees_cached(None)
            else:
                raise TypeError(
                    f"'trees' must be a dict, a str path, or None. "
                    f"Got: {type(trees).__name__}")
        else:
            self.dict_of_trees = None

    # -----------------------------------------------------------------
    # Training helpers
    # -----------------------------------------------------------------

    def fit_estimators(self, data_X, data_Y):
        for name, model in self.estimators:
            model.fit(data_X, data_Y)

    def fit_optimize_estimators(self, data_X, data_Y, ensemble_X, ensemble_Y):
        results = []
        for name, model in self.estimators:
            optimized_model = self.bayesian_optimization(
                model, self.random_state, data_X, data_Y,
                ensemble_X, ensemble_Y, self.metric, verbose=False)
            results.append((name, optimized_model))
        self.estimators = results

    # -----------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------

    def print_best_tree(self):
        print_tree(self.best_tree)

    def best_tree_as_string(self):
        model_names = [str(i).split('(')[0] for i, x in self.estimators]
        return tree_to_string_with_names(self.best_tree, model_names)

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def predict(self, X):
        """Binary prediction using the best tree (compiled)."""
        predictions = [model.predict(X) for _, model in self.estimators]
        return execute_binary(self._compiled_best, predictions)

    def predict_multiclass(self, X):
        """Multiclass prediction using the best tree (recursive on tuples)."""
        preds_proba = np.array(
            [model.predict_proba(X) for _, model in self.estimators])
        probs = _eval_probs_recursive(self._best_tree_tuple, preds_proba)
        return np.argmax(probs, axis=1)

    # -----------------------------------------------------------------
    # fit — dispatcher
    # -----------------------------------------------------------------

    def fit(self, X, Y, include_permutations=True, random_state=123,
            n_splits=1, metric=None, n_jobs=-1, max_memory_mb=2048,
            beam_width=10, n_restarts=1, allow_partial=True,
            time_limit_seconds=60.0):
        """Fit the ensemble by selecting or constructing the best SP tree.

        Parameters
        ----------
        beam_width : int
            (greedy only) Candidates kept per merge step. 1 = pure greedy.
        n_restarts : int
            (greedy only) Independent runs with randomised tie-breaking.
        allow_partial : bool
            (greedy only) If True, the returned tree may use fewer than
            N models (the global best across all merge steps).  If False,
            only the best fully-merged tree (using all N models) is
            returned.  All N-1 merge steps are always executed regardless.
        n_jobs : int
            Parallel workers for exhaustive multiclass, random search, and
            greedy restarts (-1 = all cores, 1 = sequential).
        time_limit_seconds : float
            (random only) Wall-clock budget in seconds per worker.
        """
        if metric is None:
            metric = self.metric

        if self.strategy == 'greedy':
            if self.mode == 'multiclass':
                self._fit_multiclass_greedy(X, Y, metric, beam_width,
                                            n_restarts, random_state,
                                            n_jobs, allow_partial)
            else:
                self._fit_binary_greedy(X, Y, metric, beam_width,
                                        n_restarts, random_state,
                                        allow_partial, n_jobs)

        elif self.strategy == 'random':
            if self.mode == 'multiclass':
                self._fit_multiclass_random(X, Y, metric,
                                            time_limit_seconds,
                                            n_jobs, random_state)
            else:
                self._fit_binary_random(X, Y, metric,
                                        time_limit_seconds,
                                        n_jobs, random_state)

        else:  # exhaustive
            N     = len(self.estimators)
            trees = self.dict_of_trees[N]
            if self.mode == 'multiclass':
                self._fit_multiclass(X, Y, trees, metric, n_jobs)
            else:
                use_vectorized = (
                    self.vectorized and N <= _MAX_N_FOR_TRUTH_TABLE)
                self._fit_binary(X, Y, trees, metric,
                                 n_splits, random_state, max_memory_mb,
                                 use_vectorized)
        return self

    # -----------------------------------------------------------------
    # Internal helpers: set best tree
    # -----------------------------------------------------------------

    def _set_best(self, trees, best_idx):
        self._set_best_from_tuple(trees[best_idx])

    def _set_best_from_tuple(self, tree_tuple):
        self._best_tree_tuple = tree_tuple
        self.best_tree        = tuple_to_binarytree(self._best_tree_tuple)
        self._compiled_best   = compile_tree(self._best_tree_tuple)

    # -----------------------------------------------------------------
    # Binary exhaustive fit
    # -----------------------------------------------------------------

    def _fit_binary(self, X, Y, trees, metric,
                    n_splits, random_state, max_memory_mb, use_vectorized):
        n_trees = len(trees)
        N       = len(self.estimators)
        use_vec_f1  = (metric is f1_score)
        use_vec_acc = (metric is accuracy_score)

        if use_vectorized:
            truth_tables = _build_truth_tables(trees, N)
            compiled     = None
        else:
            truth_tables = None
            compiled     = [compile_tree(t) for t in trees]

        if n_splits <= 1:
            predictions = [est.predict(X) for _, est in self.estimators]
            scores = self._score_binary(
                predictions, Y, N, compiled, truth_tables, metric,
                use_vec_f1, use_vec_acc, max_memory_mb, use_vectorized)
            self._set_best(trees, int(np.argmax(scores)))
        else:
            kf        = KFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
            rank_sums = np.zeros(n_trees, dtype=np.float64)

            for _, test_idx in kf.split(X):
                X_test, Y_test = X[test_idx], Y[test_idx]
                predictions = [est.predict(X_test) for _, est in self.estimators]
                scores = self._score_binary(
                    predictions, Y_test, N, compiled, truth_tables, metric,
                    use_vec_f1, use_vec_acc, max_memory_mb, use_vectorized)
                rank_sums += rankdata(-scores, method='min')

            self._set_best(trees, int(np.argmin(rank_sums)))

    @staticmethod
    def _score_binary(predictions, Y, N, compiled, truth_tables, metric,
                      use_vec_f1, use_vec_acc, max_memory_mb, use_vectorized):
        """Dispatch to vectorized or compiled scoring path."""
        if use_vectorized:
            return SeriesParallelEnsemble._score_binary_vectorized(
                predictions, Y, N, truth_tables, metric,
                use_vec_f1, use_vec_acc, max_memory_mb)
        else:
            return SeriesParallelEnsemble._score_binary_compiled(
                predictions, Y, compiled, metric,
                use_vec_f1, use_vec_acc, max_memory_mb)

    @staticmethod
    def _score_binary_vectorized(predictions, Y, N, truth_tables, metric,
                                  use_vec_f1, use_vec_acc, max_memory_mb):
        Y_bool          = np.asarray(Y, dtype=np.bool_)
        pattern_indices = _predictions_to_pattern_indices(predictions, N)

        if use_vec_f1:
            return _score_f1_from_truth_tables(
                truth_tables, pattern_indices, Y_bool)
        elif use_vec_acc:
            return _score_accuracy_from_truth_tables(
                truth_tables, pattern_indices, Y_bool)

        # Fallback for custom metrics: materialise predictions
        n_trees   = truth_tables.shape[0]
        n_samples = len(Y)
        batch_size = max(1, (max_memory_mb * 1024 * 1024) // max(n_samples, 1))
        scores    = np.empty(n_trees, dtype=np.float64)

        for start in range(0, n_trees, batch_size):
            end        = min(start + batch_size, n_trees)
            batch_preds = truth_tables[start:end][:, pattern_indices]
            for i in range(end - start):
                scores[start + i] = metric(Y, batch_preds[i])
        return scores

    @staticmethod
    def _score_binary_compiled(predictions, Y, compiled, metric,
                                use_vec_f1, use_vec_acc, max_memory_mb):
        n_trees   = len(compiled)
        n_samples = len(Y)
        batch_size = max(1, (max_memory_mb * 1024 * 1024) // max(n_samples, 1))

        scores = np.empty(n_trees, dtype=np.float64)
        Y_bool = np.asarray(Y, dtype=np.bool_)

        for start in range(0, n_trees, batch_size):
            end       = min(start + batch_size, n_trees)
            batch_len = end - start

            batch_preds = np.empty((batch_len, n_samples), dtype=np.bool_)
            for i, instructions in enumerate(compiled[start:end]):
                batch_preds[i] = execute_binary(instructions, predictions)

            if use_vec_f1:
                scores[start:end] = _vectorized_f1_binary(batch_preds, Y_bool)
            elif use_vec_acc:
                scores[start:end] = _vectorized_accuracy(batch_preds, Y_bool)
            else:
                for i in range(batch_len):
                    scores[start + i] = metric(Y, batch_preds[i])

        return scores

    # -----------------------------------------------------------------
    # Multiclass exhaustive fit
    # -----------------------------------------------------------------

    def _fit_multiclass(self, X, Y, trees, metric, n_jobs):
        preds_proba = np.array(
            [est.predict_proba(X) for _, est in self.estimators])
        n_trees = len(trees)

        if n_jobs != 1 and n_trees > 50:
            n_workers  = os.cpu_count() if n_jobs == -1 else n_jobs
            chunk_size = max(1, n_trees // (n_workers * 4))
            batches    = [trees[i:i + chunk_size]
                          for i in range(0, n_trees, chunk_size)]
            results    = Parallel(n_jobs=n_jobs)(
                delayed(_score_batch_multiclass)(batch, preds_proba, Y, metric)
                for batch in batches)
            scores = np.array(
                [s for batch_scores in results for s in batch_scores])
        else:
            scores = np.empty(n_trees, dtype=np.float64)
            for idx, tree in enumerate(trees):
                probs        = _eval_probs_recursive(tree, preds_proba)
                scores[idx]  = metric(Y, np.argmax(probs, axis=1))

        self._set_best(trees, int(np.argmax(scores)))

    # -----------------------------------------------------------------
    # Greedy fits
    # -----------------------------------------------------------------

    def _fit_multiclass_greedy(self, X, Y, metric, beam_width,
                                n_restarts, random_state, n_jobs,
                                allow_partial):
        preds_proba = np.array(
            [est.predict_proba(X) for _, est in self.estimators])
        best_tree, _ = _greedy_build_tree_multiclass(
            preds_proba, Y, metric,
            beam_width=beam_width, n_restarts=n_restarts,
            random_state=random_state, n_jobs=n_jobs,
            allow_partial=allow_partial)
        self._set_best_from_tuple(best_tree)

    def _fit_binary_greedy(self, X, Y, metric, beam_width,
                            n_restarts, random_state, allow_partial,
                            n_jobs):
        predictions = [est.predict(X) for _, est in self.estimators]
        best_tree, _ = _greedy_build_tree_binary(
            predictions, Y, metric,
            beam_width=beam_width, n_restarts=n_restarts,
            random_state=random_state, allow_partial=allow_partial,
            n_jobs=n_jobs)
        self._set_best_from_tuple(best_tree)

    # -----------------------------------------------------------------
    # Random fits
    # -----------------------------------------------------------------

    def _fit_binary_random(self, X, Y, metric, time_limit_seconds,
                            n_jobs, random_state):
        predictions = [est.predict(X) for _, est in self.estimators]
        best_tree, _, n_evaluated = _random_search_binary(
            predictions, Y, metric,
            time_limit_seconds=time_limit_seconds,
            n_jobs=n_jobs, random_state=random_state)
        self._set_best_from_tuple(best_tree)
        self._random_search_n_evaluated = n_evaluated

    def _fit_multiclass_random(self, X, Y, metric, time_limit_seconds,
                                n_jobs, random_state):
        preds_proba = np.array(
            [est.predict_proba(X) for _, est in self.estimators])
        best_tree, _, n_evaluated = _random_search_multiclass(
            preds_proba, Y, metric,
            time_limit_seconds=time_limit_seconds,
            n_jobs=n_jobs, random_state=random_state)
        self._set_best_from_tuple(best_tree)
        self._random_search_n_evaluated = n_evaluated