"""
series_parallel — Series-Parallel Ensemble package.

Usage from a notebook:

    from series_parallel import *

"""

from .trees import (
    BinaryTree,
    enraizar,
    print_tree,
    tuple_to_binarytree,
    binarytree_to_tuple,
    tree_to_string,
    tree_to_string_with_names,
    _generate_random_tree,
)

from .io import (
    decode_tree,
    load_compact,
    load_compact_lazy,
    load_trees,
    _get_or_convert,
    _load_trees_cached,
    _eager_load,
)

from .compilation import (
    _LOAD,
    _PARALLEL,
    _SERIES,
    compile_tree,
    execute_binary,
    predict_from_tree,
    predict_from_tree_multiclass_probs,
    predict_from_tree_multiclass,
)

from .scoring import (
    _MAX_N_FOR_TRUTH_TABLE,
    _vectorized_f1_binary,
    _vectorized_accuracy,
    _build_truth_tables,
    _predictions_to_pattern_indices,
    _score_f1_from_truth_tables,
    _score_accuracy_from_truth_tables,
    _eval_probs_recursive,
    _score_batch_multiclass,
)

from .search_greedy import (
    _greedy_restart_worker_binary,
    _greedy_build_tree_binary,
    _greedy_restart_worker_multiclass,
    _greedy_build_tree_multiclass,
)

from .search_random import (
    _random_search_worker_binary,
    _random_search_binary,
    _random_search_worker_multiclass,
    _random_search_multiclass,
)

from .ensemble import SeriesParallelEnsemble

__all__ = [
    # trees
    "BinaryTree", "enraizar", "print_tree",
    "tuple_to_binarytree", "binarytree_to_tuple",
    "tree_to_string", "tree_to_string_with_names",
    "_generate_random_tree",
    # io
    "decode_tree", "load_compact", "load_compact_lazy",
    "load_trees", "_get_or_convert", "_load_trees_cached", "_eager_load",
    # compilation
    "_LOAD", "_PARALLEL", "_SERIES",
    "compile_tree", "execute_binary",
    "predict_from_tree", "predict_from_tree_multiclass_probs",
    "predict_from_tree_multiclass",
    # scoring
    "_MAX_N_FOR_TRUTH_TABLE",
    "_vectorized_f1_binary", "_vectorized_accuracy",
    "_build_truth_tables", "_predictions_to_pattern_indices",
    "_score_f1_from_truth_tables", "_score_accuracy_from_truth_tables",
    "_eval_probs_recursive", "_score_batch_multiclass",
    # search_greedy
    "_greedy_restart_worker_binary", "_greedy_build_tree_binary",
    "_greedy_restart_worker_multiclass", "_greedy_build_tree_multiclass",
    # search_random
    "_random_search_worker_binary", "_random_search_binary",
    "_random_search_worker_multiclass", "_random_search_multiclass",
    # ensemble
    "SeriesParallelEnsemble",
]
