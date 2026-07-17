"""
io.py — Loading and persistence of SP trees.

Covers:
  - Decoding and loading trees from compact gzip format
  - Lazy streaming loader for large N
  - Legacy pickle support
  - Module-level caches to avoid repeated loading across instantiations
"""

import gzip
import os
import pickle
import struct

from .trees import BinaryTree, binarytree_to_tuple


# =====================================================================
# Compact gzip format
# =====================================================================

def decode_tree(data, offset=0):
    """Decode bytes back into a tuple-tree. Returns (tree, next_offset)."""
    b = data[offset]
    if b & 0x80:  # leaf
        return (b & 0x7F), offset + 1
    else:
        op = 'P' if b == 0x01 else 'S'
        left, off = decode_tree(data, offset + 1)
        right, off = decode_tree(data, off)
        return (op, left, right), off


def load_compact(path):
    """Load compact gzip file → {N: [tuple_tree, ...]}.

    Warning: for large N (>=9) this loads all trees into memory.
    Use load_compact_lazy() for streaming access.
    """
    result = {}
    with gzip.open(path, 'rb') as f:
        while True:
            header = f.read(9)  # 4 + 4 + 1
            if len(header) < 9:
                break
            N, count, tree_size = struct.unpack('<IIB', header)
            trees = []
            for _ in range(count):
                data = f.read(tree_size)
                tree, _ = decode_tree(data)
                trees.append(tree)
            result[N] = trees
            print(f"  Loaded N={N}: {len(trees)} trees")
    return result


def load_compact_lazy(path, target_N):
    """Generator that yields tuple-trees for a specific N without
    loading everything into memory. Skips other N values."""
    with gzip.open(path, 'rb') as f:
        while True:
            header = f.read(9)
            if len(header) < 9:
                return
            N, count, tree_size = struct.unpack('<IIB', header)
            if N == target_N:
                for _ in range(count):
                    data = f.read(tree_size)
                    tree, _ = decode_tree(data)
                    yield tree
                return
            else:
                # Skip this N entirely
                f.seek(count * tree_size, 1)


# =====================================================================
# Main loader (auto-detects format)
# =====================================================================

def load_trees(path=None):
    """Load trees from compact gzip or legacy pickle.

    Returns {N: [tuple_tree, ...]}
    """
    if path is not None:
        if path.endswith('.gz'):
            return load_compact(path)
        else:
            # Legacy pickle with BinaryTree objects — convert to tuples
            with open(path, 'rb') as f:
                bt_dict = pickle.load(f)
            return {n: [binarytree_to_tuple(t) for t in trees]
                    for n, trees in bt_dict.items()}

    # Auto-detect
    if os.path.exists('possible_trees_compact.gz'):
        return load_compact('possible_trees_compact.gz')
    elif os.path.exists('possible_trees.pickle'):
        with open('possible_trees.pickle', 'rb') as f:
            bt_dict = pickle.load(f)
        return {n: [binarytree_to_tuple(t) for t in trees]
                for n, trees in bt_dict.items()}
    else:
        raise FileNotFoundError(
            "No tree file found. Run generate_trees.py first, or provide "
            "a path via dict_of_trees or the path parameter.")


# =====================================================================
# Caches: avoid repeated loading / conversion across instantiations
# =====================================================================

_bt_to_tuple_cache = {}   # id(dict) → converted dict
_load_trees_cache = {}     # resolved path → tuple dict


def _get_or_convert(bt_dict):
    """Convert a {N: [BinaryTree, ...]} dict to {N: [tuple, ...]} once,
    caching the result by object id so repeated calls are free."""
    key = id(bt_dict)
    if key not in _bt_to_tuple_cache:
        _bt_to_tuple_cache[key] = {
            n: [binarytree_to_tuple(t) for t in trees]
            for n, trees in bt_dict.items()
        }
    return _bt_to_tuple_cache[key]


def _load_trees_cached(path=None):
    """Load trees from file, caching by resolved path so repeated calls
    (e.g. from multiple SeriesParallelEnsemble instantiations) are free."""
    key = os.path.abspath(path) if path else "__auto__"
    if key not in _load_trees_cache:
        _load_trees_cache[key] = load_trees(path)
    return _load_trees_cache[key]


def _eager_load():
    """Pre-load trees into cache at import time (optional, disabled by default).

    Skips execution when called from a joblib worker process.
    """
    if os.environ.get("LOKY_MAX_DEPTH"):
        return  # We are inside a worker — do not load

    path = None
    if os.path.exists('possible_trees_compact.gz'):
        path = os.path.abspath('possible_trees_compact.gz')
    elif os.path.exists('possible_trees.pickle'):
        path = os.path.abspath('possible_trees.pickle')
    else:
        return

    if "__auto__" not in _load_trees_cache:
        print(f"[series_parallel] Loading trees from {path} ...")
        trees = load_trees(path)
        _load_trees_cache["__auto__"] = trees
        _load_trees_cache[path] = trees
        print("[series_parallel] Trees ready in cache.")

# Uncomment the line below to eagerly load trees at import time:
# _eager_load()
