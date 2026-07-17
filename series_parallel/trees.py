"""
trees.py — Tree data structures, conversions and string representations.

Covers:
  - BinaryTree class (used only for visualisation)
  - Conversion between tuple-trees and BinaryTree objects
  - String representations (with and without model names)
  - Random SP tree generation
"""

import numpy as np

try:
    from PrettyPrint import PrettyPrintTree
    import colorama
    from colorama import Back
    _HAS_PRETTYPRINT = True
except ImportError:
    _HAS_PRETTYPRINT = False


# =====================================================================
# BinaryTree (only used for visualisation of best_tree)
# =====================================================================

class BinaryTree:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

    def getLeftChild(self):
        return self.left

    def getRightChild(self):
        return self.right

    def setNodeValue(self, value):
        self.value = value

    def getNodeValue(self):
        return self.value

    def isLeave(self):
        return self.left is None and self.right is None


def enraizar(leftTree, rightTree, value):
    tree = BinaryTree(value)
    tree.right = rightTree
    tree.left = leftTree
    return tree


def print_tree(tree):
    if not _HAS_PRETTYPRINT:
        print(tree_to_string(tree))
        return
    pt = PrettyPrintTree(
        lambda x: [] if x is None or x.left is x.right is None else [x.left, x.right],
        lambda x: x.value if x else (colorama.Back.BLACK + '   ' + colorama.Back.LIGHTBLACK_EX),
        color=Back.WHITE)
    pt(tree)


# =====================================================================
# Tuple ↔ BinaryTree conversion
# =====================================================================

def tuple_to_binarytree(t):
    """Convert tuple ('S'/'P', left, right) to BinaryTree for visualisation."""
    if isinstance(t, int):
        return BinaryTree(t)
    op, left, right = t
    node = BinaryTree("Paralelo" if op == 'P' else "Serie")
    node.left = tuple_to_binarytree(left)
    node.right = tuple_to_binarytree(right)
    return node


def binarytree_to_tuple(tree):
    """Convert BinaryTree to tuple representation."""
    if tree.left is None and tree.right is None:
        return tree.value
    op = 'P' if tree.value == "Paralelo" else 'S'
    return (op, binarytree_to_tuple(tree.left), binarytree_to_tuple(tree.right))


# =====================================================================
# String representations (work with both formats)
# =====================================================================

def tree_to_string(tree):
    """String representation — accepts BinaryTree or tuple."""
    if isinstance(tree, int):
        return str(tree)
    if isinstance(tree, tuple):
        op, left, right = tree
        return f"[{op}, {tree_to_string(left)}, {tree_to_string(right)}]"
    # BinaryTree
    if tree is None:
        return "[]"
    elif tree.left is tree.right is None:
        return str(tree.value)
    else:
        return ("[" + str(tree.value) + ", "
                + tree_to_string(tree.left) + ", "
                + tree_to_string(tree.right) + "]")


def tree_to_string_with_names(tree, model_names):
    """String with model names — accepts BinaryTree or tuple."""
    if isinstance(tree, int):
        return model_names[tree]
    if isinstance(tree, tuple):
        op, left, right = tree
        op_name = "Paralelo" if op == 'P' else "Serie"
        return (f"[{op_name}, {tree_to_string_with_names(left, model_names)}, "
                f"{tree_to_string_with_names(right, model_names)}]")
    # BinaryTree
    if tree is None:
        return "[]"
    elif tree.left is tree.right is None:
        return model_names[tree.value]
    else:
        return ("[" + tree.value + ", "
                + tree_to_string_with_names(tree.left, model_names) + ", "
                + tree_to_string_with_names(tree.right, model_names) + "]")


# =====================================================================
# Random SP tree generation
# =====================================================================

def _generate_random_tree(N, rng):
    """Generate a random SP tree with N labeled leaves via random merging.

    Starting from N individual leaves {0, ..., N-1}, at each step two
    components are chosen uniformly at random and merged with a random
    operator (S or P).  This is repeated N-1 times until a single tree
    remains.

    Parameters
    ----------
    N : int
        Number of leaves (base models).
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    tree : tuple
        A random SP tree as a nested tuple.
    """
    pool = list(range(N))

    for _ in range(N - 1):
        idx_a = rng.randint(0, len(pool))
        idx_b = rng.randint(0, len(pool) - 1)
        if idx_b >= idx_a:
            idx_b += 1

        a, b = pool[idx_a], pool[idx_b]
        op = 'S' if rng.randint(0, 2) == 0 else 'P'
        merged = (op, a, b)

        if idx_a > idx_b:
            pool.pop(idx_a)
            pool.pop(idx_b)
        else:
            pool.pop(idx_b)
            pool.pop(idx_a)
        pool.append(merged)

    return pool[0]
