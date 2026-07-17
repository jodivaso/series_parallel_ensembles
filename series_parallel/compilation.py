"""
compilation.py — Compilation of SP trees to postorder instruction lists
                  and their execution for binary classification.

Also contains the legacy BinaryTree-based prediction functions kept for
backward compatibility.

Covers:
  - Opcodes and compile_tree / _compile_node
  - execute_binary (stack-based interpreter)
  - Legacy predict_from_tree (BinaryTree-based, direct inference)
"""

import numpy as np

from .trees import BinaryTree


# =====================================================================
# Opcodes
# =====================================================================

_LOAD     = 0
_PARALLEL = 1
_SERIES   = 2


# =====================================================================
# Compilation: tuple-tree → postorder instruction list
# =====================================================================

def compile_tree(tree):
    """Convert a tuple-tree (or BinaryTree) into flat (opcode, operand) pairs."""
    instructions = []
    _compile_node(tree, instructions)
    return tuple(instructions)


def _compile_node(node, out):
    # Tuple format
    if isinstance(node, int):
        out.append((_LOAD, node))
        return
    if isinstance(node, tuple):
        op, left, right = node
        _compile_node(left, out)
        _compile_node(right, out)
        out.append((_PARALLEL, -1) if op == 'P' else (_SERIES, -1))
        return
    # BinaryTree format (backward compat)
    if node.left is None and node.right is None:
        out.append((_LOAD, node.value))
    else:
        _compile_node(node.left, out)
        _compile_node(node.right, out)
        out.append((_PARALLEL, -1) if node.value == "Paralelo"
                   else (_SERIES, -1))


# =====================================================================
# Compiled execution — binary
# =====================================================================

def execute_binary(instructions, predictions):
    """Evaluate compiled instructions on boolean/int arrays."""
    stack = []
    for op, arg in instructions:
        if op == _LOAD:
            stack.append(predictions[arg])
        elif op == _PARALLEL:
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)
    return stack[0]


# =====================================================================
# Legacy standalone prediction (BinaryTree-based, kept for compat)
# =====================================================================

def predict_from_tree(tree, data, estimators):
    """Recursively evaluate a BinaryTree on data using the given estimators."""
    if tree.right is None and tree.left is None:
        return estimators[tree.value].predict(data)
    elif tree.value == "Paralelo":
        return (predict_from_tree(tree.left, data, estimators)
                | predict_from_tree(tree.right, data, estimators))
    else:
        return (predict_from_tree(tree.left, data, estimators)
                & predict_from_tree(tree.right, data, estimators))


def predict_from_tree_multiclass_probs(tree, data, estimators):
    """Recursively compute class probability vectors for a BinaryTree."""
    if tree.right is None and tree.left is None:
        return estimators[tree.value].predict_proba(data)
    elif tree.value == "Paralelo":
        p_l = predict_from_tree_multiclass_probs(tree.left, data, estimators)
        p_r = predict_from_tree_multiclass_probs(tree.right, data, estimators)
        return 1 - (1 - p_l) * (1 - p_r)
    else:
        p_l = predict_from_tree_multiclass_probs(tree.left, data, estimators)
        p_r = predict_from_tree_multiclass_probs(tree.right, data, estimators)
        return p_l * p_r


def predict_from_tree_multiclass(tree, data, estimators):
    """Return argmax class predictions for a BinaryTree (multiclass)."""
    data = np.array(data)
    probs = predict_from_tree_multiclass_probs(tree, data, estimators)
    return np.argmax(probs, axis=1)
