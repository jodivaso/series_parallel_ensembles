"""
series_parallel_random.py — Backward-compatibility shim.

Notebooks that already contain:

    from series_parallel_random import *

continue to work without any changes.
"""

from series_parallel import *  # noqa: F401, F403
