"""
Cython implementation of Monotonic Alignment Search (MAS) for VITS.

Compatible with NumPy 2.x — uses typed memoryviews only, no ndarray.dimensions.

Build:
    cd monotonic_align && python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _maximum_path_c(float[:, ::1] path, float[:, ::1] value, int t_x, int t_y) noexcept nogil:
    """
    Viterbi-style DP to find the maximum-sum monotonic alignment path.

    Args:
        path:  (t_x, t_y) output — will be filled with 0/1 hard alignment.
        value: (t_x, t_y) input  — attention/score matrix (modified in-place as DP table).
        t_x:   number of text frames   (rows).
        t_y:   number of mel frames     (columns).
    """
    cdef int x, y, index
    cdef float v_prev, v_cur

    # ── Forward pass: accumulate DP scores in-place ──
    for y in range(t_y):
        for x in range(t_x):
            if x == y:
                # On the diagonal: can only come from (x-1, y-1)
                if x > 0 and y > 0:
                    value[x, y] = value[x, y] + value[x - 1, y - 1]
                # else: value[0,0] stays as is
            elif y > x:
                # Below diagonal: can come from (x, y-1) or (x-1, y-1)
                if x == 0:
                    value[x, y] = value[x, y] + value[x, y - 1]
                else:
                    v_prev = value[x - 1, y - 1]
                    v_cur  = value[x, y - 1]
                    if v_prev > v_cur:
                        value[x, y] = value[x, y] + v_prev
                    else:
                        value[x, y] = value[x, y] + v_cur
            else:
                # x > y: unreachable region (more text than mel so far)
                value[x, y] = -INFINITY

    # ── Backtrack: extract the hard alignment path ──
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1.0
        # Decide whether to step diagonally (index-1) or stay (index)
        if index > 0 and y > 0:
            if value[index - 1, y - 1] >= value[index, y - 1]:
                index = index - 1


def maximum_path_c(float[:, ::1] path, float[:, ::1] value, int t_x, int t_y):
    """
    Python-callable wrapper for the nogil MAS implementation.

    Args:
        path:  (t_x, t_y) float32 zeroed array — output alignment.
        value: (t_x, t_y) float32 score matrix — MODIFIED in-place.
        t_x:   valid text length.
        t_y:   valid mel length.
    """
    _maximum_path_c(path, value, t_x, t_y)
