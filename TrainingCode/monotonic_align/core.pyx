# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Monotonic Alignment Search (MAS) — Cython implementation.
Compatible with Cython 3.x and Python 3.10+.
"""

import cython
import numpy as np
cimport numpy as np
from cython cimport floating


ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _maximum_path_c(DTYPE_t[:, ::1] path,
                          const DTYPE_t[:, ::1] value,
                          int t_x, int t_y) noexcept nogil:
    """Find the maximum-sum monotonic path through the 2D value matrix."""
    cdef int x, y
    cdef DTYPE_t v_prev, v_cur
    cdef int index = t_x - 1

    # Forward pass — compute cumulative sums along valid monotonic paths
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = 0.0  # Force diagonal at start
            elif x == 0:
                v_cur = 0.0  # leftmost column
            else:
                v_cur = value[x, y - 1]  # from left
            if y == 0:
                v_prev = 0.0
            else:
                v_prev = value[x - 1, y - 1] if x > 0 else -1e9
            # Take the better of: stay at same x, or advance x
            # We abuse value in-place for the DP table
            pass  # DP is done via the backtracking approach below

    # Reset — use a simpler backtracking MAS
    # Build the opt table
    cdef DTYPE_t[:, ::1] opt = np.zeros((t_x, t_y), dtype=np.float32)
    
    # Initialize
    opt[0, 0] = value[0, 0]
    for y in range(1, t_y):
        if t_x + y - t_y <= 0:
            opt[0, y] = opt[0, y - 1] + value[0, y]
    for x in range(1, t_x):
        opt[x, x] = opt[x - 1, x - 1] + value[x, x]
    
    for y in range(1, t_y):
        for x in range(max(1, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                opt[x, y] = opt[x - 1, y - 1] + value[x, y]
            else:
                v_prev = opt[x - 1, y - 1] if x > 0 else -1e9
                v_cur = opt[x, y - 1]
                if v_prev >= v_cur:
                    opt[x, y] = v_prev + value[x, y]
                else:
                    opt[x, y] = v_cur + value[x, y]

    # Backtrack
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1.0
        if index > 0 and y > 0:
            if opt[index - 1, y - 1] >= opt[index, y - 1]:
                index -= 1


def maximum_path_c(DTYPE_t[:, ::1] path,
                   DTYPE_t[:, ::1] value,
                   int t_x, int t_y):
    """Python-callable wrapper for the Cython MAS function."""
    with nogil:
        _maximum_path_c(path, value, t_x, t_y)
