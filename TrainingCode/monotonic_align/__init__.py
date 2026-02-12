"""
Monotonic Alignment Search module.

Provides maximum_path() for finding the optimal monotonic alignment
between text and mel-spectrogram frames during VITS training.

Build the Cython extension first:
    cd monotonic_align && python setup.py build_ext --inplace
"""

import numpy as np
import torch
from torch.nn import functional as F


def maximum_path(neg_cent, mask):
    """
    Find the maximum-sum monotonic alignment path.
    
    Uses compiled Cython core if available, falls back to
    a pure-numpy implementation otherwise.
    
    Args:
        neg_cent: (batch, t_x, t_y) — negative log-centroids (attention scores).
        mask: (batch, t_x, t_y) — binary mask.
    
    Returns:
        path: (batch, t_x, t_y) — hard monotonic alignment path (0/1 values).
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.float32)
    mask = mask.data.cpu().numpy().astype(np.float32)

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)

    try:
        from monotonic_align.core import maximum_path_c
        for i in range(neg_cent.shape[0]):
            maximum_path_c(path[i], neg_cent[i], t_x_max[i], t_y_max[i])
    except ImportError:
        # Pure numpy fallback (slower but works without compilation)
        for i in range(neg_cent.shape[0]):
            _maximum_path_numpy(path[i], neg_cent[i],
                                int(t_x_max[i]), int(t_y_max[i]))

    return torch.from_numpy(path).to(device=device, dtype=dtype)


def _maximum_path_numpy(path, value, t_x, t_y):
    """
    Pure-numpy fallback for MAS.
    Slower than Cython but works everywhere.
    """
    # Build DP table
    opt = np.full((t_x, t_y), -1e9, dtype=np.float32)
    opt[0, 0] = value[0, 0]
    
    for y in range(1, t_y):
        if t_x + y - t_y <= 0:
            opt[0, y] = opt[0, y - 1] + value[0, y]
    
    for y in range(1, t_y):
        for x in range(max(1, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                opt[x, y] = opt[x - 1, y - 1] + value[x, y]
            else:
                v_prev = opt[x - 1, y - 1] if x > 0 else -1e9
                v_cur = opt[x, y - 1]
                opt[x, y] = max(v_prev, v_cur) + value[x, y]
    
    # Backtrack
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1.0
        if index > 0 and y > 0:
            if opt[index - 1, y - 1] >= opt[index, y - 1]:
                index -= 1
