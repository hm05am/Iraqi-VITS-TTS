"""
Common utility functions for VITS.
"""

import math
import torch
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
    """Initialize module weights with normal distribution."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """Calculate padding size for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)


def intersperse(lst, item):
    """Insert item between every element of lst. Used for blank token insertion."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def slice_segments(x, ids_str, segment_size=4):
    """Slice segments from tensor x starting at ids_str positions."""
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """Randomly slice segments from tensor x."""
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str_max = torch.clamp(ids_str_max, min=1)
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max.float()).long()
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def sequence_mask(length, max_length=None):
    """Create a boolean mask from sequence lengths."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    Generate monotonic alignment path from durations.
    
    Args:
        duration: (batch, 1, t_x) — integer durations.
        mask: (batch, 1, t_y, t_x) — attention mask.
    
    Returns:
        path: (batch, 1, t_y, t_x) — one-hot alignment path.
    """
    device = duration.device
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    """Clip gradient values and return the total gradient norm."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def convert_pad_shape(pad_shape):
    """Convert pad shape from [[x1, x2], [y1, y2]] to [y1, y2, x1, x2]."""
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape
