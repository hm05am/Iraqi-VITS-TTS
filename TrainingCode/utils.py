"""
Utility functions for VITS training and inference.
"""

import os
import glob
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


logging.basicConfig(stream=None, level=logging.INFO)
logger = logging


MATPLOTLIB_FLAG = False


class HParams:
    """
    Hyperparameter container. Loads from a JSON config file.
    Supports attribute-style access (hps.train.batch_size).
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams_from_file(config_path):
    """Load hyperparameters from a JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hparams = HParams(**data)
    return hparams


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
    
    Returns:
        Tuple of (model, optimizer, learning_rate, iteration).
    """
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    iteration = checkpoint_dict.get('iteration', 0)
    learning_rate = checkpoint_dict.get('learning_rate', 0)

    if optimizer is not None and 'optimizer' in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    saved_state_dict = checkpoint_dict['model']
    model_state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k in saved_state_dict and saved_state_dict[k].shape == v.shape:
            new_state_dict[k] = saved_state_dict[k]
        else:
            logger.info(f"Skipping parameter: {k}")
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    """Save a training checkpoint."""
    logger.info(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
        'learning_rate': learning_rate,
    }, checkpoint_path)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """Find the latest checkpoint file in a directory."""
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if len(f_list) == 0:
        return None
    return f_list[-1]


def load_filepaths_and_text(filename, split="|"):
    """
    Load filelists in format: path|text
    
    Returns list of [filepath, text] pairs.
    Handles UTF-8 encoding for Arabic text.
    """
    filepaths_and_text = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(split, maxsplit=1)
            if len(parts) == 2:
                filepaths_and_text.append(parts)
            else:
                logger.info(f"Skipping malformed line: {line[:50]}...")
    return filepaths_and_text


def plot_spectrogram_to_numpy(spectrogram):
    """Convert a spectrogram tensor to a numpy image for TensorBoard."""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend('Agg')
        MATPLOTLIB_FLAG = True
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def summarize(writer, global_step, scalars={}, histograms={}, images={}):
    """Write summaries to TensorBoard."""
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
