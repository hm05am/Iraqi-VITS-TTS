"""
Mel-spectrogram processing for VITS.

Uses torch.stft for GPU-accelerated spectrogram computation.
All log operations use torch.clamp(x, min=1e-5) to prevent NaN.
"""

import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


mel_basis = {}
hann_window = {}

# ================================================================
# Dynamic range compression / decompression
# ================================================================

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """Log-compress magnitudes with a safety clamp to prevent NaN."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """Inverse of dynamic_range_compression_torch."""
    return torch.exp(x) / C


# ================================================================
# Spectrogram functions
# ================================================================

def spectrogram(y, n_fft, hop_size, win_size, center=False):
    """
    Compute linear spectrogram using torch.stft.

    Args:
        y: Audio waveform tensor (batch, time).
        n_fft: FFT size.
        hop_size: Hop length.
        win_size: Window length.
        center: Whether to center-pad the signal.

    Returns:
        Magnitude spectrogram (batch, n_fft//2+1, frames).
    """
    if torch.min(y) < -1.0:
        print('Warning: min value is ', torch.min(y))
    if torch.max(y) > 1.0:
        print('Warning: max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect'
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center, pad_mode='reflect',
        normalized=False, onesided=True,
        return_complex=True
    )
    spec = torch.abs(spec)
    return spec


# Alias for compatibility — some code imports spectrogram_torch
spectrogram_torch = spectrogram


def spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """
    Convert linear spectrogram to mel spectrogram.

    Args:
        spec: Linear spectrogram (batch, n_fft//2+1, frames).
        n_fft: FFT size.
        num_mels: Number of mel bands.
        sampling_rate: Audio sample rate.
        fmin: Minimum mel frequency.
        fmax: Maximum mel frequency.

    Returns:
        Mel spectrogram (batch, num_mels, frames).
    """
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel_fb = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft,
            n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel_fb).to(
            dtype=spec.dtype, device=spec.device
        )
    mel_spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel_spec = spectral_normalize_torch(mel_spec)
    return mel_spec


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size,
                    fmin, fmax, center=False):
    """
    Compute mel spectrogram from audio waveform.
    Combines spectrogram() and spec_to_mel() in one call.
    """
    spec = spectrogram(y, n_fft, hop_size, win_size, center)
    mel = spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    return mel


# Alias for compatibility — some code imports mel_spectrogram_torch
mel_spectrogram_torch = mel_spectrogram


# ================================================================
# Spectral normalization
# ================================================================

def spectral_normalize_torch(magnitudes):
    """Log-scale spectral normalization with safety clamp to prevent NaN."""
    return torch.log(torch.clamp(magnitudes, min=1e-5))


def spectral_de_normalize_torch(magnitudes):
    """Inverse of spectral_normalize_torch."""
    return torch.exp(magnitudes)
