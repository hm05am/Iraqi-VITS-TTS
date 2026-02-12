"""
VITS Inference — Standalone text-to-speech generation.

Usage:
    from inference import infer_text
    audio = infer_text("مرحبا بالعالم", "checkpoints/G_10000.pth", "configs/iraqi_base.json")
    
Or from command line:
    python inference.py --text "مرحبا" --model checkpoints/G_10000.pth --config configs/iraqi_base.json
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
import scipy.io.wavfile

from text import text_to_sequence
from text.cleaners import iraqi_cleaner
from text.symbols import symbols
from models import SynthesizerTrn
from utils import get_hparams_from_file
import commons


def infer_text(text, model_path, config_path, output_path="output.wav",
               noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
    """
    Generate speech from text using a trained VITS model.
    
    Args:
        text: Input text in Arabic/Iraqi Arabic.
        model_path: Path to generator checkpoint (.pth).
        config_path: Path to config JSON.
        output_path: Path to save output WAV file.
        noise_scale: Controls voice variation (lower = more stable).
        length_scale: Controls speaking speed (>1 = slower, <1 = faster).
        noise_scale_w: Controls duration variation.
    
    Returns:
        audio: Generated audio as numpy array.
    """
    # Load config
    hps = get_hparams_from_file(config_path)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    n_vocab = len(symbols)
    spec_channels = hps.data.filter_length // 2 + 1
    segment_size = hps.train.segment_size // hps.data.hop_length

    net_g = SynthesizerTrn(
        n_vocab,
        spec_channels,
        segment_size,
        **vars(hps.model)
    ).to(device)
    net_g.eval()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        net_g.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint: {model_path} (step {checkpoint.get('iteration', '?')})")
    else:
        net_g.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {model_path}")

    # Clean text
    cleaned_text = iraqi_cleaner(text)
    print(f"Input text:   {text}")
    print(f"Cleaned text: {cleaned_text}")

    # Convert to sequence
    text_norm = text_to_sequence(cleaned_text, hps.data.text_cleaners)
    if getattr(hps.data, 'add_blank', True):
        text_norm = commons.intersperse(text_norm, 0)
    
    text_tensor = torch.LongTensor(text_norm).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([len(text_norm)]).to(device)

    print(f"Sequence length: {len(text_norm)}")

    # Generate
    with torch.no_grad():
        audio_tensor, attn, y_mask, _ = net_g.infer(
            text_tensor, text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w
        )
    
    # Convert to numpy
    audio = audio_tensor[0, 0].cpu().numpy()
    audio = audio / max(abs(audio.max()), abs(audio.min()) + 1e-8) * 0.95  # Normalize

    # Save WAV
    sampling_rate = hps.data.sampling_rate
    scipy.io.wavfile.write(
        output_path, sampling_rate,
        (audio * 32767).astype(np.int16)
    )
    print(f"Saved audio to: {output_path}")
    print(f"Audio duration: {len(audio) / sampling_rate:.2f}s")

    # Try IPython playback (works in Colab/Jupyter)
    try:
        from IPython.display import Audio, display
        display(Audio(audio, rate=sampling_rate))
        print("🔊 Audio player displayed above (Colab/Jupyter)")
    except ImportError:
        pass

    return audio


def main():
    parser = argparse.ArgumentParser(description='VITS Inference')
    parser.add_argument('--text', '-t', required=True, help='Input text (Arabic)')
    parser.add_argument('--model', '-m', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', '-c', required=True, help='Path to config JSON')
    parser.add_argument('--output', '-o', default='output.wav', help='Output WAV path')
    parser.add_argument('--noise_scale', type=float, default=0.667)
    parser.add_argument('--noise_scale_w', type=float, default=0.8)
    parser.add_argument('--length_scale', type=float, default=1.0)
    args = parser.parse_args()

    infer_text(
        text=args.text,
        model_path=args.model,
        config_path=args.config,
        output_path=args.output,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
    )


if __name__ == '__main__':
    main()
