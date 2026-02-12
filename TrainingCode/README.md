# VITS Iraqi Arabic TTS — Training Framework

A production-ready, single-speaker VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) framework for **Iraqi Arabic (Mesopotamian dialect)**, optimized for **Google Colab (T4 GPU)**.

## Features

- ✅ **Character-based** — no external phonemizer needed
- ✅ **Iraqi Arabic support** — includes گ چ پ ژ ڤ
- ✅ **Single-GPU** — no DDP/distributed errors on Colab
- ✅ **Fixed monotonic_align** — Cython 3.x compatible with numpy fallback
- ✅ **Mixed precision** (FP16) — fits in T4 GPU memory
- ✅ **Alignment monitoring** — clear logging of convergence status

---

## Google Drive Folder Structure

```
MyDrive/
└── TTS/
    └── TrainingCode/
        ├── setup_colab.sh          # Run first!
        ├── train.py                # Single-GPU training script
        ├── inference.py            # Text-to-speech generation
        ├── models.py               # VITS model architecture
        ├── modules.py              # Neural network modules
        ├── attentions.py           # Transformer encoder
        ├── transforms.py           # Normalizing flow transforms
        ├── commons.py              # Utility functions
        ├── losses.py               # Loss functions
        ├── mel_processing.py       # Mel spectrogram computation
        ├── data_utils.py           # Data loading
        ├── utils.py                # Checkpoint & config utilities
        ├── configs/
        │   └── iraqi_base.json     # Training configuration
        ├── text/
        │   ├── __init__.py         # Text-to-sequence conversion
        │   ├── symbols.py          # Iraqi Arabic character set
        │   └── cleaners.py         # Text normalization
        ├── monotonic_align/
        │   ├── __init__.py         # MAS wrapper + numpy fallback
        │   ├── core.pyx            # Cython MAS implementation
        │   └── setup.py            # Patched build script
        ├── filelists/
        │   ├── train.txt           # Training filelist
        │   └── val.txt             # Validation filelist
        └── checkpoints/            # Saved during training
            └── iraqi/
```

---

## Dataset Format

Each line in `filelists/train.txt` and `filelists/val.txt`:

```
/content/drive/MyDrive/TTS/dataset/audio_001.wav|مرحبا شلونك
/content/drive/MyDrive/TTS/dataset/audio_002.wav|اني بخير الحمدلله
/content/drive/MyDrive/TTS/dataset/audio_003.wav|گلي شنو صار
```

**Requirements:**
- WAV files: **22050 Hz**, mono, 16-bit
- Text: UTF-8 encoded Iraqi Arabic
- Separator: pipe character `|`
- Recommended: 3-15 seconds per utterance
- Minimum dataset: ~1 hour for initial results, 5+ hours for good quality

---

## Quick Start (Google Colab)

### Cell 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Navigate to project
```python
import os
os.chdir('/content/drive/MyDrive/TTS/TrainingCode')
```

### Cell 3: Setup environment
```python
!bash setup_colab.sh
```

### Cell 4: Start training
```python
!python train.py \
    --config configs/iraqi_base.json \
    --model_dir checkpoints/iraqi
```

### Cell 5: Monitor with TensorBoard
```python
%load_ext tensorboard
%tensorboard --logdir checkpoints/iraqi/logs
```

### Cell 6: Generate speech (after training)
```python
from inference import infer_text

audio = infer_text(
    text="مرحبا شلونك اليوم",
    model_path="checkpoints/iraqi/G_10000.pth",
    config_path="configs/iraqi_base.json",
    output_path="test_output.wav"
)
```

---

## Training Tips for Convergence

1. **Watch `loss_dur`** — This is your alignment indicator:
   - `loss_dur > 10`: Alignment hasn't started yet (normal for first ~2000 steps)
   - `loss_dur < 5`: Alignment is progressing
   - `loss_dur < 1`: Alignment has converged ✅

2. **Don't reduce batch size below 16** — smaller batches make alignment harder

3. **Audio quality checklist:**
   - All WAVs must be 22050 Hz mono
   - Remove silence at start/end of each file
   - Remove files with background noise
   - Keep utterances between 3-15 seconds

4. **If you get robot noise / silence:**
   - Train longer (at least 20k-50k steps)
   - Check your filelist paths are correct
   - Ensure text matches audio content
   - Try reducing `noise_scale` during inference (e.g., 0.3)

5. **Resume training after Colab timeout:**
   - Checkpoints auto-save every `eval_interval` (1000) steps
   - Training auto-resumes from the latest checkpoint

---

## Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_scale` | 0.667 | Prior noise (lower = more stable but less variation) |
| `noise_scale_w` | 0.8 | Duration noise (lower = more consistent timing) |
| `length_scale` | 1.0 | Speaking speed (>1 = slower, <1 = faster) |

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `monotonic_align` import error | Run `setup_colab.sh` — it builds the Cython extension |
| CUDA out of memory | Reduce `batch_size` to 8 in config |
| Socket/port errors | This is a DDP issue — this codebase doesn't use DDP |
| Robot noise output | Train for more steps, check audio quality |
| Silent output | Check filelist paths, ensure WAVs aren't silent |
