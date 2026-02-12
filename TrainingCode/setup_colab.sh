#!/bin/bash
# ============================================================
# Iraqi Arabic VITS TTS — Google Colab Environment Setup
# ============================================================
# Run this in a Colab cell:  !bash setup_colab.sh
# ============================================================
set -e

echo "============================================="
echo "  Iraqi Arabic VITS — Environment Setup"
echo "============================================="

# 1. Install Python dependencies
echo "[1/3] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q \
    torch torchaudio \
    scipy==1.11.4 \
    librosa==0.10.1 \
    matplotlib==3.8.2 \
    tensorboard==2.15.1 \
    Cython==3.0.8 \
    numpy==1.24.4 \
    unidecode==1.3.8 \
    soundfile==0.12.1

echo "[1/3] ✅ Dependencies installed."

# 2. Build monotonic_align Cython extension
echo "[2/3] Building monotonic_align Cython extension..."
cd monotonic_align
python setup.py build_ext --inplace
cd ..
echo "[2/3] ✅ monotonic_align built successfully."

# 3. Verify installation
echo "[3/3] Verifying installation..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

from text.symbols import symbols
print(f'  Symbol vocabulary size: {len(symbols)}')

from text.cleaners import iraqi_cleaner
test = iraqi_cleaner('مَرحَبًا بالعالم گلي چيف')
print(f'  Cleaner test: \"{test}\"')

from text import text_to_sequence
seq = text_to_sequence('مرحبا', ['iraqi_cleaner'])
print(f'  text_to_sequence test: {seq}')

try:
    from monotonic_align.core import maximum_path_c
    print('  monotonic_align (Cython): ✅ compiled')
except ImportError:
    print('  monotonic_align (Cython): ⚠️  using numpy fallback')
    from monotonic_align import _maximum_path_numpy
    print('  monotonic_align (numpy fallback): ✅ available')

print()
print('✅ All checks passed! Ready to train.')
"
echo "============================================="
echo "  Setup complete!"
echo "============================================="
