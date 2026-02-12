"""
Bilingual (Arabic + English) symbol set for character-based VITS TTS.
Includes standard Arabic, Iraqi-specific characters (گ چ پ ژ ڤ),
English lowercase alphabet, punctuation, and numerals.
"""

_pad = '_'
_punctuation = '؟،.!؛,?; '
_english_punctuation = ["'", '"', '-', ':', '(', ')']  # English-specific punctuation
_special = ['ـ']  # Tatweel (kashida)

# Standard Arabic letters
_arabic_letters = [
    'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ',
    'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ',
    'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض',
    'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل',
    'م', 'ن', 'ه', 'و', 'ي', 'ى',
]

# Iraqi dialect specific characters
_iraqi_letters = ['گ', 'چ', 'پ', 'ژ', 'ڤ']

# Arabic diacritics (harakat) — kept in symbol set for optional use
_harakat = ['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ']

# Eastern Arabic numerals
_numerals = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']

# English lowercase letters
_english_letters = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]

# Western (ASCII) numerals
_western_numerals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Build the complete symbol list
# Index 0 = _pad (blank token for CTC alignment)
# NOTE: Existing symbol indices are preserved — new groups are appended at the end.
symbols = (
    [_pad]
    + list(_punctuation)
    + _special
    + _arabic_letters
    + _iraqi_letters
    + _harakat
    + _numerals
    + _english_letters
    + _western_numerals
    + _english_punctuation
)

# Lookup tables
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
