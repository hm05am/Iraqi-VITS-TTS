"""
Text cleaners for Iraqi Arabic + English TTS.

Available cleaners:
  - iraqi_cleaner:      Arabic-only text normalization
  - english_cleaner:    English-only text normalization
  - bilingual_cleaner:  Mixed Arabic + English text normalization

The iraqi_cleaner performs:
  1. Unicode NFC normalization
  2. Diacritic (harakat) removal for character-based approach
  3. Alef normalization (أ إ آ → ا)
  4. Tatweel removal (ـ)
  5. Collapse multiple whitespace
  6. Strip unsupported characters

The english_cleaner performs:
  1. Unicode NFC normalization
  2. Lowercase conversion
  3. Collapse multiple whitespace
  4. Strip unsupported characters

The bilingual_cleaner combines both pipelines:
  1. Unicode NFC normalization
  2. Lowercase English letters
  3. Remove Arabic diacritics (harakat)
  4. Normalize Alef variants → ا
  5. Remove tatweel
  6. Collapse whitespace
  7. Strip unsupported characters
"""

import re
import unicodedata


# Arabic diacritics (harakat) Unicode range
_HARAKAT_RE = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')

# Alef variants
_ALEF_VARIANTS = re.compile(r'[إأآٱ]')

# Multiple whitespace
_WHITESPACE_RE = re.compile(r'\s+')

# Characters we allow through — built lazily from symbols
_ALLOWED_CHARS = None


def _get_allowed_chars():
    """Lazily load the allowed character set from symbols.py."""
    global _ALLOWED_CHARS
    if _ALLOWED_CHARS is None:
        from text.symbols import symbols
        _ALLOWED_CHARS = set(symbols)
    return _ALLOWED_CHARS


def remove_harakat(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel/harakat)."""
    return _HARAKAT_RE.sub('', text)


def normalize_alef(text: str) -> str:
    """Normalize all Alef variants to plain Alef (ا)."""
    return _ALEF_VARIANTS.sub('ا', text)


def remove_tatweel(text: str) -> str:
    """Remove tatweel/kashida (ـ) elongation character."""
    return text.replace('ـ', '')


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return _WHITESPACE_RE.sub(' ', text)


def strip_unsupported(text: str) -> str:
    """Remove any characters not in our symbol set."""
    allowed = _get_allowed_chars()
    return ''.join(c for c in text if c in allowed)


def iraqi_cleaner(text: str) -> str:
    """
    Main cleaner for Iraqi Arabic text.
    
    Pipeline:
      1. Unicode NFC normalization
      2. Remove diacritics (harakat)
      3. Normalize Alef variants → ا
      4. Remove tatweel
      5. Collapse whitespace
      6. Strip unsupported characters
      7. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize('NFC', text)
    text = remove_harakat(text)
    text = normalize_alef(text)
    text = remove_tatweel(text)
    text = collapse_whitespace(text)
    text = strip_unsupported(text)
    text = text.strip()
    return text


def english_cleaner(text: str) -> str:
    """
    Cleaner for English text.

    Pipeline:
      1. Unicode NFC normalization
      2. Lowercase conversion
      3. Collapse whitespace
      4. Strip unsupported characters
      5. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = collapse_whitespace(text)
    text = strip_unsupported(text)
    text = text.strip()
    return text


def bilingual_cleaner(text: str) -> str:
    """
    Cleaner for mixed Arabic + English text.

    Applies both Arabic normalization (harakat removal, alef normalization,
    tatweel removal) and English normalization (lowercase) in a single pass.

    Pipeline:
      1. Unicode NFC normalization
      2. Lowercase English letters
      3. Remove Arabic diacritics (harakat)
      4. Normalize Alef variants → ا
      5. Remove tatweel
      6. Collapse whitespace
      7. Strip unsupported characters
      8. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = remove_harakat(text)
    text = normalize_alef(text)
    text = remove_tatweel(text)
    text = collapse_whitespace(text)
    text = strip_unsupported(text)
    text = text.strip()
    return text
