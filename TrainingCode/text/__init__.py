"""
Text processing module for VITS Iraqi Arabic TTS.
Converts text strings to integer sequences using the symbol table.
"""

from text.symbols import symbols, _symbol_to_id, _id_to_symbol
from text import cleaners as cleaners_module


def text_to_sequence(text: str, cleaner_names: list) -> list:
    """
    Convert a string of text to a sequence of IDs.
    
    Args:
        text: Input text string.
        cleaner_names: List of cleaner function names to apply
                       (e.g., ["iraqi_cleaner"]).
    
    Returns:
        List of integer IDs corresponding to symbols.
    """
    for name in cleaner_names:
        cleaner_fn = getattr(cleaners_module, name)
        text = cleaner_fn(text)
    
    sequence = []
    for symbol in text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
    return sequence


def sequence_to_text(sequence: list) -> str:
    """
    Convert a sequence of IDs back to a text string.
    
    Args:
        sequence: List of integer IDs.
    
    Returns:
        Decoded text string.
    """
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result += _id_to_symbol[symbol_id]
    return result


def cleaned_text_to_sequence(cleaned_text: str) -> list:
    """
    Convert already-cleaned text to a sequence of IDs.
    Skips the cleaning step.
    """
    sequence = []
    for symbol in cleaned_text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
    return sequence
