"""
Text preprocessing utilities for SentimentScope.
"""
import re


def clean_text(text: str) -> str:
    """
    Clean and normalize an input text string.

    Args:
        text (str): Raw input sentence or paragraph.

    Returns:
        str: Normalized text string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Normalize whitespace (replace multiple spaces, newlines, tabs with a single space)
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing spaces
    return text.strip()
