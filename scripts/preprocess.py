import re

#method to normalize character level missmatch such as ጸሀይ and ፀሐይ
def normalize_char_level_missmatch(input_word):
    """
    Normalize character level mismatches in Ethiopian script.

    This method replaces characters that are commonly interchanged or misspelled 
    in Ethiopian scripts with a standard character. For example, it normalizes 
    characters such as ጸሀይ and ፀሐይ to a standard form.

    Args:
        input_word (str): The input word containing potential character mismatches.

    Returns:
        str: The word with normalized characters.
    """

    replacements = [
        ('[ሃኅኃሐሓኻ]', 'ሀ'), ('[ሑኁዅ]', 'ሁ'), ('[ኂሒኺ]', 'ሂ'), ('[ኌሔዄ]', 'ሄ'), ('[ሕኅ]', 'ህ'), ('[ኆሖኾ]', 'ሆ'),
        ('[ሠ]', 'ሰ'), ('[ሡ]', 'ሱ'), ('[ሢ]', 'ሲ'), ('[ሣ]', 'ሳ'), ('[ሤ]', 'ሴ'), ('[ሥ]', 'ስ'), ('[ሦ]', 'ሶ'),
        ('[ዓኣዐ]', 'አ'), ('[ዑ]', 'ኡ'), ('[ዒ]', 'ኢ'), ('[ዔ]', 'ኤ'), ('[ዕ]', 'እ'), ('[ዖ]', 'ኦ'),
        ('[ጸ]', 'ፀ'), ('[ጹ]', 'ፁ'), ('[ጺ]', 'ፂ'), ('[ጻ]', 'ፃ'), ('[ጼ]', 'ፄ'), ('[ጽ]', 'ፅ'), ('[ጾ]', 'ፆ'),
        ('(ሉ[ዋአ])', 'ሏ'), ('(ሙ[ዋአ])', 'ሟ'), ('(ቱ[ዋአ])', 'ቷ'), ('(ሩ[ዋአ])', 'ሯ'), ('(ሱ[ዋአ])', 'ሷ'),
        ('(ሹ[ዋአ])', 'ሿ'), ('(ቁ[ዋአ])', 'ቋ'), ('(ቡ[ዋአ])', 'ቧ'), ('(ቹ[ዋአ])', 'ቿ'), ('(ሁ[ዋአ])', 'ኋ'),
        ('(ኑ[ዋአ])', 'ኗ'), ('(ኙ[ዋአ])', 'ኟ'), ('(ኩ[ዋአ])', 'ኳ'), ('(ዙ[ዋአ])', 'ዟ'), ('(ጉ[ዋአ])', 'ጓ'),
        ('(ደ[ዋአ])', 'ዷ'), ('(ጡ[ዋአ])', 'ጧ'), ('(ጩ[ዋአ])', 'ጯ'), ('(ጹ[ዋአ])', 'ጿ'), ('(ፉ[ዋአ])', 'ፏ'),
        ('[ቊ]', 'ቁ'), ('[ኵ]', 'ኩ')
    ]
    for pattern, replacement in replacements:
        input_word = re.sub(pattern, replacement, input_word)
    return input_word


def clean_document(document):
    """
    Clean a single document by removing links, English words, and unnecessary whitespace.

    Args:
        document (str): The document text to clean.

    Returns:
        str: The cleaned document text.
    """
    if document is None:
        return ""  # Return an empty string if the document is None

    # Remove links
    document = re.sub(r'http\S+|www\S+|https\S+', '', document)
    # Remove English words
    document = re.sub(r'[a-zA-Z]', '', document)
    # Remove extra whitespace and unwanted characters
    document = re.sub(r'\s+', ' ', document).strip()
    # Remove square brackets, parenthese, colons and backlashes.
    document = re.sub(r'[\[\]\(\)]|:|\\', '', document)
    # Remove non-word characters
    document = re.sub(r'[^\w\s]','', document)
    return document

