"""
Text preprocessing module for the RAG pipeline.

This module handles text preprocessing tasks including field combination,
temperature extraction, tokenization, and lemmatization.
"""

import re
import string
from typing import List, Dict, Any

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

import config

# Initialize NLTK components
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    STOPWORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()
SPELL_CHECKER = SpellChecker()
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def extract_preheat_temperature(recipe: Dict[str, Any]) -> str:
    """
    Extract preheat temperature from recipe steps.

    This function searches for patterns like "preheat oven to 425" in the
    recipe steps and extracts the temperature value.

    Args:
        recipe: Dictionary containing recipe data with a 'steps' field.

    Returns:
        String containing formatted temperature information, or empty string if not found.
    """
    steps = recipe.get("steps", "").lower()
    temps = []

    # Pattern: "preheat oven to 425" or variations
    preheat = re.search(
        r"(?:preheat\s+(?:oven|toaster oven)?\s+to\s+)?(\d{2,3})", steps
    )
    if preheat:
        temps.append(f"preheat {preheat.group(1)} degrees")

    return " ".join(temps)


def combine_fields(recipe: Dict[str, Any]) -> str:
    """
    Combine recipe fields into a single searchable text with emphasis.

    This function combines multiple recipe fields and repeats important ones
    (like name) to give them higher weight in TF-IDF vectorization.

    Args:
        recipe: Dictionary containing recipe data.

    Returns:
        Combined string representation of the recipe.
    """
    name = recipe.get("name", "")
    ingredients = recipe.get("ingredients", "")
    description = recipe.get("description", "")
    tags = recipe.get("tags", "")
    steps = recipe.get("steps", "")

    # Extract temperature-related phrases
    temps = extract_preheat_temperature(recipe)

    # Repeat important fields for emphasis in TF-IDF
    # Name is repeated 3x, ingredients 2x, temperature 2x
    if temps:
        return (
            f"{name} {name} {name} "
            f"{ingredients} {ingredients} "
            f"{tags} "
            f"{description} "
            f"TEMPERATURE {temps} TEMPERATURE {temps}"
        )
    else:
        return (
            f"{name} {name} {name} "
            f"{ingredients} {ingredients} "
            f"{tags} "
            f"{description} "
        )


def preprocess(text: str) -> List[str]:
    """
    Preprocess text: lowercase, remove punctuation, tokenize, and lemmatize.

    This function applies the following transformations:
    1. Convert to lowercase
    2. Remove punctuation
    3. Tokenize into words
    4. Remove stopwords
    5. Lemmatize each word

    Args:
        text: Raw text to preprocess.

    Returns:
        List of preprocessed tokens.
    """
    # Lowercase and remove punctuation
    text = text.lower().translate(PUNCT_TABLE)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords and lemmatize
    return [LEMMATIZER.lemmatize(w) for w in words if w not in STOPWORDS]


def simplify_query(query: str) -> str:
    """
    Simplify a query by removing common question phrases.

    Args:
        query: Raw query text.

    Returns:
        Simplified query.
    """
    query = query.lower()

    remove_phrases = [
        "what is",
        "what are",
        "how do i",
        "can i",
        "what temperature",
        "how long",
        "where does",
        "is it",
        "do i",
        "should i",
    ]

    for phrase in remove_phrases:
        query = query.replace(phrase, "")

    return query.strip()


def spell_correct(text: str) -> str:
    """
    Correct spelling errors in text.

    Args:
        text: Text to correct.

    Returns:
        Spell-corrected text.
    """
    corrected = []
    for word in text.split():
        corrected_word = SPELL_CHECKER.correction(word)
        corrected.append(corrected_word if corrected_word else word)
    return " ".join(corrected)


def process_query(
    query: str,
    simplify: bool = None,
    spell_correct_enabled: bool = None,
    filter_custom_stopwords: bool = None,
) -> str:
    """
    Process a query for retrieval.

    Applies optional transformations based on configuration:
    - Simplify: Remove common question phrases
    - Spell correction: Correct spelling errors
    - Custom stopword filtering: Remove domain-specific stopwords

    Args:
        query: Raw query text.
        simplify: Whether to simplify query. If None, uses config.SIMPLIFY_QUERY.
        spell_correct_enabled: Whether to apply spell correction. If None, uses config.SPELL_CORRECT_QUERY.
        filter_custom_stopwords: Whether to filter custom stopwords. If None, uses config.FILTER_CUSTOM_STOPWORDS.

    Returns:
        Processed query string.
    """
    # Use config values if not specified
    simplify = (
        config.SIMPLIFY_QUERY if simplify is None else simplify
    )
    spell_correct_enabled = (
        config.SPELL_CORRECT_QUERY if spell_correct_enabled is None else spell_correct_enabled
    )
    filter_custom_stopwords = (
        config.FILTER_CUSTOM_STOPWORDS if filter_custom_stopwords is None else filter_custom_stopwords
    )

    if simplify:
        query = simplify_query(query)

    if spell_correct_enabled:
        query = spell_correct(query)

    # Standard preprocessing
    tokens = preprocess(query)

    if filter_custom_stopwords:
        tokens = [w for w in tokens if w not in config.CUSTOM_STOPWORDS]

    return " ".join(tokens)


def preprocess_dataset(dataset) -> List[str]:
    """
    Preprocess all recipes in the dataset.

    Args:
        dataset: HuggingFace dataset containing recipes.

    Returns:
        List of preprocessed recipe texts.
    """
    if config.VERBOSE:
        print("Combining recipe fields...")

    combined_texts = [combine_fields(recipe) for recipe in dataset]

    if config.VERBOSE:
        print("Preprocessing texts...")

    preprocessed_texts = [preprocess(doc) for doc in combined_texts]
    joined_texts = [" ".join(tokens) for tokens in preprocessed_texts]

    if config.VERBOSE:
        print(f"✓ Preprocessed {len(joined_texts)} recipes")

    return joined_texts
