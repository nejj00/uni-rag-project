"""
Configuration module for the RAG pipeline.

This module contains all constants and configuration settings used throughout
the RAG pipeline for recipe reasoning.
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure output directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset and query files
DATASET_FILE = INPUTS_DIR / "irse_documents_2026_recipes.parquet"
QUERIES_FILE = INPUTS_DIR / "irse_queries_2026_recipes.json"

# Output files
VERSION = "FINAL"
RETRIEVED_DOCS_OUTPUT = RESULTS_DIR / f"retrieved_docs_{VERSION}.json"
GENERATED_ANSWERS_OUTPUT = RESULTS_DIR / f"generated_answers_{VERSION}.json"

# Retrieval configuration
RETRIEVAL_K = 10  # Number of top documents to retrieve
RETRIEVAL_DROP_RATIO = 0.5  # Ratio for drop-off logic in retrieval
RETRIEVAL_MIN_K = 2  # Minimum number of results to return

# TF-IDF configuration
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# Text preprocessing
CUSTOM_STOPWORDS = [
    "temperature",
    "preheat",
    "oven",
    "cook",
    "make",
    "recipe",
    "time",
    "minutes",
    "heat",
]

# Query processing options (can be toggled)
SIMPLIFY_QUERY = False
SPELL_CORRECT_QUERY = False
FILTER_CUSTOM_STOPWORDS = False

# Model configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Generation parameters
MAX_NEW_TOKENS = 1000
DO_SAMPLE = True

# Quantization configuration
BNB_LOAD_IN_4BIT = True
BNB_USE_DOUBLE_QUANT = True
BNB_QUANT_TYPE = "nf4"

# Sample retrieval queries (for testing)
SAMPLE_RETRIEVAL_QUERIES = [
    "cajun style gumbo with an easy roux",
    "shrimp tacos recipe for tonight",
    "vegetarian lasagna but easy",
    "spageti and meatballs?",
    "lunch baked at 200 °C",
]

# Logging
VERBOSE = False
