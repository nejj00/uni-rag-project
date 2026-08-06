"""
Configuration module for Part 2: ACL Anthology RAG Pipeline.

This module contains all configuration settings for dense retrieval,
long document handling, query augmentation, and LLM generation.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Random seed for reproducible sampling
R_NUMBER_SEED = 1034593

# Number of random documents to add to query documents
DOCS_TO_ADD = 2000

# Dataset files
QUERY_DOCS_FILE = INPUTS_DIR / "acl_anthology_queries.parquet"
ALL_DOCS_FILE = INPUTS_DIR / "acl_anthology_full.parquet"
QUERIES_FILE = INPUTS_DIR / "acl_anthology_queries.json"
ANTHOLOGY_SAMPLE_PARQUET = INPUTS_DIR / "anthology_sample.parquet"

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================

# Sentence Transformer model
# All-MiniLM-L6-v2: Fast, efficient (384-dim), good semantic similarity
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Batch size for encoding
EMBEDDING_BATCH_SIZE = 32

# ============================================================================
# RETRIEVAL STRATEGIES
# ============================================================================

# Dense retrieval (full document)
DENSE_K = 5  # Number of documents to retrieve

# Chunking strategy
CHUNK_SIZE = 200  # Words per chunk
CHUNK_OVERLAP = 50  # Overlapping words between chunks

# Hierarchical two-stage retrieval strategy
HIERARCHICAL_TOP_CANDIDATES = 50  # Number of papers to search deeply (Stage 2)
# Section weights are in hierarchical_retrieval.py

# ============================================================================
# QUERY AUGMENTATION
# ============================================================================

# Query rewriting
NUM_QUERY_REWRITES = 3  # Number of alternative queries to generate

# Hypothetical Document Embeddings (HyDE)
HYDE_MAX_TOKENS = 150
HYDE_TEMPERATURE = 0.7

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Model
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Generation parameters
LLM_MAX_NEW_TOKENS = 1000
LLM_DO_SAMPLE = True
LLM_TEMPERATURE = 0.7

# Quantization
LLM_LOAD_IN_4BIT = True
LLM_USE_DOUBLE_QUANT = True
LLM_QUANT_TYPE = "nf4"

# ============================================================================
# RAG PIPELINE
# ============================================================================

# Default retrieval strategy
# Options: "dense", "chunks", "hierarchical"
DEFAULT_RETRIEVAL_STRATEGY = "chunks"

# Top-k documents for final answer generation
RAG_TOP_K = 5

# Whether to use query augmentation (rewriting + HyDE)
USE_QUERY_AUGMENTATION = True
USE_QUERY_REWRITING = True
USE_HYDE = False

# ============================================================================
# EVALUATION
# ============================================================================

RUN_EVALUATION = False

# Evaluation K
EVAL_K = 5

# Verbose evaluation output
EVAL_VERBOSE = True

# ============================================================================
# OUTPUT FILES
# ============================================================================

def build_version():
    parts = [DEFAULT_RETRIEVAL_STRATEGY]

    if USE_QUERY_AUGMENTATION:
        if USE_QUERY_REWRITING:
            parts.append("rewrite")
        if USE_HYDE:
            parts.append("hyde")
    else:
        parts.append("noaug")

    return "_".join(parts)

VERSION = build_version()

# Evaluation results
EVAL_OUTPUT_FILE = RESULTS_DIR / f"evaluation_results_{VERSION}.json"

# Retrieval results
RETRIEVAL_OUTPUT_FILE = RESULTS_DIR / f"retrieval_results_{VERSION}.json"

# Generation results
GENERATION_OUTPUT_FILE = RESULTS_DIR / f"generation_results_{VERSION}.json"

# Detailed evaluation logs (by strategy)
EVAL_DETAILED_DIR = RESULTS_DIR / "evaluation_details"
EVAL_DETAILED_DIR.mkdir(exist_ok=True)

# ============================================================================
# GENERAL OPTIONS
# ============================================================================

VERBOSE = False
