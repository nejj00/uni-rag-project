"""
Data loading module for the RAG pipeline.

This module handles loading the recipe dataset and query data from the
HuggingFace datasets library and JSON files.
"""

import json
from typing import Dict, Any, List

import datasets

import config


class DataLoader:
    """Handles loading and management of datasets and queries."""

    def __init__(self):
        """Initialize the data loader."""
        self.dataset = None
        self.queries = None

    def load_dataset(self, dataset_path: str = None) -> datasets.Dataset:
        """
        Load the recipe dataset from a parquet file.

        Args:
            dataset_path: Path to the parquet file. If None, uses config.DATASET_FILE.

        Returns:
            The loaded HuggingFace dataset.

        Raises:
            FileNotFoundError: If dataset file not found.
        """
        if dataset_path is None:
            dataset_path = config.DATASET_FILE

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        if config.VERBOSE:
            print(f"Loading dataset from {dataset_path}...")

        self.dataset = datasets.load_dataset(
            "parquet", data_files=str(dataset_path)
        )["train"]

        if config.VERBOSE:
            print(f"✓ Dataset loaded successfully: {len(self.dataset)} recipes")

        return self.dataset

    def load_queries(self, queries_path: str = None) -> Dict[str, Any]:
        """
        Load the queries from a JSON file.

        Args:
            queries_path: Path to the queries JSON file. If None, uses config.QUERIES_FILE.

        Returns:
            Dictionary containing query data with 'queries' key.

        Raises:
            FileNotFoundError: If queries file not found.
        """
        if queries_path is None:
            queries_path = config.QUERIES_FILE

        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found at {queries_path}")

        if config.VERBOSE:
            print(f"Loading queries from {queries_path}...")

        with open(queries_path, "r") as f:
            self.queries = json.load(f)

        num_queries = len(self.queries.get("queries", []))
        if config.VERBOSE:
            print(f"✓ Queries loaded successfully: {num_queries} queries")

        return self.queries

    def get_dataset(self) -> datasets.Dataset:
        """Get the loaded dataset."""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        return self.dataset

    def get_queries(self) -> Dict[str, Any]:
        """Get the loaded queries."""
        if self.queries is None:
            raise RuntimeError("Queries not loaded. Call load_queries() first.")
        return self.queries


def load_all_data() -> tuple:
    """
    Convenience function to load both dataset and queries.

    Returns:
        Tuple of (dataset, queries).
    """
    loader = DataLoader()
    dataset = loader.load_dataset()
    queries = loader.load_queries()
    return dataset, queries
