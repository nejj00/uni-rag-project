"""
Data loading module for ACL Anthology dataset.

Handles loading and preparation of the ACL Anthology corpus with
query documents and random sampling for robustness.
"""

import json
from typing import Dict, Any, List

import datasets

import config


class ACLDataLoader:
    """Loads and manages the ACL Anthology dataset."""

    def __init__(self):
        """Initialize the data loader."""
        self.query_documents = None
        self.all_documents = None
        self.anthology_sample = None
        self.queries = None

    def load_datasets(self) -> datasets.Dataset:
        """
        Load and prepare the ACL Anthology dataset.

        Steps:
        1. Load query-relevant documents
        2. Load all documents
        3. Sample random documents using R_NUMBER_SEED
        4. Concatenate and shuffle

        Returns:
            The prepared anthology sample dataset.
        """
        if config.VERBOSE:
            print("Loading ACL Anthology datasets...")

        # Load query documents
        if config.VERBOSE:
            print(f"  Loading query documents from {config.QUERY_DOCS_FILE}...")

        self.query_documents = datasets.load_dataset(
            "parquet", data_files=str(config.QUERY_DOCS_FILE)
        )["train"]

        if config.VERBOSE:
            print(f"  ✓ Loaded {len(self.query_documents)} query documents")

        # Load all documents
        if config.VERBOSE:
            print(f"  Loading all documents from {config.ALL_DOCS_FILE}...")

        self.all_documents = datasets.load_dataset(
            "parquet", data_files=str(config.ALL_DOCS_FILE)
        )["train"]

        if config.VERBOSE:
            print(f"  ✓ Loaded {len(self.all_documents)} total documents")

        # Sample random documents with seed for reproducibility
        if config.VERBOSE:
            print(
                f"  Sampling {config.DOCS_TO_ADD} random documents "
                f"(seed={config.R_NUMBER_SEED})..."
            )

        random_documents = self.all_documents.shuffle(
            seed=config.R_NUMBER_SEED
        ).take(config.DOCS_TO_ADD)

        if config.VERBOSE:
            print(f"  ✓ Sampled {len(random_documents)} random documents")

        # Concatenate query documents with random sample
        if config.VERBOSE:
            print("  Concatenating and shuffling datasets...")

        self.anthology_sample = datasets.concatenate_datasets(
            [self.query_documents, random_documents]
        ).shuffle(seed=config.R_NUMBER_SEED)

        if config.VERBOSE:
            print(f"✓ Final anthology sample: {len(self.anthology_sample)} documents")

        # Cache to parquet to avoid re-downloading
        if config.VERBOSE:
            print(f"  Caching to {config.ANTHOLOGY_SAMPLE_PARQUET}...")

        self.anthology_sample.to_parquet(str(config.ANTHOLOGY_SAMPLE_PARQUET))

        if config.VERBOSE:
            print("✓ Dataset preparation complete!")

        return self.anthology_sample

    def load_queries(self, queries_path: str = None) -> Dict[str, Any]:
        """
        Load the evaluation queries from JSON file.

        Args:
            queries_path: Path to queries JSON. If None, uses config.QUERIES_FILE.

        Returns:
            Dictionary containing queries with structure:
            {"queries": [{"q": "...", "r": [...]}, ...]}

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
            print(f"✓ Loaded {num_queries} queries")

        return self.queries

    def get_sample(self) -> datasets.Dataset:
        """Get the loaded anthology sample."""
        if self.anthology_sample is None:
            raise RuntimeError(
                "Dataset not loaded. Call load_datasets() first."
            )
        return self.anthology_sample

    def get_queries(self) -> Dict[str, Any]:
        """Get the loaded queries."""
        if self.queries is None:
            raise RuntimeError("Queries not loaded. Call load_queries() first.")
        return self.queries

    def print_sample_doc(self, idx: int = 20) -> None:
        """Print a sample document for inspection."""
        if self.anthology_sample is None:
            raise RuntimeError("Dataset not loaded.")

        doc = self.anthology_sample[idx]
        print(f"\n{'='*80}")
        print(f"Sample Document {idx}:")
        print(f"{'='*80}")
        for key, value in doc.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"\n{key}:\n{value[:200]}...\n")
            else:
                print(f"{key}: {value}")

    def print_sample_query(self, idx: int = 0) -> None:
        """Print a sample query for inspection."""
        if self.queries is None:
            raise RuntimeError("Queries not loaded.")

        if idx >= len(self.queries["queries"]):
            print(f"Query {idx} not found. Available: {len(self.queries['queries'])}")
            return

        query = self.queries["queries"][idx]
        print(f"\n{'='*80}")
        print(f"Sample Query {idx}:")
        print(f"{'='*80}")
        for key, value in query.items():
            print(f"{key.upper()}: {value}")


def load_all_data() -> tuple:
    """
    Convenience function to load both datasets and queries.

    Returns:
        Tuple of (anthology_sample, queries).
    """
    loader = ACLDataLoader()
    anthology = loader.load_datasets()
    queries = loader.load_queries()
    return anthology, queries
