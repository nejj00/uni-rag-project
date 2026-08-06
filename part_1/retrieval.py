"""
Information retrieval module using TF-IDF and cosine similarity.

This module implements document retrieval functionality for the RAG pipeline
using TF-IDF vectorization and cosine similarity scoring.
"""

from typing import Tuple, Dict, List, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import config
import preprocessing


class TFIDFRetriever:
    """TF-IDF based document retriever."""

    def __init__(self, ngram_range: Tuple[int, int] = None):
        """
        Initialize the TF-IDF retriever.

        Args:
            ngram_range: Tuple specifying n-gram range (min, max).
                        If None, uses config.TFIDF_NGRAM_RANGE.
        """
        if ngram_range is None:
            ngram_range = config.TFIDF_NGRAM_RANGE

        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        self.tfidf_matrix = None
        self.is_fitted = False

    def fit(self, documents: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on documents.

        Args:
            documents: List of document texts to fit on.
        """
        if config.VERBOSE:
            print("Fitting TF-IDF vectorizer...")

        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.is_fitted = True

        if config.VERBOSE:
            print(f"✓ TF-IDF fitted successfully!")
            print(f"  Documents: {self.tfidf_matrix.shape[0]}")
            print(f"  Features: {self.tfidf_matrix.shape[1]}")
            sparsity = (
                1
                - self.tfidf_matrix.nnz
                / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])
            ) * 100
            print(f"  Sparsity: {sparsity:.2f}%")
            memory_mb = self.tfidf_matrix.data.nbytes / 1024 / 1024
            print(f"  Memory: {memory_mb:.1f} MB")

    def retrieve(
        self,
        query: str,
        k: int = None,
        drop_ratio: float = None,
        min_k: int = None,
    ) -> Tuple[Optional[List[int]], Optional[List[float]]]:
        """
        Retrieve top-k most similar documents for a query.

        This method implements a dynamic retrieval strategy:
        1. Process and vectorize the query
        2. Compute cosine similarities with all documents
        3. Apply drop-off logic: stop when similarity drops below drop_ratio
        4. Ensure at least min_k results are returned

        Args:
            query: Query text.
            k: Maximum number of documents to retrieve. If None, uses config.RETRIEVAL_K.
            drop_ratio: Ratio for drop-off logic. If None, uses config.RETRIEVAL_DROP_RATIO.
            min_k: Minimum number of results to return. If None, uses config.RETRIEVAL_MIN_K.

        Returns:
            Tuple of (indices, scores) for top-k documents, or (None, None) if query is invalid.
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever not fitted. Call fit() first.")

        if k is None:
            k = config.RETRIEVAL_K
        if drop_ratio is None:
            drop_ratio = config.RETRIEVAL_DROP_RATIO
        if min_k is None:
            min_k = config.RETRIEVAL_MIN_K
    
        # Process and vectorize query
        processed_query = preprocessing.process_query(query)
        query_vec = self.vectorizer.transform([processed_query])

        # Case 1: Query has usable terms
        if query_vec.nnz > 0:
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            # distances = euclidean_distances(query_vec, self.tfidf_matrix).flatten()
            # similarities = -distances  # invert so higher is better

            # Sort by similarity (descending)
            sorted_indices = similarities.argsort()[::-1]
            sorted_scores = similarities[sorted_indices]

            # Always include the best match
            selected_indices = [sorted_indices[0]]

            # Apply drop-off logic: stop if similarity drops significantly
            for i in range(1, len(sorted_scores)):
                prev_score = sorted_scores[i - 1]
                curr_score = sorted_scores[i]

                # Stop if sharp drop
                if curr_score < drop_ratio * prev_score:
                    break

                selected_indices.append(sorted_indices[i])

                # Respect max k
                if len(selected_indices) >= k:
                    break

            # Ensure at least min_k results
            if len(selected_indices) < min_k:
                selected_indices = sorted_indices[:min_k]

            selected_scores = similarities[selected_indices]
            return selected_indices, selected_scores

        # Case 2: Try spell correction fallback
        corrected = preprocessing.spell_correct(query)
        processed_query = preprocessing.process_query(corrected)
        query_vec = self.vectorizer.transform([processed_query])

        if query_vec.nnz > 0:
            # similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            distances = euclidean_distances(query_vec, self.tfidf_matrix).flatten()
            similarities = -distances  # invert so higher is better
            
            sorted_indices = similarities.argsort()[::-1]
            return sorted_indices[:k], similarities[sorted_indices[:k]]

        # Case 3: No usable query
        if config.VERBOSE:
            print(f"⚠️  Query '{query}' could not be processed into a valid vector.")
        return None, None


def evaluate_retriever(
    retriever: TFIDFRetriever,
    queries_data: Dict[str, Any],
    k: int = None,
) -> Dict[str, float]:
    """
    Evaluate the retriever on a set of queries with gold relevance judgments.

    Computes both macro and micro averaged metrics:
    - Precision, Recall, F1-score
    - Mean Average Precision (MAP)

    Args:
        retriever: Fitted TFIDFRetriever instance.
        queries_data: Dictionary with 'queries' key containing list of query dicts.
                     Each query dict should have 'q' (query text) and 'r' (relevance pairs).
        k: Number of top results to consider. If None, uses config.RETRIEVAL_K.

    Returns:
        Dictionary with evaluation metrics.
    """
    if k is None:
        k = config.RETRIEVAL_K

    if config.VERBOSE:
        print(f"Evaluating retriever with k={k}...")

    all_precisions = []
    all_recalls = []
    all_f1s = []
    average_precisions = []

    total_relevant_retrieved = 0
    total_retrieved = 0
    total_relevant_possible = 0

    for entry in queries_data.get("queries", []):
        query_text = entry["q"]

        # Extract gold relevant document IDs
        gold_ids = set([pair[0] for pair in entry["r"]])

        # Retrieve documents
        retrieved_indices, _ = retriever.retrieve(query_text, k=k)

        if retrieved_indices is None:
            continue

        # Convert to 1-indexed IDs (as in the dataset)
        retrieved_ids = [idx + 1 for idx in retrieved_indices]

        # Check which retrieved docs are relevant
        hits = [1 if doc_id in gold_ids else 0 for doc_id in retrieved_ids]
        num_relevant_retrieved = sum(hits)

        # Macro metrics
        precision = num_relevant_retrieved / k
        recall = num_relevant_retrieved / len(gold_ids) if len(gold_ids) > 0 else 0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        # Average Precision (AP)
        precisions_at_hits = []
        hits_so_far = 0
        for i, hit in enumerate(hits):
            if hit == 1:
                hits_so_far += 1
                precisions_at_hits.append(hits_so_far / (i + 1))

        ap = np.mean(precisions_at_hits) if precisions_at_hits else 0
        average_precisions.append(ap)

        # Micro accumulators
        total_relevant_retrieved += num_relevant_retrieved
        total_retrieved += k
        total_relevant_possible += len(gold_ids)

    # Micro metrics
    micro_precision = (
        total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    )
    micro_recall = (
        total_relevant_retrieved / total_relevant_possible
        if total_relevant_possible > 0
        else 0
    )
    micro_f1 = (
        (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0
    )

    results = {
        "MAP": np.mean(average_precisions),
        "Macro_Precision": np.mean(all_precisions),
        "Macro_Recall": np.mean(all_recalls),
        "Macro_F1": np.mean(all_f1s),
        "Micro_Precision": micro_precision,
        "Micro_Recall": micro_recall,
        "Micro_F1": micro_f1,
    }

    if config.VERBOSE:
        print("✓ Evaluation complete:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

    return results
