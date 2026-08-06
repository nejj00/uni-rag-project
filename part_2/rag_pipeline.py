"""
Integrated RAG pipeline for ACL Anthology question answering.

Combines retrieval strategies, query augmentation, and LLM generation
into a cohesive pipeline for academic paper retrieval and answering.
"""

from typing import Dict, List, Any, Callable, Optional

import numpy as np

import config
from embeddings import EmbeddingManager, combine_acl_fields
from chunking import ChunkRetrieval
from hierarchical_retrieval import HierarchicalRetrieval
from prompt_builder import build_prompt, LLMGenerator, ReferenceTracker


class RAGPipeline:
    """
    Complete RAG pipeline for academic paper retrieval and Q&A.

    Implements multiple retrieval strategies and optional query augmentation,
    integrated with an LLM for answer generation.
    """

    def __init__(
        self,
        anthology_sample,
        embedding_manager: EmbeddingManager,
        llm_generator: LLMGenerator,
        chunk_retrieval: ChunkRetrieval = None,
        hierarchical_retrieval: HierarchicalRetrieval = None,
        query_expander=None,
        retrieval_strategy: str = None,
        top_k: int = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            anthology_sample: HuggingFace dataset of documents.
            embedding_manager: EmbeddingManager instance.
            llm_generator: LLMGenerator instance.
            chunk_retrieval: Pre-built ChunkRetrieval instance (optional).
            hierarchical_retrieval: Pre-built HierarchicalRetrieval instance (optional).
            query_expander: Optional QueryExpander for query augmentation.
            retrieval_strategy: Retrieval strategy ("dense", "chunks", "hierarchical").
                               If None, uses config.DEFAULT_RETRIEVAL_STRATEGY.
            top_k: Number of documents to retrieve. If None, uses config.RAG_TOP_K.
        """
        self.anthology_sample = anthology_sample
        self.embedding_manager = embedding_manager
        self.llm_generator = llm_generator
        self.query_expander = query_expander
        self.retrieval_strategy = retrieval_strategy or config.DEFAULT_RETRIEVAL_STRATEGY
        self.top_k = top_k or config.RAG_TOP_K

        # Use pre-built retrieval modules or initialize for lazy building
        self.chunk_retrieval = chunk_retrieval
        self.hierarchical_retrieval = hierarchical_retrieval

    def _ensure_dense_index(self) -> None:
        """Ensure dense document index exists."""
        if "dense" in self.embedding_manager.indices:
            return

        if config.VERBOSE:
            print("Building dense document index...")

        docs = [combine_acl_fields(doc) for doc in self.anthology_sample]
        embeddings = self.embedding_manager.encode(docs, show_progress=True)
        self.embedding_manager.build_index(embeddings, strategy="dense")

    def _ensure_chunk_index(self) -> None:
        """Ensure chunked document index exists."""
        if self.chunk_retrieval is not None:
            return

        if config.VERBOSE:
            print("Building chunk index...")

        self.chunk_retrieval = ChunkRetrieval(self.embedding_manager)
        self.chunk_retrieval.build_chunk_index(self.anthology_sample)

    def _ensure_hierarchical_index(self) -> None:
        """Ensure hierarchical two-stage retrieval index exists."""
        if self.hierarchical_retrieval is not None:
            return

        if config.VERBOSE:
            print("Building hierarchical retrieval index...")

        self.hierarchical_retrieval = HierarchicalRetrieval(self.embedding_manager)
        self.hierarchical_retrieval.build_index(self.anthology_sample)

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query to embedding."""
        return self.embedding_manager.encode([query], show_progress=False)

    def retrieve_dense(self, query: str, k: int = None) -> tuple:
        """Retrieve using full-document dense embeddings."""
        if k is None:
            k = self.top_k

        self._ensure_dense_index()
        query_emb = self._encode_query(query)
        indices, scores = self.embedding_manager.search(query_emb, "dense", k)
        return indices, scores

    def retrieve_chunks(self, query: str, k: int = None) -> tuple:
        """Retrieve using chunked documents."""
        if k is None:
            k = self.top_k

        self._ensure_chunk_index()
        query_emb = self._encode_query(query)
        indices, scores = self.chunk_retrieval.retrieve(query_emb, k)
        return indices, scores

    def retrieve_hierarchical(self, query: str, k: int = None) -> tuple:
        """Retrieve using hierarchical two-stage retrieval."""
        if k is None:
            k = self.top_k

        self._ensure_hierarchical_index()
        query_emb = self._encode_query(query)
        indices, scores = self.hierarchical_retrieval.retrieve(query, query_emb, k)
        return indices, scores

    def retrieve(
        self,
        query: str,
        strategy: str = None,
        k: int = None,
    ) -> tuple:
        """
        Retrieve documents using specified strategy.

        Args:
            query: Query text.
            strategy: Retrieval strategy ("dense", "chunks", "hierarchical").
                     If None, uses self.retrieval_strategy.
            k: Number of documents. If None, uses self.top_k.

        Returns:
            Tuple of (doc_indices, scores).
        """
        if strategy is None:
            strategy = self.retrieval_strategy
        if k is None:
            k = self.top_k

        if strategy == "dense":
            return self.retrieve_dense(query, k)
        elif strategy == "chunks":
            return self.retrieve_chunks(query, k)
        elif strategy == "hierarchical":
            return self.retrieve_hierarchical(query, k)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

    def process_query(self, query: str) -> List[str]:
        """
        Process query with optional augmentation.

        Args:
            query: Original query.

        Returns:
            List of queries (original + augmented variants).
        """
        if self.query_expander is None or not config.USE_QUERY_AUGMENTATION:
            return [query]

        expanded = self.query_expander.expand_query(query)
        return expanded

    def retrieve_multiple_queries(
        self,
        queries: List[str],
        k: int = None,
        fusion_method: str = "voting",
    ) -> tuple:
        """
        Retrieve documents using multiple query variants with fusion.

        Args:
            queries: List of query variants.
            k: Number of final documents to return. If None, uses self.top_k.
            fusion_method: How to combine results ("voting" or "score_sum").

        Returns:
            Tuple of (doc_indices, fused_scores).
        """
        if k is None:
            k = self.top_k

        if len(queries) == 1:
            return self.retrieve(queries[0], k=k)

        if config.VERBOSE:
            print(f"Retrieving with {len(queries)} query variants...")

        # Retrieve with each query
        all_results = {}  # doc_idx -> list of scores

        for q in queries:
            indices, scores = self.retrieve(q, k=k * 2)

            for idx, score in zip(indices, scores):
                if idx not in all_results:
                    all_results[idx] = []
                all_results[idx].append(score)

        # Fuse results
        if fusion_method == "voting":
            # Count how many queries retrieved each doc
            fused_scores = {
                idx: float(len(scores)) for idx, scores in all_results.items()
            }
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Rank and return top-k
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = ranked[:k]

        doc_indices = [idx for idx, _ in top_docs]
        doc_scores = [score for _, score in top_docs]

        return doc_indices, doc_scores

    def generate_answer(
        self,
        query: str,
        retrieved_indices: List[int],
        strategy: str = None,
    ) -> str:
        """
        Generate an answer using the LLM.

        Args:
            query: Original question.
            retrieved_indices: Indices of retrieved documents.
            strategy: Retrieval strategy (for logging). If None, uses self.retrieval_strategy.

        Returns:
            Generated answer string.
        """
        if strategy is None:
            strategy = self.retrieval_strategy

        if config.VERBOSE:
            print(f"Generating answer ({strategy} strategy)...")

        # Fetch document details
        retrieved_docs = [
            self.anthology_sample[int(idx)] for idx in retrieved_indices
        ]

        # Generate answer with query and documents
        answer = self.llm_generator.generate_answer(query, retrieved_docs)

        return answer

    def run(
        self,
        query: str,
        strategy: str = None,
        use_augmentation: bool = None,
        k: int = None,
    ) -> Dict[str, Any]:
        """
        Run complete RAG pipeline: query processing, retrieval, answer generation.

        Args:
            query: User question.
            strategy: Retrieval strategy. If None, uses self.retrieval_strategy.
            use_augmentation: Whether to use query augmentation. If None, uses config.USE_QUERY_AUGMENTATION.
            k: Number of documents to retrieve. If None, uses self.top_k.

        Returns:
            Dictionary with results including query, expanded_queries, retrieved_docs, and answer.
        """
        if strategy is None:
            strategy = self.retrieval_strategy
        if use_augmentation is None:
            use_augmentation = config.USE_QUERY_AUGMENTATION
        if k is None:
            k = self.top_k

        if config.VERBOSE:
            print(f"\n{'='*80}")
            print(f"RAG Pipeline Query")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Strategy: {strategy}")
            print(f"Augmentation: {use_augmentation}")

        # Step 1: Process query
        if use_augmentation and self.query_expander:
            expanded_queries = self.process_query(query)
            if config.VERBOSE:
                print(f"Expanded to {len(expanded_queries)} queries")
        else:
            expanded_queries = [query]

        # Step 2: Retrieve
        if use_augmentation and len(expanded_queries) > 1:
            doc_indices, scores = self.retrieve_multiple_queries(
                expanded_queries, k=k, fusion_method="voting"
            )
        else:
            doc_indices, scores = self.retrieve(expanded_queries[0], strategy=strategy, k=k)

        if config.VERBOSE:
            print(f"Retrieved {len(doc_indices)} documents")

        # Step 3: Generate answer
        answer = self.generate_answer(query, doc_indices, strategy=strategy)

        # Step 4: Validate references
        valid_refs, invalid_refs = ReferenceTracker.validate_references(
            answer, len(doc_indices)
        )

        if invalid_refs and config.VERBOSE:
            print(f"⚠️  Invalid references found: {invalid_refs}")

        return {
            "query": query,
            "expanded_queries": expanded_queries,
            "doc_indices": doc_indices,
            "retrieved_docs": [
                self.anthology_sample[int(idx)] for idx in doc_indices
            ],
            "answer": answer,
            "valid_references": valid_refs,
            "invalid_references": invalid_refs,
            "strategy": strategy,
        }
