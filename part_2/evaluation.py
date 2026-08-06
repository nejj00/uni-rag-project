"""
Comprehensive evaluation metrics for retrieval systems.

Implements precision, recall, F1, MAP, MRR, nDCG for evaluating
retrieval quality against gold relevance judgments.
"""

from typing import Dict, Any, List, Callable, Optional
from pathlib import Path

import numpy as np

import config
from query_augmentation import QueryExpander


def evaluate_retriever(
    retriever_fn: Callable,
    queries_data: Dict[str, Any],
    anthology_sample,
    k: int = None,
    verbose: bool = None,
    verbose_detailed: bool = False,
    output_file: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate retriever on query set with gold relevance judgments.

    Computes both macro (per-query average) and micro (overall aggregate) metrics:
    - Precision, Recall, F1@k
    - Mean Average Precision (MAP)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (nDCG)

    Args:
        retriever_fn: Function with signature (query) -> (indices, scores)
                     Returns document indices and similarity scores.
        queries_data: Dictionary with "queries" key containing list of query dicts.
                     Each query dict has "q" (query text) and "r" (gold docs).
        anthology_sample: HuggingFace dataset for mapping indices to acl_ids.
        k: Evaluation cutoff. If None, uses config.EVAL_K.
        verbose: Print detailed results. If None, uses config.EVAL_VERBOSE.
        verbose_detailed: Write detailed per-query results to file.
        output_file: File to write detailed results. If None and verbose_detailed=True,
                    creates file in config.EVAL_DETAILED_DIR.

    Returns:
        Dictionary of evaluation metrics.
    """
    if k is None:
        k = config.EVAL_K
    if verbose is None:
        verbose = config.EVAL_VERBOSE

    # Set up output file if detailed logging requested
    file_handle = None
    if verbose_detailed or output_file:
        if output_file is None:
            output_file = config.EVAL_DETAILED_DIR / f"evaluation_details_{config.VERSION}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        file_handle = open(output_file, "w")
        print(f"📝 Detailed results will be saved to: {output_file}", file=None)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating Retriever (k={k})")
        print(f"{'='*80}\n")

    precisions = []
    recalls = []
    f1s = []
    reciprocal_ranks = []
    ndcgs = []
    average_precisions = []

    total_relevant_retrieved = 0
    total_retrieved = 0
    total_relevant_possible = 0

    queries_list = queries_data.get("queries", [])

    for q_idx, entry in enumerate(queries_list):
        query_text = entry["q"]

        # Parse gold relevant documents
        if isinstance(entry["r"][0], (list, tuple)):
            gold_ids = set(pair[0] for pair in entry["r"])
        else:
            gold_ids = set(entry["r"])

        # Retrieve documents
        doc_indices, scores = retriever_fn(query_text, k=k)

        if doc_indices is None or len(doc_indices) == 0:
            if verbose:
                print(f"⚠️  Query {q_idx + 1}: No results retrieved")
            continue

        # Map indices to ACL IDs
        retrieved_ids = [anthology_sample[int(i)]["acl_id"] for i in doc_indices]

        # Check hits
        hits = [1 if doc_id in gold_ids else 0 for doc_id in retrieved_ids]
        num_relevant_retrieved = sum(hits)

        # Precision, Recall, F1
        precision = num_relevant_retrieved / k
        recall = num_relevant_retrieved / len(gold_ids) if gold_ids else 0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in gold_ids:
                mrr = 1 / rank
                break
        reciprocal_ranks.append(mrr)

        # nDCG
        dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold_ids), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcgs.append(ndcg)

        # Average Precision (AP)
        precisions_at_hits = []
        hits_so_far = 0
        for i, hit in enumerate(hits):
            if hit:
                hits_so_far += 1
                precisions_at_hits.append(hits_so_far / (i + 1))
        ap = np.mean(precisions_at_hits) if precisions_at_hits else 0
        average_precisions.append(ap)

        # Micro accumulators
        total_relevant_retrieved += num_relevant_retrieved
        total_retrieved += k
        total_relevant_possible += len(gold_ids)

        # Write detailed results
        if verbose_detailed and file_handle:
            details = f"Query {q_idx + 1}: {query_text[:60]}...\n"
            details += f"  Gold docs: {sorted(gold_ids)}\n"

            for rank, (doc_id, score) in enumerate(zip(retrieved_ids, scores), 1):
                mark = "✅" if doc_id in gold_ids else "❌"
                details += f"  {rank:2d}. {doc_id} [{score:.4f}] {mark}\n"

            details += f"  P@{k}={precision:.3f} R@{k}={recall:.3f} F1={f1:.3f}\n"
            details += f"  AP={ap:.3f} MRR={mrr:.3f} nDCG={ndcg:.3f}\n\n"

            file_handle.write(details)

    # Macro metrics
    macro_precision = np.mean(precisions) if precisions else 0
    macro_recall = np.mean(recalls) if recalls else 0
    macro_f1 = np.mean(f1s) if f1s else 0
    map_score = np.mean(average_precisions) if average_precisions else 0

    # Micro metrics
    micro_precision = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
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

    # Print summary
    if verbose:
        print(f"{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Precision@{k}:  {macro_precision:.4f}")
        print(f"Recall@{k}:     {macro_recall:.4f}")
        print(f"F1@{k}:         {macro_f1:.4f}")
        print(f"MAP:            {map_score:.4f}")
        print(f"Micro Precision:{micro_precision:.4f}")
        print(f"Micro Recall:   {micro_recall:.4f}")
        print(f"Micro F1:       {micro_f1:.4f}")
        print(f"{'='*80}\n")

    # Write summary to file
    if file_handle:
        summary = f"\n{'='*80}\n"
        summary += "EVALUATION SUMMARY\n"
        summary += f"{'='*80}\n"
        summary += f"Precision@{k}:   {macro_precision:.4f}\n"
        summary += f"Recall@{k}:      {macro_recall:.4f}\n"
        summary += f"F1@{k}:          {macro_f1:.4f}\n"
        summary += f"MAP:             {map_score:.4f}\n"
        summary += f"Micro Precision: {micro_precision:.4f}\n"
        summary += f"Micro Recall:    {micro_recall:.4f}\n"
        summary += f"Micro F1:        {micro_f1:.4f}\n"
        summary += f"{'='*80}\n"
        file_handle.write(summary)
        file_handle.close()

    return {
        "Precision@k": macro_precision,
        "Recall@k": macro_recall,
        "F1@k": macro_f1,
        "MAP": map_score,
        "Micro_Precision": micro_precision,
        "Micro_Recall": micro_recall,
        "Micro_F1": micro_f1,
    }


def compare_strategies(
    strategies: Dict[str, Callable],
    queries_data: Dict[str, Any],
    anthology_sample,
    k: int = None,
    save_detailed: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple retrieval strategies on the same query set.

    Args:
        strategies: Dict of strategy_name -> retriever_fn.
        queries_data: Query data with gold judgments.
        anthology_sample: Dataset.
        k: Evaluation cutoff. If None, uses config.EVAL_K.
        save_detailed: Whether to save detailed per-query results.

    Returns:
        Dict of strategy_name -> metrics_dict.
    """
    if k is None:
        k = config.EVAL_K

    results = {}

    for strategy_name, retriever_fn in strategies.items():
        if config.VERBOSE:
            print(f"\n{'#'*80}")
            print(f"Evaluating strategy: {strategy_name}")
            print(f"{'#'*80}")

        # Create strategy-specific output file
        output_file = None
        if save_detailed:
            output_file = (
                config.EVAL_DETAILED_DIR / f"evaluation_{strategy_name}_{config.VERSION}.txt"
            )

        metrics = evaluate_retriever(
            retriever_fn,
            queries_data,
            anthology_sample,
            k=k,
            verbose=True,
            verbose_detailed=save_detailed,
            output_file=output_file,
        )
        results[strategy_name] = metrics

    return results


def evaluate_augmentation_strategies(
    rag_pipeline,
    queries_data: Dict[str, Any],
    anthology_sample,
    base_strategy: str = "chunks",
    k: int = None,
    save_detailed: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate query augmentation strategies (rewriting, HyDE) for a retrieval strategy.

    Tests different query augmentation configurations:
    - Baseline: No augmentation
    - Query Rewriting: LLM generates alternative query phrasings
    - HyDE: LLM generates hypothetical document
    - Combined: Both rewriting and HyDE

    Args:
        rag_pipeline: Initialized RAGPipeline with query_expander.
        queries_data: Query data with gold judgments.
        anthology_sample: HuggingFace dataset for mapping indices to acl_ids.
        base_strategy: Base retrieval strategy to test augmentation with (default: "chunks").
        k: Evaluation cutoff. If None, uses config.EVAL_K.
        save_detailed: Whether to save detailed per-query results.

    Returns:
        Dict of augmentation_config -> metrics_dict.
    """
    if k is None:
        k = config.EVAL_K

    if rag_pipeline.query_expander is None:
        print("⚠️  Query expander not initialized. Cannot evaluate augmentation strategies.")
        return {}

    results = {}

    # Define augmentation configurations
    augmentation_configs = {
        "Baseline": {
            "use_rewriting": False,
            "use_hyde": False,
        },
        "Query Rewriting": {
            "use_rewriting": True,
            "use_hyde": False,
        },
        "HyDE": {
            "use_rewriting": False,
            "use_hyde": True,
        },
        "Rewriting + HyDE": {
            "use_rewriting": True,
            "use_hyde": True,
        },
    }

    queries_list = queries_data.get("queries", [])

    for config_name, config_settings in augmentation_configs.items():
        if config.VERBOSE:
            print(f"\n{'#'*80}")
            print(f"Evaluating augmentation: {config_name}")
            print(f"  Strategy: {base_strategy}")
            print(f"  Rewriting: {config_settings['use_rewriting']}")
            print(f"  HyDE: {config_settings['use_hyde']}")
            print(f"{'#'*80}")

        def make_retriever(settings, strategy, pipeline, k_val):
            """Factory function to create retriever with proper closure."""
            def retriever_fn(query_text, k=None):
                """Retriever function that applies augmentation and retrieval."""
                if k is None:
                    k = k_val
                
                # Get augmented queries
                if settings["use_rewriting"] or settings["use_hyde"]:
                    augmented_queries = pipeline.query_expander.expand_query(
                        query_text,
                        use_rewriting=settings["use_rewriting"],
                        use_hyde=settings["use_hyde"],
                    )
                else:
                    augmented_queries = [query_text]

                # Retrieve with multiple queries
                if len(augmented_queries) > 1:
                    indices, scores = pipeline.retrieve_multiple_queries(
                        augmented_queries,
                        k=k,
                        fusion_method="voting",
                    )
                else:
                    indices, scores = pipeline.retrieve(augmented_queries[0], strategy=strategy, k=k)

                return indices, scores
            return retriever_fn

        retriever_fn = make_retriever(config_settings, base_strategy, rag_pipeline, k)

        # Create output file for this augmentation config
        output_file = None
        if save_detailed:
            output_file = (
                config.EVAL_DETAILED_DIR
                / f"evaluation_{base_strategy}_{config_name.replace(' ', '_')}_{config.VERSION}.txt"
            )

        metrics = evaluate_retriever(
            retriever_fn,
            queries_data,
            anthology_sample,
            k=k,
            verbose=True,
            verbose_detailed=save_detailed,
            output_file=output_file,
        )
        results[config_name] = metrics

    return results


def evaluate_all_combinations(
    query_expander: QueryExpander,
    embedding_manager,
    chunk_retrieval,
    hierarchical_retrieval,
    queries_data: Dict[str, Any],
    anthology_sample,
    k: int = None,
    save_detailed: bool = True,
    output_file: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Comprehensive evaluation of all (retrieval_strategy × augmentation_config) combinations.

    Systematically evaluates all combinations of:
    - 3 retrieval strategies: Dense, Chunks, Hierarchical
    - 4 augmentation configurations: Baseline, Query Rewriting, HyDE, Rewriting+HyDE
    Total: 12 combinations

    Returns nested dictionary structure:
    {
        "Dense": {
            "Baseline": {metrics...},
            "Query Rewriting": {metrics...},
            ...
        },
        "Chunks": {...},
        "Hierarchical": {...}
    }

    Args:
        query_expander: Initialized QueryExpander instance.
        embedding_manager: EmbeddingManager with built indices.
        chunk_retrieval: ChunkRetrieval instance.
        hierarchical_retrieval: HierarchicalRetrieval instance.
        queries_data: Query data with gold judgments.
        anthology_sample: HuggingFace dataset for mapping indices to acl_ids.
        k: Evaluation cutoff. If None, uses config.EVAL_K.
        save_detailed: Whether to save detailed per-query results.
        output_file: File to save consolidated results. If None, uses config.EVAL_OUTPUT_FILE.

    Returns:
        Nested dictionary of all combination results.
    """
    import json
    
    if k is None:
        k = config.EVAL_K

    if output_file is None:
        output_file = config.EVAL_OUTPUT_FILE

    if query_expander is None:
        print("⚠️  Query expander not initialized. Cannot evaluate augmentation strategies.")
        return {}

    # Define all retrieval strategies
    retrieval_strategies = {
        "Dense": {
            "retrieve_fn": lambda query_text, k_val: embedding_manager.search(
                embedding_manager.encode([query_text], show_progress=False), "dense", k_val
            ),
            "display_name": "Dense (Full-Document)"
        },
        "Chunks": {
            "retrieve_fn": lambda query_text, k_val: chunk_retrieval.retrieve(
                embedding_manager.encode([query_text], show_progress=False), k_val
            ),
            "display_name": "Chunks (Overlapping Windows)"
        },
        "Hierarchical": {
            "retrieve_fn": lambda query_text, k_val: hierarchical_retrieval.retrieve(
                query_text, embedding_manager.encode([query_text], show_progress=False), k_val
            ),
            "display_name": "Hierarchical (Two-Stage)"
        },
    }

    # Define augmentation configurations
    augmentation_configs = {
        "Baseline": {
            "use_rewriting": False,
            "use_hyde": False,
            "display_name": "No Augmentation"
        },
        "Query Rewriting": {
            "use_rewriting": True,
            "use_hyde": False,
            "display_name": "LLM Query Rewriting (3 alternatives)"
        },
        "HyDE": {
            "use_rewriting": False,
            "use_hyde": True,
            "display_name": "Hypothetical Document"
        },
    }

    all_results = {}
    total_combinations = len(retrieval_strategies) * len(augmentation_configs)
    current_combination = 0

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION: All {total_combinations} Combinations")
    print(f"Strategies: {len(retrieval_strategies)} × Augmentations: {len(augmentation_configs)}")
    print(f"{'='*80}\n")

    # Iterate through all combinations
    for strategy_name, strategy_config in retrieval_strategies.items():
        all_results[strategy_name] = {}
        strategy_retrieve_fn = strategy_config["retrieve_fn"]

        for aug_name, aug_config in augmentation_configs.items():
            current_combination += 1
            
            if config.VERBOSE:
                print(f"\n[{current_combination}/{total_combinations}] "
                      f"{strategy_name} + {aug_name}")
                print(f"  Strategy: {strategy_config['display_name']}")
                print(f"  Augmentation: {aug_config['display_name']}")
                print(f"  {'-'*76}")

            def make_retriever(aug_settings, strategy_retrieve, query_expander, k_val):
                """Factory function to create retriever with proper closure."""
                def retriever_fn(query_text, k=None):
                    """Retriever function that applies augmentation and retrieval."""
                    if k is None:
                        k = k_val

                    # Get augmented queries
                    if aug_settings["use_rewriting"] or aug_settings["use_hyde"]:
                        augmented_queries = query_expander.expand_query(
                            query_text,
                            use_rewriting=aug_settings["use_rewriting"],
                            use_hyde=aug_settings["use_hyde"],
                        )
                    else:
                        augmented_queries = [query_text]

                    # Retrieve with multiple queries
                    if len(augmented_queries) > 1:
                        # For multi-query retrieval, need to aggregate results
                        all_indices_scores = []
                        for expanded_query in augmented_queries:
                            indices, scores = strategy_retrieve(expanded_query, k)
                            all_indices_scores.append((indices, scores))

                        # Voting fusion: count occurrences of each document
                        doc_votes = {}
                        doc_max_score = {}
                        for indices, scores in all_indices_scores:
                            for idx, score in zip(indices, scores):
                                idx_int = int(idx)
                                doc_votes[idx_int] = doc_votes.get(idx_int, 0) + 1
                                doc_max_score[idx_int] = max(doc_max_score.get(idx_int, 0), score)

                        # Sort by vote count, then by max score
                        sorted_docs = sorted(
                            doc_votes.items(),
                            key=lambda x: (x[1], doc_max_score[x[0]]),
                            reverse=True
                        )
                        
                        indices = [idx for idx, _ in sorted_docs[:k]]
                        scores = [doc_max_score[idx] for idx, _ in sorted_docs[:k]]
                    else:
                        indices, scores = strategy_retrieve(augmented_queries[0], k)

                    return indices, scores

                return retriever_fn

            retriever_fn = make_retriever(aug_config, strategy_retrieve_fn, query_expander, k)

            # Create output file for this combination
            output_details_file = None
            if save_detailed:
                output_details_file = (
                    config.EVAL_DETAILED_DIR
                    / f"evaluation_{strategy_name}_{aug_name.replace(' ', '_')}.txt"
                )

            # Evaluate this combination
            metrics = evaluate_retriever(
                retriever_fn,
                queries_data,
                anthology_sample,
                k=k,
                verbose=config.VERBOSE,
                verbose_detailed=save_detailed,
                output_file=output_details_file,
            )
            all_results[strategy_name][aug_name] = metrics

    # Save comprehensive results to single file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ All {total_combinations} combinations evaluated!")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Detailed results saved to: {config.EVAL_DETAILED_DIR}/")
    print(f"{'='*80}\n")

    return all_results
