"""
Main entry point for Part 2: ACL Anthology RAG Pipeline.

Orchestrates data loading, index building, evaluation, and RAG pipeline execution.
"""

import json

import config
from data_loader import ACLDataLoader, load_all_data
from embeddings import EmbeddingManager
from prompt_builder import LLMGenerator
from query_augmentation import QueryAugmenter, QueryExpander
from evaluation import evaluate_all_combinations
from rag_pipeline import RAGPipeline
from chunking import ChunkRetrieval
from hierarchical_retrieval import HierarchicalRetrieval

import time


def main():
    """Run the complete Part 2 RAG pipeline."""

    print("=" * 80)
    print("ACL Anthology RAG Pipeline - Part 2")
    print("=" * 80)

    start = time.perf_counter()
    # ========== Step 1: Load Data ==========
    print("\n[Step 1] Loading data...")
    anthology_sample, queries = load_all_data()
    end = time.perf_counter()
    print(f"⏱ [Step 1: Loading data] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 30 seconds
    
    start = time.perf_counter()
    # ========== Step 2: Initialize Components ==========
    print("\n[Step 2] Initializing components...")

    # Embedding manager
    embedding_manager = EmbeddingManager()
    embedding_manager.load_model()

    # LLM generator
    llm_generator = LLMGenerator()
    llm_generator.load_model()

    # Query augmenter (optional)
    query_augmenter = QueryAugmenter(tokenizer=llm_generator.tokenizer, model=llm_generator.model)
    query_augmenter.device = llm_generator.device
    query_expander = QueryExpander(query_augmenter)
    end = time.perf_counter()
    print(f"⏱ [Step 2: Initializing components] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 320 seconds (5 minutes 20 seconds)

    start = time.perf_counter()
    # ========== Step 3: Build Retrieval Indices ==========
    print("\n[Step 3] Building retrieval indices...")
    print("  This may take a few minutes for larger datasets...")

    # Dense retrieval index
    from embeddings import combine_acl_fields
    
    docs = [combine_acl_fields(doc) for doc in anthology_sample]
    embeddings = embedding_manager.encode(docs, show_progress=True)
    embedding_manager.build_index(embeddings, strategy="dense")

    # Chunk retrieval index
    chunk_retrieval = ChunkRetrieval(embedding_manager)
    chunk_retrieval.build_chunk_index(anthology_sample)

    # Hierarchical retrieval index
    hierarchical_retrieval = HierarchicalRetrieval(embedding_manager)
    hierarchical_retrieval.build_index(anthology_sample)

    print("✓ All indices built successfully!")
    end = time.perf_counter()
    print(f"⏱ [Step 3: Building retrieval indices] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 125 seconds (2 minutes 5 seconds)

    # # ========== Step 4: Evaluate Retrieval Strategies ==========
    
    if config.RUN_EVALUATION:
        start = time.perf_counter()
        eval_output_file = config.RESULTS_DIR / f"evaluation_results_ALL.json"
        
        evaluate_all_combinations(
            query_expander=query_expander,
            embedding_manager=embedding_manager,
            chunk_retrieval=chunk_retrieval,
            hierarchical_retrieval=hierarchical_retrieval,
            queries_data=queries,
            anthology_sample=anthology_sample,
            k=config.EVAL_K,
            save_detailed=config.EVAL_VERBOSE,
            output_file=eval_output_file,
        )
        end = time.perf_counter()
        print(f"⏱ [Step 4: Evaluating retrieval strategies] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
        # 60 - 75 mins
    
    start = time.perf_counter()
    # ========== Step 5: Initialize RAG Pipeline ==========
    print("\n[Step 5] Initializing RAG pipeline...")

    rag_pipeline = RAGPipeline(
        anthology_sample=anthology_sample,
        embedding_manager=embedding_manager,
        llm_generator=llm_generator,
        chunk_retrieval=chunk_retrieval,
        hierarchical_retrieval=hierarchical_retrieval,
        query_expander=query_expander,
        retrieval_strategy=config.DEFAULT_RETRIEVAL_STRATEGY,
        top_k=config.RAG_TOP_K,
    )
    end = time.perf_counter()
    print(f"⏱ [Step 5: Initializing RAG pipeline] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # instantaneous
    
    start = time.perf_counter()
    # ========== Step 6: Test Queries ==========
    print("\n[Step 6] Running full pipeline...")

    # test_queries = [
    #     "What are some multilingual pretrained models for African languages?",
    #     "How can transformers be used for machine translation?",
    #     "What are recent advances in few-shot learning?",
    # ]

    results = []

    queries_list = queries.get("queries", [])
    for query in queries_list:
        print(f"\nTest Query: {query['q']}")
        result = rag_pipeline.run(
            query['q'],
            strategy=config.DEFAULT_RETRIEVAL_STRATEGY,
            use_augmentation=config.USE_QUERY_AUGMENTATION,
        )
        results.append(result)

        if config.VERBOSE:
            print(f"\n📄 Retrieved Documents:")
            for i, doc in enumerate(result["retrieved_docs"][:config.RAG_TOP_K], 1):
                print(f"{i}. {doc['title']} ({doc['acl_id']})")

            print(f"\n🤖 Generated Answer:")
            print(result["answer"])

    # Save generation results
    generation_results = {
        f"query_{i}": {
            "query": r["query"],
            "expanded_queries": r["expanded_queries"],
            "strategy": r["strategy"],
            "retrieved_docs": [
                {"acl_id": doc["acl_id"], "title": doc["title"]}
                for doc in r["retrieved_docs"]
            ],
            "answer": r["answer"],
        }
        for i, r in enumerate(results)
    }

    with open(config.GENERATION_OUTPUT_FILE, "w") as f:
        json.dump(generation_results, f, indent=2)

    print(f"\n✓ Generation results saved to {config.GENERATION_OUTPUT_FILE}")
    
    end = time.perf_counter()
    print(f"⏱ [Step 6: Running full pipeline] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # (17 minutes) - chunking + rewrites
    
    # ========== Pipeline Complete ==========
    print("\n" + "=" * 80)
    print("✨ ACL Anthology RAG Pipeline Complete!")
    print("=" * 80)

    print(f"\nOutput files:")
    print(f"  • Evaluation: {config.EVAL_OUTPUT_FILE}")
    print(f"  • Generation: {config.GENERATION_OUTPUT_FILE}")

    return {
        "anthology_sample": anthology_sample,
        "queries": queries,
        "embedding_manager": embedding_manager,
        "llm_generator": llm_generator,
        "rag_pipeline": rag_pipeline,
        "results": results,
    }


if __name__ == "__main__":
    main()
