"""
Main entry point for the RAG pipeline.

This module orchestrates the entire recipe RAG (Retrieval-Augmented Generation)
pipeline:
1. Load dataset and queries
2. Preprocess recipe texts
3. Build and fit TF-IDF retriever
4. Evaluate retriever performance
5. Export retrieval results
6. Load LLM model
7. Generate answers

Usage:
    python main.py
"""

import json

import config as config
import data_loader as data_loader
import preprocessing as preprocessing
import retrieval as retrieval
import prompt_builder as prompt_builder

import time


def main():
    """Run the complete RAG pipeline."""

    print("=" * 80)
    print("RAG Pipeline for Recipe Reasoning")
    print("=" * 80)

    # Startup takes 1 min

    start = time.perf_counter()
    # ========== Step 1: Load Data ==========
    print("\n[Step 1] Loading data...")
    loader = data_loader.DataLoader()
    dataset = loader.load_dataset()
    queries = loader.load_queries()
    
    end = time.perf_counter()
    print(f"⏱ [Step 1: Loading data] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 1.5 seconds

    start = time.perf_counter()
    # ========== Step 2: Preprocess Dataset ==========
    print("\n[Step 2] Preprocessing dataset...")
    preprocessed_texts = preprocessing.preprocess_dataset(dataset)
    end = time.perf_counter()
    print(f"⏱ [Step 2: Preprocessing] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 110 seconds

    start = time.perf_counter()
    # ========== Step 3: Build and Fit TF-IDF Retriever ==========
    print("\n[Step 3] Building TF-IDF retriever...")
    retriever = retrieval.TFIDFRetriever()
    retriever.fit(preprocessed_texts)
    end = time.perf_counter()
    print(f"⏱ [Step 3: Building retriever] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 22 seconds

    start = time.perf_counter()
    # ========== Step 4: Evaluate Retriever ==========
    print("\n[Step 4] Evaluating retriever on dataset queries...")
    eval_results = retrieval.evaluate_retriever(
        retriever, queries, k=config.RETRIEVAL_K
    )
    end = time.perf_counter()
    print(f"⏱ [Step 4: Evaluating retriever] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 16 seconds

    # Print evaluation results
    print("\n--- Evaluation Metrics ---")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")

    # Save evaluation results
    eval_output_file = config.RESULTS_DIR / f"evaluation_{config.VERSION}.json"
    with open(eval_output_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    if config.VERBOSE:
        print(f"✓ Evaluation results saved to {eval_output_file}")

    start = time.perf_counter()
    # ========== Step 5: Export Retrieval Results ==========
    print("\n[Step 5] Exporting retrieval results...")
    all_queries = config.SAMPLE_RETRIEVAL_QUERIES
    retrievals = prompt_builder.export_retrievals(
        retriever, all_queries, queries, dataset, k=config.RETRIEVAL_K
    )
    end = time.perf_counter()
    print(f"⏱ [Step 5: Exporting retrievals] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 20 seconds
    
    # Save retrieval results
    with open(config.RETRIEVED_DOCS_OUTPUT, "w") as f:
        json.dump(retrievals, f, indent=2)
    print(f"✓ Retrieval results saved to {config.RETRIEVED_DOCS_OUTPUT}")

    start = time.perf_counter()
    # ========== Step 6: Load LLM Model ==========
    print("\n[Step 6] Loading LLM model...")
    print("  This may take a few minutes on first run...")
    generator = prompt_builder.LLMGenerator()
    generator.load_model()
    end = time.perf_counter()
    print(f"⏱ [Step 6: Loading LLM] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 280 seconds (4.5 min)

    start = time.perf_counter()
    # ========== Step 7: Generate Answers ==========
    print("\n[Step 7] Generating answers using LLM...")
    answers = prompt_builder.generate_answers(generator, retrievals)
    end = time.perf_counter()
    print(f"⏱ [Step 7: Generating answers] took {end - start:.2f} seconds ({(end - start)/60:.2f} min)")
    # 210 seconds (3.5 min)

    # ========== Pipeline Complete ==========
    print("\n" + "=" * 80)
    print("✨ RAG Pipeline Complete!")
    print("=" * 80)

    print(f"\nResults saved to:")
    print(f"  • Evaluation: {eval_output_file}")
    print(f"  • Retrievals: {config.RETRIEVED_DOCS_OUTPUT}")
    print(f"  • Answers: {config.GENERATED_ANSWERS_OUTPUT}")

    return {
        "dataset": dataset,
        "queries": queries,
        "retriever": retriever,
        "evaluation": eval_results,
        "retrievals": retrievals,
        "answers": answers,
        "generator": generator,
    }


if __name__ == "__main__":
    print("\n🚀 Starting RAG pipeline...")
    main()
