"""
Prompt building and LLM generation module for the RAG pipeline.

This module handles prompt construction and LLM-based answer generation
using HuggingFace Transformers with quantized model loading.
"""

import json
from typing import Dict, List, Any, Tuple

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import config


def build_prompt(
    query: str, retrieved_docs: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Build a system and user prompt for the LLM.

    The system prompt provides instructions for the assistant to use only
    the provided recipes. The user prompt contains the retrieved recipes
    and the actual question.

    Args:
        query: The user's question.
        retrieved_docs: List of retrieved document dictionaries, each containing
                       'score', 'name', 'ingredients', 'description', 'tags', 'steps'.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    # Format retrieved documents
    context = "\n\n".join(
        [
            # f"Score: {doc['score']}\n"
            f"Name: {doc['name']}\n"
            f"Ingredients: {doc['ingredients']}\n"
            f"Description: {doc['description']}\n"
            f"Tags: {doc['tags']}\n"
            f"Steps: {doc['steps']}\n"
            for doc in retrieved_docs
        ]
    )

    system_prompt = """You are a cooking assistant.

    You MUST follow these rules exactly:
    
    1. Use ONLY the provided recipes.
    2. If the answer is not explicitly supported, respond EXACTLY with:
       "I don't have enough information from the provided recipes."
    3. Do NOT guess, infer, or use outside knowledge.
    4. Do NOT continue the conversation.
    5. Do NOT generate additional questions or answers.
    6. Output ONLY the final answer, nothing else.
    7. Keep the answer concise.
    8. If naming something, it MUST appear explicitly in the recipes.
    9. Respect any constraints in the question (e.g., no tomatoes, vegetarian, etc.).
    10. If the question asks for a recipe, provide structured cooking steps.
    11. If ANY recipe contains relevant information, you MUST use it.
    """
    
    # system_prompt = """
    # You are a cooking assistant.

    # You MUST follow these rules exactly:

    # 1. Use ONLY the provided recipes.
    # 2. Retrieval scores indicate estimated relevance:
    # - Higher score = more relevant and trustworthy for answering the question.
    # - Prefer information from higher-scoring recipes when conflicts occur.
    # - Do not rely heavily on low-scoring recipes unless multiple recipes support the same point.
    # 3. If the answer is not explicitly supported by the recipes, respond EXACTLY with:
    # "I don't have enough information from the provided recipes."
    # 4. Do NOT use outside knowledge.
    # 5. Do NOT invent ingredients, recipes, substitutions, or cooking steps.
    # 6. Do NOT continue the conversation.
    # 7. Output ONLY the final answer.
    # 8. Keep the answer concise but complete.
    # 9. Respect all constraints in the question
    # (e.g., vegetarian, gluten-free, no tomatoes).
    # 10. If the question asks for a recipe:
    #     - Prefer the highest-scoring matching recipe.
    #     - Provide clear structured cooking steps.
    # 11. If multiple high-scoring recipes are relevant:
    #     - Combine overlapping information carefully.
    #     - Mention recipe names explicitly.
    # 12. Ignore recipes that are clearly unrelated to the question,
    #     especially if they have low retrieval scores.
    # """

    user_prompt = f"""
    RECIPES:
    {context}
    
    QUESTION:
    {query}
    """

    return system_prompt, user_prompt


class LLMGenerator:
    """Handles LLM model loading and answer generation."""

    def __init__(self, model_id: str = None):
        """
        Initialize the LLM generator.

        Args:
            model_id: HuggingFace model ID. If None, uses config.MODEL_ID.
        """
        self.model_id = model_id or config.MODEL_ID
        self.tokenizer = None
        self.model = None
        self.device = None

    def _setup_device(self) -> None:
        """Setup GPU/CPU device and check CUDA availability."""
        if torch.cuda.is_available():
            self.device = "cuda"
            if config.VERBOSE:
                print(f"✓ CUDA available. Using GPU.")
        else:
            self.device = "cpu"
            if config.VERBOSE:
                print("⚠️  CUDA not available. Using CPU (inference will be slow).")

    def load_model(self) -> None:
        """
        Load tokenizer and quantized model from HuggingFace.

        Uses 4-bit quantization with BitsAndBytes to reduce memory usage.
        """
        self._setup_device()

        if config.VERBOSE:
            print(f"Loading tokenizer for {self.model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if config.VERBOSE:
            print(f"Loading model {self.model_id}...")
            print("  This may take a few minutes...")

        # Configure quantization
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=config.BNB_LOAD_IN_4BIT,
            bnb_4bit_use_double_quant=config.BNB_USE_DOUBLE_QUANT,
            bnb_4bit_quant_type=config.BNB_QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )

        if config.VERBOSE:
            print(f"✓ Model loaded successfully!")

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        max_new_tokens: int = None,
        do_sample: bool = None,
    ) -> str:
        """
        Generate an answer using the LLM for a given query and documents.

        Args:
            query: The user's question.
            retrieved_docs: List of retrieved recipe documents.
            max_new_tokens: Maximum new tokens to generate. If None, uses config.MAX_NEW_TOKENS.
            do_sample: Whether to use sampling. If None, uses config.DO_SAMPLE.

        Returns:
            Generated answer string.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if max_new_tokens is None:
            max_new_tokens = config.MAX_NEW_TOKENS
        if do_sample is None:
            do_sample = config.DO_SAMPLE

        # Build prompts
        system_prompt, user_prompt = build_prompt(query, retrieved_docs)

        # Format as chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        encoded_prompt = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        encoded_prompt = encoded_prompt.to(self.device)

        input_len = encoded_prompt["input_ids"].shape[-1]

        # Generate
        generated_ids = self.model.generate(
            **encoded_prompt, max_new_tokens=max_new_tokens, do_sample=do_sample
        )

        # Extract only the generated part (not the input)
        gen_only = generated_ids[0][input_len:]
        answer = self.tokenizer.decode(gen_only, skip_special_tokens=True)

        return answer.strip()


def export_retrievals(
    retriever,
    retrieval_queries: List[str],
    queries_data: Dict[str, Any],
    dataset,
    k: int = None,
) -> Dict[str, Any]:
    """
    Export all retrieved documents for both custom and dataset queries.

    Args:
        retriever: Fitted TFIDFRetriever instance.
        retrieval_queries: List of custom test queries.
        queries_data: Dictionary containing dataset queries.
        dataset: The recipe dataset.
        k: Number of documents to retrieve. If None, uses config.RETRIEVAL_K.

    Returns:
        Dictionary with all retrieval results.
    """
    if k is None:
        k = config.RETRIEVAL_K

    if config.VERBOSE:
        print(f"Exporting retrieval results for {len(retrieval_queries)} custom queries...")

    results = {}

    # Handle custom retrieval queries
    for i, q in enumerate(retrieval_queries):
        indices, scores = retriever.retrieve(q, k=k)

        if indices is None:
            if config.VERBOSE:
                print(f"  Skipping custom query {i}: No valid results")
            continue

        docs = []
        for idx, score in zip(indices, scores):
            doc = dataset[int(idx)]
            docs.append(
                {
                    "id": int(idx),
                    "score": float(score),
                    "name": doc["name"],
                    "ingredients": doc["ingredients"],
                    "description": doc["description"],
                    "tags": doc["tags"],
                    "steps": doc["steps"],
                }
            )

        results[f"retrieval_query_{i}"] = {"query": q, "documents": docs}

    if config.VERBOSE:
        print(
            f"Exporting retrieval results for {len(queries_data.get('queries', []))} dataset queries..."
        )

    # Handle dataset queries
    for i, entry in enumerate(queries_data.get("queries", [])):
        q = entry["q"]
        indices, scores = retriever.retrieve(q, k=k)

        if indices is None:
            if config.VERBOSE:
                print(f"  Skipping dataset query {i}: No valid results")
            continue

        docs = []
        for idx, score in zip(indices, scores):
            doc = dataset[int(idx)]
            docs.append(
                {
                    "id": int(idx),
                    "score": float(score),
                    "name": doc["name"],
                    "ingredients": doc["ingredients"],
                    "description": doc["description"],
                    "tags": doc["tags"],
                    "steps": doc["steps"],
                }
            )

        results[f"dataset_query_{i}"] = {"query": q, "documents": docs}

    if config.VERBOSE:
        print(f"✓ Exported {len(results)} retrieval results")

    return results


def generate_answers(
    generator: LLMGenerator,
    retrievals: Dict[str, Any],
    output_file: str = None,
) -> Dict[str, Any]:
    """
    Generate answers for all retrieved queries using the LLM.

    Args:
        generator: Initialized LLMGenerator instance.
        retrievals: Dictionary of retrieval results from export_retrievals().
        output_file: Path to save generated answers. If None, uses config.GENERATED_ANSWERS_OUTPUT.

    Returns:
        Dictionary with all generated answers.
    """
    if output_file is None:
        output_file = config.GENERATED_ANSWERS_OUTPUT

    if config.VERBOSE:
        print(f"Generating answers for {len(retrievals)} queries...")

    outputs = {}

    for i, (key, item) in enumerate(retrievals.items(), 1):
        query = item["query"]
        retrieved_docs = item["documents"]

        if config.VERBOSE:
            print(f"\n🔍 Processing {i}/{len(retrievals)}: {key}")
            print(f"   Query: {query}")

        answer = generator.generate_answer(query, retrieved_docs)

        outputs[key] = {"query": query, "answer": answer}

        if config.VERBOSE:
            print(f"✅ Done")

    if config.VERBOSE:
        print(f"\nSaving answers to {output_file}...")

    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=2)

    if config.VERBOSE:
        print(f"🎉 Saved {len(outputs)} answers to {output_file}")

    return outputs
