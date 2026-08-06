"""
Query augmentation strategies.

Implements query rewriting and Hypothetical Document Embeddings (HyDE)
to improve retrieval through query expansion and reformulation.
"""

from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import config


def build_rewrite_prompt(query: str, n: int) -> Tuple[str, str]:
    """
    Build system and user prompts for query rewriting.

    Args:
        query: Original research query.
        n: Number of alternative queries to generate.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are a search query optimization expert.
        Your task is to generate alternative search queries that explore different keywords and phrasings.
        These queries should help find papers that answer the same research question from different angles.
        RULES:
        - Generate exactly the requested number of queries
        - Each query should be a complete search phrase
        - Use academic terminology where appropriate
        - Make queries concise but specific
        - Do NOT include numbering or bullets in the output
        - Output one query per line"""

    user_prompt = f"""Generate {n} alternative search queries for this research question:

        Original query: {query}

        Alternative queries:"""

    return system_prompt, user_prompt


def build_hyde_prompt(query: str) -> Tuple[str, str]:
    """
    Build system and user prompts for hypothetical document generation.

    Args:
        query: Research question.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are an academic researcher.
        Your task is to write a short academic paragraph that directly answers a research question.
        This paragraph should be informative and include relevant terminology and concepts.
        RULES:
        - Write in academic style
        - Include specific technical terms when relevant
        - Make it informative and comprehensive
        - Keep it focused on answering the question
        - Do NOT include citations or reference numbers"""

    user_prompt = f"""Write an academic paragraph that answers this research question:

        Question: {query}

        Answer:"""

    return system_prompt, user_prompt


class QueryAugmenter:
    """Handles query rewriting and augmentation strategies."""

    def __init__(self, model_id: str = None, tokenizer=None, model=None):
        """
        Initialize query augmenter.

        Args:
            model_id: HuggingFace model ID for generation. If None, uses config.LLM_MODEL_ID.
            tokenizer: Pre-loaded tokenizer. If None, loads from model_id.
            model: Pre-loaded model. If None, loads from model_id.
        """
        self.model_id = model_id or config.LLM_MODEL_ID
        self.tokenizer = tokenizer
        self.model = model
        self.device = None

    def load_model(self) -> None:
        """Load tokenizer and model for query generation."""
        if self.tokenizer is not None and self.model is not None:
            if config.VERBOSE:
                print("Using pre-loaded tokenizer and model")
            return

        if config.VERBOSE:
            print(f"Loading model for query augmentation: {self.model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Setup device
        if torch.cuda.is_available():
            self.device = "cuda"
            if config.VERBOSE:
                print("  Using CUDA")
        else:
            self.device = "cpu"
            if config.VERBOSE:
                print("  Using CPU")

        import transformers

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=config.LLM_LOAD_IN_4BIT,
            bnb_4bit_use_double_quant=config.LLM_USE_DOUBLE_QUANT,
            bnb_4bit_quant_type=config.LLM_QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )

        if config.VERBOSE:
            print(f"✓ Model loaded!")

    def rewrite_query(self, query: str, n: int = None) -> List[str]:
        """
        Generate alternative query formulations using LLM.

        Args:
            query: Original query.
            n: Number of rewrites. If None, uses config.NUM_QUERY_REWRITES.
            verbose: Whether to print verbose output.

        Returns:
            List of rewritten queries.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if n is None:
            n = config.NUM_QUERY_REWRITES

        # if config.VERBOSE:
        #     print(f"Generating {n} query rewrites for query: {query}")

        # Build system and user prompts
        system_prompt, user_prompt = build_rewrite_prompt(query, n)

        # Format as chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template for the specific model
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Extract only the generated part (not the input)
        input_len = inputs["input_ids"].shape[-1]
        gen_only = outputs[0][input_len:]
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)
        
        # Extract and parse rewrites
        rewrites_text = text.strip()

        # Parse individual queries
        rewrites = []
        for line in rewrites_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Remove numbering and bullets
            line = line.lstrip("0123456789.-) ")

            if line and len(line) > 5:  # Avoid very short fragments
                rewrites.append(line)

        # Return only n rewrites
        return rewrites[:n]

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that answers the query (HyDE).

        This creates an artificial "gold" document that would answer the query,
        which is then embedded to find similar real documents.

        Args:
            query: Research question.

        Returns:
            Generated hypothetical document.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # if config.VERBOSE:
        #     print(f"Generating hypothetical document (HyDE) for query: {query}")

        # Build system and user prompts
        system_prompt, user_prompt = build_hyde_prompt(query)

        # Format as chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template for the specific model
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.HYDE_MAX_TOKENS,
                do_sample=True,
                temperature=config.HYDE_TEMPERATURE,
                top_p=0.9,
            )

        # Extract only the generated part (not the input)
        input_len = inputs["input_ids"].shape[-1]
        gen_only = outputs[0][input_len:]
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)

        return text.strip()


class QueryExpander:
    """Combines multiple query augmentation strategies."""

    def __init__(self, augmenter: QueryAugmenter):
        """
        Initialize query expander.

        Args:
            augmenter: QueryAugmenter instance.
        """
        self.augmenter = augmenter

    def expand_query(
        self,
        query: str,
        use_rewriting: bool = None,
        use_hyde: bool = None,
        num_rewrites: int = None,
    ) -> List[str]:
        """
        Expand query using multiple augmentation strategies.

        Args:
            query: Original query.
            use_rewriting: Whether to use query rewriting. If None, uses config.USE_QUERY_REWRITING.
            use_hyde: Whether to use HyDE. If None, uses config.USE_HYDE.
            num_rewrites: Number of rewrites. If None, uses config.NUM_QUERY_REWRITES.

        Returns:
            List of query variations (including original).
        """
        if use_rewriting is None:
            use_rewriting = config.USE_QUERY_REWRITING
        if use_hyde is None:
            use_hyde = config.USE_HYDE
        if num_rewrites is None:
            num_rewrites = config.NUM_QUERY_REWRITES

        queries = [query]  # Always include original

        # Query rewriting
        if use_rewriting:
            try:
                rewrites = self.augmenter.rewrite_query(query, n=num_rewrites)
                queries.extend(rewrites)
                if config.VERBOSE:
                    print(f"✓ Added {len(rewrites)} rewrites for query: {query}")
            except Exception as e:
                if config.VERBOSE:
                    print(f"⚠️  Query rewriting failed for query '{query}': {e}")

        # HyDE
        if use_hyde:
            try:
                hypo_doc = self.augmenter.generate_hypothetical_document(query)
                queries = [hypo_doc]
                if config.VERBOSE:
                    print(f"✓ Added hypothetical document for query: {query}")
            except Exception as e:
                if config.VERBOSE:
                    print(f"⚠️  HyDE generation failed for query '{query}': {e}")

        return queries
