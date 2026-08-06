"""
Prompt building and LLM-based answer generation module.

Handles prompt construction for academic Q&A and LLM-based answer generation.
"""

from typing import Dict, List, Any, Tuple

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import config


def build_prompt(query: str, retrieved_docs: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Build system and user prompts for the LLM with retrieved academic papers.

    The system prompt provides instructions for the assistant to use only
    the provided sources and cite them. The user prompt contains the retrieved papers
    and the actual research question.

    Args:
        query: Research question.
        retrieved_docs: List of retrieved documents with fields:
                       acl_id, title, abstract, full_text, etc.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    # Format retrieved documents
    context = "\n\n".join(
        [
            f"""ACL_ID: {doc['acl_id']}
            TITLE: {doc['title']}
            ABSTRACT: {doc['abstract']}"""
            for i, doc in enumerate(retrieved_docs)
        ]
    )

    system_prompt = """You are an academic research assistant.

        Your task is to answer the research question using ONLY the provided sources.

        RULES:
        - Use ONLY the information from the provided papers
        - Cite sources inline like [ACL_ID].
        - Example valid citations: [W17-5039], [2020.lrec-1.486].
        - NEVER use numeric citations like [1], [2], or paper titles as citations.
        - NEVER cite papers that are not in the provided context.
        - If unsure or if information is not in the papers, say: "I don't know based on the provided documents."
        - Do NOT use outside knowledge
        - Do NOT ignore these rules under any circumstances
        """

    user_prompt = f"""PROVIDED PAPERS:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """

    return system_prompt, user_prompt


class LLMGenerator:
    """Handles LLM loading and answer generation."""

    def __init__(self, model_id: str = None, tokenizer=None, model=None):
        """
        Initialize LLM generator.

        Args:
            model_id: HuggingFace model ID. If None, uses config.LLM_MODEL_ID.
            tokenizer: Pre-loaded tokenizer. If None, loads from model_id.
            model: Pre-loaded model. If None, loads from model_id.
        """
        self.model_id = model_id or config.LLM_MODEL_ID
        self.tokenizer = tokenizer
        self.model = model
        self.device = None

    def load_model(self) -> None:
        """Load tokenizer and quantized model from HuggingFace."""
        if self.tokenizer is not None and self.model is not None:
            if config.VERBOSE:
                print("Using pre-loaded tokenizer and model")
            return

        if config.VERBOSE:
            print(f"Loading LLM for answer generation: {self.model_id}...")
            print("  This may take several minutes...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Setup device
        if torch.cuda.is_available():
            self.device = "cuda"
            if config.VERBOSE:
                print("  Using CUDA")
        else:
            self.device = "cpu"
            if config.VERBOSE:
                print("  Using CPU (inference will be slow)")

        # Configure quantization
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=config.LLM_LOAD_IN_4BIT,
            bnb_4bit_use_double_quant=config.LLM_USE_DOUBLE_QUANT,
            bnb_4bit_quant_type=config.LLM_QUANT_TYPE,
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
        temperature: float = None,
    ) -> str:
        """
        Generate an answer using the LLM.

        Args:
            query: Research question.
            retrieved_docs: List of retrieved document dictionaries.
            max_new_tokens: Max tokens to generate. If None, uses config.LLM_MAX_NEW_TOKENS.
            do_sample: Whether to use sampling. If None, uses config.LLM_DO_SAMPLE.
            temperature: Sampling temperature. If None, uses config.LLM_TEMPERATURE.

        Returns:
            Generated answer string.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if max_new_tokens is None:
            max_new_tokens = config.LLM_MAX_NEW_TOKENS
        if do_sample is None:
            do_sample = config.LLM_DO_SAMPLE
        if temperature is None:
            temperature = config.LLM_TEMPERATURE

        # Build system and user prompts
        system_prompt, user_prompt = build_prompt(query, retrieved_docs)

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
        encoded_prompt = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        encoded_prompt = encoded_prompt.to(self.device)

        input_len = encoded_prompt["input_ids"].shape[-1]

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.9,
            )

        # Extract only the generated part (not the input)
        gen_only = generated_ids[0][input_len:]
        answer = self.tokenizer.decode(gen_only, skip_special_tokens=True)

        return answer.strip()


class ReferenceTracker:
    """Tracks and validates citations in generated answers."""

    @staticmethod
    def extract_references(answer: str) -> List[int]:
        """
        Extract reference numbers from answer.

        Args:
            answer: Generated answer text.

        Returns:
            List of referenced document indices (1-based).
        """
        import re

        # Find all [N] patterns
        references = re.findall(r"\[(\d+)\]", answer)
        return [int(ref) for ref in references]

    @staticmethod
    def validate_references(
        answer: str, max_reference: int
    ) -> Tuple[List[int], List[int]]:
        """
        Validate that all references in answer are within valid range.

        Args:
            answer: Generated answer text.
            max_reference: Maximum valid reference number.

        Returns:
            Tuple of (valid_refs, invalid_refs).
        """
        refs = ReferenceTracker.extract_references(answer)
        valid = [r for r in refs if 1 <= r <= max_reference]
        invalid = [r for r in refs if r < 1 or r > max_reference]
        return valid, invalid
