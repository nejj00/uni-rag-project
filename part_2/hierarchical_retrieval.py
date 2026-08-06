"""
Hierarchical two-stage retrieval with section awareness.

Implements efficient retrieval by filtering candidates via abstract embedding,
then performing deep full-text search with section-level semantic understanding.
"""

from typing import List, Tuple, Dict
import re

import numpy as np

import config


class SectionExtractor:
    """Extracts semantic sections from academic papers."""

    # Section keywords for common ACL paper structure
    SECTION_KEYWORDS = {
        "introduction": ["introduction", "intro"],
        "related": ["related work", "background", "related"],
        "methods": ["method", "approach", "model", "architecture"],
        "experiments": ["experiment", "experiments", "evaluation", "results"],
        "results": ["result", "results", "performance", "findings"],
        "analysis": ["analysis", "discussion", "ablation"],
        "conclusion": ["conclusion", "conclusions", "future work"],
    }

    SECTION_WEIGHTS = {
        "methods": 1.0,
        "results": 0.95,
        "experiments": 0.9,
        "analysis": 0.85,
        "conclusion": 0.8,
        "introduction": 0.6,
        "related": 0.5,
        "abstract": 0.7,
    }

    def __init__(self):
        """Initialize section extractor."""
        pass

    @staticmethod
    def _find_section_start(lines: List[str], keywords: List[str]) -> int:
        """
        Find line index where a section starts.

        Args:
            lines: List of text lines.
            keywords: Keywords that mark section start.

        Returns:
            Index of section start, or -1 if not found.
        """
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                return i
        return -1

    @staticmethod
    def _extract_section(
        full_text: str, keywords: List[str], max_chars: int = 1500
    ) -> str:
        """
        Extract a section from full text using keyword heuristic.

        Args:
            full_text: Full paper text.
            keywords: Keywords marking section start.
            max_chars: Maximum characters to extract.

        Returns:
            Section text, or empty string if not found.
        """
        if not full_text or not keywords:
            return ""

        lines = full_text.split("\n")
        start_idx = SectionExtractor._find_section_start(lines, keywords)

        if start_idx < 0:
            return ""

        # Collect lines until next section marker
        section_lines = []
        common_markers = (
            "##",
            "#",
            "Abstract",
            "References",
            "Acknowledgments",
        )

        for line in lines[start_idx + 1 :]:
            # Stop at next section
            if any(line.startswith(marker) for marker in common_markers):
                break
            section_lines.append(line)

        section_text = " ".join(section_lines).strip()
        return section_text[:max_chars]

    def extract_sections(self, full_text: str) -> Dict[str, str]:
        """
        Extract all major sections from paper.

        Args:
            full_text: Full paper text.

        Returns:
            Dictionary mapping section_name -> section_text.
        """
        sections = {}

        for section_name, keywords in self.SECTION_KEYWORDS.items():
            section_text = self._extract_section(full_text, keywords)
            if section_text and len(section_text) > 50:  # Only keep substantial sections
                sections[section_name] = section_text

        # Fallback: use first 1500 chars as overview
        if not sections:
            sections["overview"] = full_text[:1500]

        return sections


class HierarchicalRetrieval:
    """Two-stage retrieval: Abstract filter → Full-text semantic search."""

    def __init__(self, embedding_manager):
        """
        Initialize hierarchical retrieval.

        Args:
            embedding_manager: EmbeddingManager instance for encoding.
        """
        self.embedding_manager = embedding_manager
        self.section_extractor = SectionExtractor()
        self.anthology_sample = None

    def build_index(self, anthology_sample) -> None:
        """
        Build hierarchical retrieval index.

        Stage 1: Build FAISS index on abstracts for fast filtering.
        Stage 2: Store anthology sample for on-demand full-text section extraction.

        Args:
            anthology_sample: HuggingFace dataset of documents.
        """
        if config.VERBOSE:
            print("Building hierarchical retrieval index...")

        # Stage 1: Index abstracts for fast filtering
        if config.VERBOSE:
            print("  Stage 1: Building abstract index...")

        abstracts = [doc.get("abstract", "") or "" for doc in anthology_sample]
        abstract_embeddings = self.embedding_manager.encode(
            abstracts, show_progress=True
        )
        self.embedding_manager.build_index(abstract_embeddings, strategy="abstracts")

        # Stage 2: Store anthology for on-demand full-text extraction
        self.anthology_sample = anthology_sample

        if config.VERBOSE:
            print(f"✓ Hierarchical index built for {len(anthology_sample)} documents")

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        top_candidates: int = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Two-stage retrieval: Filter candidates by abstract, then search full text.

        Args:
            query: Query text.
            query_embedding: Encoded query (1, embedding_dim).
            k: Number of documents to return.
            top_candidates: Number of candidates to search in Stage 2.
                           If None, uses config.HIERARCHICAL_TOP_CANDIDATES.

        Returns:
            Tuple of (doc_indices, scores).
        """
        if top_candidates is None:
            top_candidates = config.HIERARCHICAL_TOP_CANDIDATES

        # Stage 1: Fast filtering via abstract index
        if config.VERBOSE:
            print(
                f"Stage 1: Filtering {len(self.anthology_sample)} papers by abstract..."
            )

        candidate_indices, _ = self.embedding_manager.search(
            query_embedding, strategy="dense", k=top_candidates
        )

        if config.VERBOSE:
            print(f"Stage 2: Deep search in top-{len(candidate_indices)} papers...")

        # Stage 2: Semantic search in full text sections
        doc_scores = {}

        for doc_idx in candidate_indices:
            if doc_idx < 0:
                continue

            doc_idx = int(doc_idx)
            doc = self.anthology_sample[doc_idx]
            full_text = doc.get("full_text", "") or ""

            if not full_text or len(full_text) < 100:
                doc_scores[doc_idx] = 0.0
                continue

            # Extract sections
            sections = self.section_extractor.extract_sections(full_text)

            if not sections:
                doc_scores[doc_idx] = 0.0
                continue

            # Score each section
            section_scores = []

            for section_name, section_text in sections.items():
                if not section_text or len(section_text) < 20:
                    continue

                try:
                    # Encode section
                    section_embedding = self.embedding_manager.encode(
                        [section_text], show_progress=False, normalize=True
                    )

                    # Compute similarity
                    similarity = (
                        section_embedding @ query_embedding.T
                    ).flatten()[0]

                    # Weight by section importance
                    section_weight = self.section_extractor.SECTION_WEIGHTS.get(
                        section_name, 0.5
                    )
                    weighted_score = similarity * section_weight

                    section_scores.append(weighted_score)

                except Exception as e:
                    if config.VERBOSE:
                        print(f"⚠️  Error scoring section {section_name}: {e}")
                    continue

            # Aggregate section scores
            if section_scores:
                doc_scores[doc_idx] = float(np.mean(section_scores))
            else:
                doc_scores[doc_idx] = 0.0

        # Rank documents
        ranked_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        top_docs = ranked_docs[:k]
        doc_indices = [doc_id for doc_id, _ in top_docs]
        doc_scores_result = [score for _, score in top_docs]

        return doc_indices, doc_scores_result
