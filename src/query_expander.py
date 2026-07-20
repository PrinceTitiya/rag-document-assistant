# src/query_expander.py

import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate

from src.llm import retry_on_transient_groq_error, RETRYABLE_GROQ_ERRORS

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Generates alternative phrasings of a question using the LLM, so retrieval
    isn't limited to whatever exact wording the user happened to type.

    This closes a real gap that hybrid (dense + BM25) search cannot: a
    document can fully answer a question using entirely different vocabulary
    than the question itself (e.g. explaining "proof of work" and a block
    validation algorithm in detail, without ever using the words "consensus
    mechanism"). Dense embeddings match on general semantic similarity and
    BM25 matches on literal keyword overlap, but both operate on the
    question's own wording — if the answer-bearing passage shares neither
    the phrasing nor the keywords, neither retriever can find it. Rewriting
    the question into more specific, technical-term-bearing variants before
    retrieval is what actually bridges that gap.
    """

    PROMPT = ChatPromptTemplate.from_template(
        "You are helping search a technical document. Generate {n} alternative "
        "phrasings or more specific sub-questions for the question below, using "
        "different vocabulary and more concrete technical terms where relevant. "
        "This is only for improving search recall, not for answering the question.\n\n"
        "Return ONLY the {n} alternatives, one per line, no numbering, no bullets, "
        "no extra commentary.\n\n"
        "Question: {question}"
    )

    def __init__(self, llm, num_variants: int = 3):
        self.llm = llm
        self.num_variants = num_variants


    @retry_on_transient_groq_error
    def _invoke(self, prompt: str):
        return self.llm.invoke(prompt)


    def expand(self, question: str) -> List[str]:
        """
        Returns up to `num_variants` alternative phrasings of `question`.

        Query expansion is a quality *enhancement*, not something the
        pipeline can't function without — so if the LLM call still fails
        after retries (e.g. a sustained rate limit), this logs a warning
        and returns an empty list, degrading gracefully to hybrid search on
        the original question only rather than failing the whole request.
        The warning is logged (not silently swallowed) so a sustained
        pattern of failures is actually visible instead of just quietly
        making retrieval worse.
        """

        prompt = self.PROMPT.format(n=self.num_variants, question=question)

        try:
            response = self._invoke(prompt)
        except RETRYABLE_GROQ_ERRORS as e:
            logger.warning(
                "Query expansion skipped after retries failed (%s: %s) - "
                "falling back to hybrid search on the original question only.",
                type(e).__name__, e,
            )
            return []
        except Exception as e:
            logger.warning("Query expansion skipped due to unexpected error: %s", e)
            return []

        lines = [
            line.strip(" -•\t")
            for line in response.content.strip().split("\n")
            if line.strip()
        ]

        return lines[: self.num_variants]
