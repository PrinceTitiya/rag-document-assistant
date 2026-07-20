# src/llm.py

import logging
import os
from langchain_groq import ChatGroq
from groq import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def build_llm(
    model_name: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> ChatGroq:
    """
    Construct the shared Groq chat model used across the pipeline (query
    expansion and answer generation), failing fast with a clear message if
    the API key is missing rather than surfacing a confusing connection
    error later when the model is first invoked.
    """

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is not set. Add GROQ_API_KEY=your_key_here to a "
            ".env file in the project root (see README.md)."
        )

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        model_kwargs={"top_p": top_p},
    )


# Errors worth retrying: rate limits, timeouts, transient connection issues,
# and 5xx server errors. Everything else (bad request, auth failure, etc.)
# is a real bug or config problem and should surface immediately, not retry.
RETRYABLE_GROQ_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

# Shared retry policy for Groq LLM calls. This pipeline makes two LLM calls
# per user question (query expansion + answer generation), which roughly
# doubles token usage compared to a single-call pipeline. Free-tier Groq
# accounts can have fairly tight tokens-per-minute limits, so a short burst
# of requests can trip a rate limit even under normal interactive use.
# Retrying with backoff handles that transiently instead of the whole
# request failing outright.
retry_on_transient_groq_error = retry(
    retry=retry_if_exception_type(RETRYABLE_GROQ_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
