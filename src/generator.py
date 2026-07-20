# src/generator.py

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.llm import retry_on_transient_groq_error, RETRYABLE_GROQ_ERRORS


class RAGGenerator:
    """
    Handles prompt construction and LLM response generation.
    """

    def __init__(
        self,
        retriever,
        llm,
    ):
        """
        Args:
            retriever: RAGRetriever instance
            llm: shared chat model (see src/llm.py) used for answer generation.
                 Shared with the retriever's query expansion step so the
                 pipeline only holds one LLM client.
        """

        self.retriever = retriever
        self.llm = llm

        self.prompt_template = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Answer clearly and accurately
- Use only the context provided
- Do not hallucinate
- If answer is not in context, say "Answer not found in documents"

Answer:
"""
        )


    def build_context(
        self,
        documents: List[Document],
    ) -> str:
        """
        Build context string from retrieved documents.
        """

        if not documents:
            return ""

        context_parts = []

        for doc in documents:

            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "Unknown page")

            context_parts.append(
                f"Source: {source} (Page: {page})\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)


    @retry_on_transient_groq_error
    def _invoke_llm(self, prompt: str):
        return self.llm.invoke(prompt)


    def _generate_from_docs(
        self,
        query: str,
        retrieved_docs: List[Document],
    ) -> str:
        """
        Build the prompt from already-retrieved documents and call the LLM.
        """

        context = self.build_context(retrieved_docs)

        prompt = self.prompt_template.format(
            context=context,
            question=query,
        )

        try:
            response = self._invoke_llm(prompt)
        except RETRYABLE_GROQ_ERRORS as e:
            raise RuntimeError(
                f"The Groq API is rate-limited or temporarily unavailable "
                f"({type(e).__name__}), and retrying didn't succeed. Wait a "
                "minute and try again, or send fewer questions in quick "
                "succession — this pipeline makes two LLM calls per "
                "question (query expansion + answer generation), so it uses "
                "roughly double the API quota of a single-call pipeline."
            ) from e

        return response.content


    def generate(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> str:
        """
        Generate answer using retrieved documents and LLM.
        """

        if not query:
            raise ValueError("Query cannot be empty")

        retrieved_docs = self.retriever.retrieve(query, k=k)

        return self._generate_from_docs(query, retrieved_docs)


    def generate_with_sources(
        self,
        query: str,
        k: Optional[int] = None,
    ):
        """
        Generate answer and return sources.
        Useful for frontend or APIs.
        """

        if not query:
            raise ValueError("Query cannot be empty")

        # Retrieve once so the reported sources always match the context
        # actually sent to the LLM (previously this retrieved twice, doubling
        # embedding cost and letting sources drift from the generated answer).
        retrieved_docs = self.retriever.retrieve(query, k=k)

        answer = self._generate_from_docs(query, retrieved_docs)

        sources = [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            }
            for doc in retrieved_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
        }