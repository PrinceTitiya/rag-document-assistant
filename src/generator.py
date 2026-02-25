# src/generator.py

from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class RAGGenerator:
    """
    Handles prompt construction and LLM response generation.
    """

    def __init__(
        self,
        retriever,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        """
        Args:
            retriever: RAGRetriever instance
            model_name: Gemini model name
            temperature: randomness control
            top_p: nucleus sampling parameter
            max_output_tokens: response length limit
        """

        self.retriever = retriever

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
        )

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

        # Step 1: retrieve relevant docs
        retrieved_docs = self.retriever.retrieve(query, k=k)

        # Step 2: build context
        context = self.build_context(retrieved_docs)

        # Step 3: build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query,
        )

        # Step 4: call LLM
        response = self.llm.invoke(prompt)

        return response.content


    def generate_with_sources(
        self,
        query: str,
        k: Optional[int] = None,
    ):
        """
        Generate answer and return sources.
        Useful for frontend or APIs.
        """

        retrieved_docs = self.retriever.retrieve(query, k=k)

        answer = self.generate(query, k=k)

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