# src/generator.py

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY


class RAGGenerator:

    def __init__(
        self,
        retriever,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):

        self.retriever = retriever

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )


        self.prompt_template = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant. Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Answer accurately
- Use only context
- Do not hallucinate
- If not found, say "Answer not found in documents"

Answer:
"""
        )


    def build_context(
        self,
        documents: List[Document],
    ) -> str:

        context_parts = []

        for doc in documents:

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            context_parts.append(
                f"Source: {source} (Page {page})\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)


    def generate(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> str:

        retrieved_docs = self.retriever.retrieve(query, k=k)

        context = self.build_context(retrieved_docs)

        prompt = self.prompt_template.format(
            context=context,
            question=query,
        )
        print("Calling Groq LLM...")
        response = self.llm.invoke(prompt)
        print("Groq response received")

        return response.content


    def generate_with_sources(
        self,
        query: str,
        k: Optional[int] = None,
    ):

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