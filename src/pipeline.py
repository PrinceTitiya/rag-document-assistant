# src/pipeline.py

from typing import Optional

from src.document_loader import DocumentLoader
from src.embedding_manager import EmbeddingManager
from src.vectorstore_manager import VectorStoreManager
from src.retriever import RAGRetriever
from src.generator import RAGGenerator

from src.config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
)


class RAGPipeline:
    """
    Main pipeline that orchestrates the full RAG system.
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        persist_dir: str = VECTORSTORE_DIR,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        llm_model_name: str = LLM_MODEL_NAME,
    ):
        """
        Initialize full RAG pipeline.
        """

        # Step 1: Load and split documents
        self.document_loader = DocumentLoader(data_dir)

        chunks = self.document_loader.load_and_split()


        # Step 2: Initialize embedding model
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model_name
        )

        embedding_model = self.embedding_manager.get_embedding_model()


        # Step 3: Create or load vectorstore
        self.vectorstore_manager = VectorStoreManager(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )

        self.vectorstore = self.vectorstore_manager.get_vectorstore(
            chunks=chunks
        )


        # Step 4: Initialize retriever
        self.retriever = RAGRetriever(
            vectorstore=self.vectorstore,
            search_type="mmr",
            k=3,
            fetch_k=10,
            lambda_mult=0.5,
        )


        # Step 5: Initialize generator
        self.generator = RAGGenerator(
            retriever=self.retriever,
            model_name=llm_model_name,
            temperature=0.2,
            top_p=0.9,
        )


    def query(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> str:
        """
        Generate answer for given question.

        Args:
            question: user query
            k: override number of retrieved chunks

        Returns:
            Answer string
        """

        answer = self.generator.generate(
            query=question,
            k=k,
        )

        return answer


    def query_with_sources(
        self,
        question: str,
        k: Optional[int] = None,
    ):
        """
        Generate answer with source metadata.
        Useful for frontend/API usage.
        """

        result = self.generator.generate_with_sources(
            query=question,
            k=k,
        )

        return result