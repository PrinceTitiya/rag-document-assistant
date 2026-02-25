# src/retriever.py

from typing import List, Optional
from langchain_core.documents import Document


class RAGRetriever:
    """
    Handles retrieval of relevant documents from vectorstore.
    Supports MMR (Max Marginal Relevance) retrieval.
    """

    def __init__(
        self,
        vectorstore,
        search_type: str = "mmr",
        k: int = 5,
        fetch_k: int = 1,
        lambda_mult: float = 0.5,
    ):
        """
        Args:
            vectorstore: Chroma vectorstore instance
            search_type: "mmr" or "similarity"
            k: number of results to return
            fetch_k: number of candidates to fetch before MMR filtering
            lambda_mult: diversity vs relevance balance (MMR only)
        """

        self.vectorstore = vectorstore
        self.search_type = search_type
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult


    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents.

        Args:
            query: user query
            k: override default number of results

        Returns:
            List of relevant Documents
        """

        if not query:
            raise ValueError("Query cannot be empty")

        k = k or self.k

        if self.search_type == "mmr":

            results = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
            )

        elif self.search_type == "similarity":

            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
            )

        else:

            raise ValueError(
                f"Unsupported search_type: {self.search_type}"
            )

        return results


    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ):
        """
        Retrieve documents with similarity scores.
        Useful for debugging and evaluation.
        """

        k = k or self.k

        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
        )