# src/retriever.py

import re
from typing import List, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.query_expander import QueryExpander


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class RAGRetriever:
    """
    Hybrid retriever: combines dense (embedding) similarity search with BM25
    keyword search, optionally expands the question into alternate phrasings
    first, fuses every resulting ranked list with Reciprocal Rank Fusion
    (RRF), then re-ranks the fused candidates with a cross-encoder for final
    precision. This is the standard "multi-query + hybrid search + rerank"
    architecture used in production RAG systems.

    Why each piece is needed (each closes a gap the others can't):
    - Dense embeddings match on *meaning* but can miss a passage phrased
      very differently from the question.
    - BM25 catches literal keyword overlap that embeddings sometimes rank
      too low — but only if the question's own words actually appear in
      the passage.
    - Neither helps when the answer-bearing passage uses neither similar
      wording nor the same keywords as the question (e.g. a passage that
      explains "proof of work" in detail but never says "consensus
      mechanism"). Query expansion (see src/query_expander.py) rewrites the
      question into more specific/technical variants before retrieval,
      which is what actually bridges that gap.
    - The cross-encoder re-ranks the combined candidate pool directly
      against the *original* question for the final ordering, which is
      more precise than any single retrieval signal alone.
    """

    def __init__(
        self,
        vectorstore,
        chunks: List[Document],
        k: int = 5,
        fetch_k: int = 20,
        rrf_k: int = 60,
        use_reranker: bool = True,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm=None,
        num_query_variants: int = 3,
    ):
        """
        Args:
            vectorstore: Chroma vectorstore instance (dense/embedding leg)
            chunks: all indexed chunks, used to build the BM25 keyword index
            k: number of results to return
            fetch_k: number of candidates each retrieval leg fetches before fusion
            rrf_k: Reciprocal Rank Fusion constant (60 is the standard default)
            use_reranker: whether to apply a cross-encoder re-ranking pass
            reranker_model_name: cross-encoder model used for re-ranking
            llm: shared chat model used to generate query variants (see
                 src/llm.py). If None, query expansion is skipped and
                 retrieval falls back to hybrid search on the original
                 question only.
            num_query_variants: how many alternate phrasings to generate per query
        """

        self.vectorstore = vectorstore
        self.chunks = chunks
        self.k = k
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

        self._bm25 = BM25Okapi(
            [_tokenize(doc.page_content) for doc in chunks]
        )

        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model_name)

        self.query_expander = None
        if llm is not None:
            self.query_expander = QueryExpander(llm, num_variants=num_query_variants)


    def _dense_search(self, query: str, fetch_k: int) -> List[Document]:
        return self.vectorstore.similarity_search(query=query, k=fetch_k)


    def _bm25_search(self, query: str, fetch_k: int) -> List[Document]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:fetch_k]
        return [self.chunks[i] for i in ranked_indices if scores[i] > 0]


    @staticmethod
    def _doc_key(doc: Document):
        return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)


    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Document]],
    ) -> List[Document]:
        scores = {}
        doc_by_key = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                key = self._doc_key(doc)
                doc_by_key[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)

        return [doc_by_key[key] for key in ordered_keys]


    def _rerank(self, query: str, candidates: List[Document], k: int) -> List[Document]:
        if not candidates:
            return candidates

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        reranked = [
            doc
            for _, doc in sorted(
                zip(scores, candidates), key=lambda pair: pair[0], reverse=True
            )
        ]

        return reranked[:k]


    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents via hybrid search (dense + BM25 + RRF),
        optionally re-ranked by a cross-encoder.

        Args:
            query: user query
            k: override default number of results

        Returns:
            List of relevant Documents
        """

        if not query:
            raise ValueError("Query cannot be empty")

        k = k or self.k

        queries = [query]
        if self.query_expander is not None:
            queries.extend(self.query_expander.expand(query))

        ranked_lists = []
        for q in queries:
            ranked_lists.append(self._dense_search(q, self.fetch_k))
            ranked_lists.append(self._bm25_search(q, self.fetch_k))

        fused = self._reciprocal_rank_fusion(ranked_lists)

        if self.reranker is not None:
            # Re-rank a slightly wider pool than k so the cross-encoder can
            # actually reorder candidates rather than just truncating them.
            # Always rerank against the *original* question, not the
            # expanded variants, since that's what the answer must address.
            pool_size = max(self.fetch_k, k * 3)
            return self._rerank(query, fused[:pool_size], k)

        return fused[:k]


    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ):
        """
        Retrieve documents with dense similarity scores.
        Useful for debugging and evaluation (does not include BM25/rerank
        signal — for inspecting the dense leg in isolation).
        """

        k = k or self.k

        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
        )
