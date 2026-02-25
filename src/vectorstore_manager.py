import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


class VectorStoreManager:
    """
    Manages creation and loading of Chroma vectorstore.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model,
    ):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model


    def vectorstore_exists(self) -> bool:
        """
        Check if vectorstore already exists on disk.
        """

        return (
            os.path.exists(self.persist_dir)
            and os.path.isdir(self.persist_dir)
            and len(os.listdir(self.persist_dir)) > 0
        )


    def create_vectorstore(
        self,
        chunks: List[Document],
    ) -> Chroma:
        """
        Create and persist new vectorstore.

        Args:
            chunks: List of chunked documents

        Returns:
            Chroma vectorstore
        """

        if not chunks:
            raise ValueError("Chunks list is empty. Cannot create vectorstore.")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
        )

        return vectorstore


    def load_vectorstore(self) -> Chroma:
        """
        Load existing vectorstore from disk.

        Returns:
            Chroma vectorstore
        """

        if not self.vectorstore_exists():
            raise FileNotFoundError(
                f"No vectorstore found at {self.persist_dir}"
            )

        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_model,
        )

        return vectorstore


    def get_vectorstore(
        self,
        chunks: Optional[List[Document]] = None,
    ) -> Chroma:
        """
        Load vectorstore if exists, otherwise create new one.

        Args:
            chunks: Required if vectorstore does not exist

        Returns:
            Chroma vectorstore
        """

        if self.vectorstore_exists():

            return self.load_vectorstore()

        if chunks is None:

            raise ValueError(
                "Vectorstore does not exist. Chunks required to create one."
            )

        return self.create_vectorstore(chunks)