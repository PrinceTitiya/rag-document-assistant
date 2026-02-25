# src/document_loader.py

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


class DocumentLoader:
    
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def load_documents(self) -> List[Document]:
        """
        Load PDF documents from directory.

        Returns:
            List[Document]: List of page-level Document objects
        """

        loader = DirectoryLoader(
            path=self.data_dir,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        )

        documents = loader.load()

        return documents


    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents (List[Document])

        Returns:
            List[Document]: chunked documents
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        return chunks


    def load_and_split(self) -> List[Document]:
        """
        Convenience function: load and split documents.

        Returns:
            List[Document]: chunked documents
        """

        documents = self.load_documents()
        chunks = self.split_documents(documents)

        return chunks