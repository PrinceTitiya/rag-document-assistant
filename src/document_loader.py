# src/document_loader.py

from pathlib import Path
from typing import List, Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentLoader:

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def _is_real_table(
        self,
        table: List[List[Optional[str]]],
        min_nonblank_rows: int = 2,
        min_cols: int = 2,
        max_cell_len: int = 600,
        min_distinct_cols: int = 2,
    ) -> bool:
        """
        pdfplumber's table detector fires on a lot of ordinary paragraph text
        (justified spacing, bullet lists) as well as genuine tables. This
        filters out those false positives:
        - needs at least 2 non-blank rows and 2 columns (a real grid)
        - rejects "tables" that are really just one giant paragraph crammed
          into a single cell
        - requires content to actually span multiple columns, not just one
          column of list items with empty padding cells
        """

        non_blank_rows = [r for r in table if any((c or "").strip() for c in r)]

        if len(non_blank_rows) < min_nonblank_rows:
            return False

        num_cols = max(len(r) for r in table)

        if num_cols < min_cols:
            return False

        distinct_cols_with_content = set()

        for r in table:
            for idx, c in enumerate(r):
                if c and c.strip():
                    if len(c) > max_cell_len:
                        return False
                    distinct_cols_with_content.add(idx)

        return len(distinct_cols_with_content) >= min_distinct_cols


    def _table_to_markdown(
        self,
        table: List[List[Optional[str]]],
    ) -> str:
        """
        Convert a pdfplumber table (list of rows) into a Markdown table string,
        so row/column relationships survive being flattened into a single
        chunk of text (plain text extraction loses this structure entirely).
        """

        rows = [
            [(cell or "").strip().replace("\n", " ") for cell in row]
            for row in table
        ]
        rows = [row for row in rows if any(row)]

        if not rows:
            return ""

        num_cols = max(len(row) for row in rows)
        rows = [row + [""] * (num_cols - len(row)) for row in rows]

        header, *body = rows

        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join([" --- "] * num_cols) + "|",
        ]

        for row in body:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


    def load_documents(self) -> List[Document]:
        """
        Load PDF documents from directory, page by page, using pdfplumber.

        Any genuine tables on a page are additionally rendered as a Markdown
        table and appended to that page's text, so the LLM sees row/column
        structure explicitly instead of a flattened, ambiguous stream of text.

        Returns:
            List[Document]: List of page-level Document objects
        """

        documents = []
        pdf_paths = sorted(Path(self.data_dir).glob("*.pdf"))

        for pdf_path in pdf_paths:

            with pdfplumber.open(pdf_path) as pdf:

                for page_number, page in enumerate(pdf.pages):

                    text = page.extract_text() or ""

                    tables = [
                        t for t in page.extract_tables()
                        if self._is_real_table(t)
                    ]

                    markdown_tables = [
                        md for md in (self._table_to_markdown(t) for t in tables)
                        if md
                    ]

                    if markdown_tables:
                        text = (
                            text
                            + "\n\nStructured version of the table(s) on this page:\n\n"
                            + "\n\n".join(markdown_tables)
                        )

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_path),
                                "page": page_number,
                            },
                        )
                    )

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
