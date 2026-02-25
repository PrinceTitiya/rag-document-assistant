from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingManager:
    """
    Manages embedding model initialization.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name


    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Initialize and return embedding model.

        Returns:
            HuggingFaceEmbeddings
        """

        embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name
        )

        return embedding_model