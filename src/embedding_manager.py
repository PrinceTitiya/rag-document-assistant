import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

class EmbeddingManager:
    """
    Manages embedding model initialization.

    Uses the Hugging Face hosted Inference API (not a local download): the
    model runs on Hugging Face's servers and is called over the network,
    authenticated with HUGGING_FACE_TOKEN. This keeps the pipeline free of
    large local model downloads while still allowing a stronger embedding
    model than a small local one.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        self.model_name = model_name


    def get_embedding_model(self) -> HuggingFaceEndpointEmbeddings:
        """
        Initialize and return embedding model, backed by the Hugging Face
        Inference API.

        Returns:
            HuggingFaceEndpointEmbeddings
        """

        token = os.getenv("HUGGING_FACE_TOKEN")

        if not token:
            raise ValueError(
                "HUGGING_FACE_TOKEN is not set. Add HUGGING_FACE_TOKEN=your_token_here "
                "to a .env file in the project root (see README.md)."
            )

        embedding_model = HuggingFaceEndpointEmbeddings(
            model=self.model_name,
            huggingfacehub_api_token=token,
        )

        return embedding_model
