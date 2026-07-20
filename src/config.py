# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data/raw"
VECTORSTORE_DIR = "data/vectorstore"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
