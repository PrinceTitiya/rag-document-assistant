# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data/raw"
VECTORSTORE_DIR = "data/vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")