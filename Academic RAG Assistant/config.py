"""
Configuration settings for Academic RAG Assistant
"""

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    # Application Settings
    APP_NAME: str = "Academic RAG Assistant"
    APP_VERSION: str = "1.0.0"
    
    # Document Processing
    SUPPORTED_EXTENSIONS: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".txt"])
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES: int = 50
    
    # Text Chunking
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    
    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Vector Database
    VECTOR_DB_PERSIST_DIR: str = "./vector_store"
    SIMILARITY_TOP_K: int = 4
    
    # LLM Settings
    LLM_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.3
    
    # RAG Settings
    RETRIEVAL_SCORE_THRESHOLD: float = 0.7

config = AppConfig()