"""
Core RAG utilities for Academic RAG Assistant - Simplified Version
"""

import os
import tempfile
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_community.llms import Ollama

from config import config

class DocumentProcessor:
    """Document processing with basic strategies"""
    
    def __init__(self):
        self.processed_docs = []
        
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Load document with appropriate loader"""
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_type == 'docx':
            # For simplicity, we'll use TextLoader for docx (requires python-docx)
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        try:
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'source': Path(file_path).name,
                    'file_type': file_type,
                    'processed_at': datetime.now().isoformat(),
                    'doc_id': hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
                })
            
            print(f"Loaded {len(docs)} pages from {file_path}")
            return docs
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Document], 
                       chunk_size: int = config.DEFAULT_CHUNK_SIZE,
                       chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP) -> List[Document]:
        """Chunk documents"""
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        
        chunks = splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': f"{chunk.metadata.get('doc_id', 'unknown')}_{i}",
                'chunk_index': i,
                'total_chunks': len(chunks),
            })
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def extract_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract and analyze document metadata"""
        metadata = {
            'total_documents': len(documents),
            'total_pages': sum([doc.metadata.get('page', 1) for doc in documents]),
            'total_chars': sum(len(doc.page_content) for doc in documents),
            'file_types': {},
            'sources': set(),
        }
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            metadata['file_types'][file_type] = metadata['file_types'].get(file_type, 0) + 1
            metadata['sources'].add(doc.metadata.get('source', 'unknown'))
        
        metadata['sources'] = list(metadata['sources'])
        return metadata


class VectorStoreManager:
    """Vector store management"""
    
    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_store = None
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        """Initialize embedding model"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"Initialized embeddings with model: {self.embedding_model}")
    
    def create_vector_store(self, chunks: List[Document], 
                          collection_name: str = "academic_docs",
                          persist: bool = True) -> Chroma:
        """Create vector store"""
        
        # Ensure persist directory exists
        if persist:
            os.makedirs(config.VECTOR_DB_PERSIST_DIR, exist_ok=True)
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=config.VECTOR_DB_PERSIST_DIR if persist else None,
        )
        
        if persist:
            self.vector_store.persist()
        
        print(f"Created vector store with {len(chunks)} chunks")
        return self.vector_store
    
    def get_retriever(self, k: int = config.SIMILARITY_TOP_K):
        """Get retriever"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = {"k": k}
        
        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        print(f"Created retriever with k={k}")
        return retriever
    
    def search_similar(self, query: str, k: int = 5):
        """Search for similar documents"""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.vector_store:
            return {}
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'total_chunks': count,
                'collection_name': collection.name,
                'embedding_dimension': 384,
                'distance_function': 'cosine'
            }
        except:
            return {}


class LLMManager:
    """Manage LLM integration"""
    
    def __init__(self):
        self.llm = None
        
    def initialize_llm(self, model: str = config.LLM_MODEL, **kwargs):
        """Initialize LLM"""
        
        llm_config = {
            'temperature': config.LLM_TEMPERATURE,
            **kwargs
        }
        
        try:
            self.llm = Ollama(
                model=model,
                **llm_config
            )
            print(f"Initialized Ollama LLM with model: {model}")
            return self.llm
        except Exception as e:
            print(f"Failed to initialize LLM: {str(e)}")
            print("Make sure Ollama is running: `ollama serve`")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM"""
        if not self.llm:
            return {}
        
        return {
            'model': config.LLM_MODEL,
            'temperature': config.LLM_TEMPERATURE
        }


class PromptManager:
    """Manage prompts for academic tasks"""
    
    @staticmethod
    def get_prompt_template(task: str = "qa") -> str:
        """Get prompt template for academic task"""
        
        prompts = {
            "qa": """
            You are Academic Assistant, an AI specialized in helping with academic research.
            
            CONTEXT FROM DOCUMENTS:
            {context}
            
            QUESTION: {question}
            
            INSTRUCTIONS:
            1. Answer based on the provided context only.
            2. If information is not in context, say: "I cannot find this information in the provided documents."
            3. Provide clear, academic-style answers.
            4. Structure your answer properly.
            5. Cite sources using [Source: Document Name].
            
            ANSWER:
            """,
            
            "summarize": """
            You are Academic Assistant, tasked with summarizing academic content.
            
            TEXT TO SUMMARIZE:
            {context}
            
            INSTRUCTIONS:
            1. Create a concise academic summary.
            2. Include main points and conclusions.
            3. Format as an academic abstract.
            
            SUMMARY:
            """,
            
            "explain": """
            You are Academic Assistant, explaining concepts to students.
            
            CONCEPT: {question}
            
            RELEVANT CONTEXT:
            {context}
            
            INSTRUCTIONS:
            1. Explain the concept clearly.
            2. Use simple language.
            3. Provide examples if available.
            
            EXPLANATION:
            """
        }
        
        return prompts.get(task, prompts["qa"])