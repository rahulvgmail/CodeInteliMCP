"""Vector store service using ChromaDB."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
import uuid

from ..core import BaseService
from ..config import VectorConfig
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VectorService(BaseService):
    """Service for vector-based semantic search using ChromaDB."""
    
    def __init__(self, config: VectorConfig, embedding_service: EmbeddingService):
        super().__init__()
        self.config = config
        self.embedding_service = embedding_service
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
    
    async def _initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        self._logger.info("Initializing VectorService with ChromaDB")
        
        # Ensure data directory exists
        self.config.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.config.chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name="code_symbols")
            self._logger.info("Using existing ChromaDB collection")
        except Exception as e:
            self._logger.info(f"Creating new ChromaDB collection: {e}")
            self.collection = self.client.create_collection(
                name="code_symbols",
                metadata={"description": "Code symbols for semantic search"}
            )
            self._logger.info("Created new ChromaDB collection")
    
    async def _shutdown(self) -> None:
        """Shutdown vector service."""
        self.collection = None
        self.client = None
    
    async def add_symbol(
        self,
        symbol_id: int,
        symbol_data: Dict[str, Any],
        code_content: Optional[str] = None
    ) -> None:
        """Add a symbol to the vector store."""
        # Create embedding text
        embedding_text = self.embedding_service.create_symbol_embedding_text(symbol_data)
        
        # If code content provided, include it
        if code_content:
            embedding_text = f"{embedding_text} {code_content}"
        
        # Generate embedding
        embedding = self.embedding_service.embed_single(embedding_text)
        
        # Prepare metadata
        metadata = {
            "symbol_id": symbol_id,
            "name": symbol_data.get("name", ""),
            "qualified_name": symbol_data.get("qualified_name", ""),
            "symbol_type": symbol_data.get("symbol_type", ""),
            "file_path": symbol_data.get("file_path", ""),
            "line_start": symbol_data.get("line_start", 0),
            "language": symbol_data.get("language", "")
        }
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[embedding_text],
            ids=[f"symbol_{symbol_id}"]
        )
    
    async def add_symbols_batch(
        self,
        symbols: List[Tuple[int, Dict[str, Any], Optional[str]]]
    ) -> None:
        """Add multiple symbols to the vector store in batch."""
        if not symbols:
            return
        
        embeddings = []
        metadatas = []
        documents = []
        ids = []
        
        for symbol_id, symbol_data, code_content in symbols:
            # Create embedding text
            embedding_text = self.embedding_service.create_symbol_embedding_text(symbol_data)
            if code_content:
                embedding_text = f"{embedding_text} {code_content}"
            
            documents.append(embedding_text)
            
            # Prepare metadata
            metadata = {
                "symbol_id": symbol_id,
                "name": symbol_data.get("name", ""),
                "qualified_name": symbol_data.get("qualified_name", ""),
                "symbol_type": symbol_data.get("symbol_type", ""),
                "file_path": symbol_data.get("file_path", ""),
                "line_start": symbol_data.get("line_start", 0),
                "language": symbol_data.get("language", "")
            }
            metadatas.append(metadata)
            ids.append(f"symbol_{symbol_id}")
        
        # Generate embeddings in batch
        embedding_array = self.embedding_service.embed_code(documents)
        embeddings = embedding_array.tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids
        )
    
    async def search_similar(
        self,
        query: str,
        n_results: int = 10,
        symbol_types: Optional[List[str]] = None,
        file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar code symbols."""
        # Generate query embedding
        query_embedding = self.embedding_service.embed_single(query)
        
        # Build where clause for filtering
        where = {}
        if symbol_types:
            where["symbol_type"] = {"$in": symbol_types}
        if file_path:
            where["file_path"] = file_path
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where if where else None
        )
        
        # Format results
        formatted_results = []
        if results and results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                result = {
                    **metadata,
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'document': results['documents'][0][i] if 'documents' in results else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    async def find_similar_symbols(
        self,
        symbol_id: int,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Find symbols similar to a given symbol."""
        # Get the symbol's embedding
        result = self.collection.get(
            ids=[f"symbol_{symbol_id}"],
            include=['embeddings', 'metadatas', 'documents']
        )
        
        if not result or not result['embeddings']:
            return []
        
        embedding = result['embeddings'][0]
        
        # Search for similar, excluding the original
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results + 1,  # Get extra to exclude self
            where={"symbol_id": {"$ne": symbol_id}}
        )
        
        # Format results
        formatted_results = []
        if results and results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                if metadata['symbol_id'] != symbol_id:  # Double-check exclusion
                    result = {
                        **metadata,
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'document': results['documents'][0][i] if 'documents' in results else None
                    }
                    formatted_results.append(result)
        
        return formatted_results[:n_results]
    
    async def delete_symbol(self, symbol_id: int) -> None:
        """Delete a symbol from the vector store."""
        self.collection.delete(ids=[f"symbol_{symbol_id}"])
    
    async def delete_file_symbols(self, file_path: str) -> None:
        """Delete all symbols from a file."""
        # Get all symbols from the file
        results = self.collection.get(
            where={"file_path": file_path},
            include=['ids']
        )
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        # Get collection count
        count = self.collection.count()
        
        # Get sample of symbol types
        sample = self.collection.get(
            limit=1000,
            include=['metadatas']
        )
        
        symbol_type_counts = {}
        if sample and sample['metadatas']:
            for metadata in sample['metadatas']:
                symbol_type = metadata.get('symbol_type', 'unknown')
                symbol_type_counts[symbol_type] = symbol_type_counts.get(symbol_type, 0) + 1
        
        return {
            'total_vectors': count,
            'symbol_types': symbol_type_counts,
            'collection_name': 'code_symbols',
            'embedding_model': self.config.embedding_model
        }
    
    async def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection("code_symbols")
        self.collection = self.client.create_collection(
            name="code_symbols",
            metadata={"description": "Code symbols for semantic search"}
        )