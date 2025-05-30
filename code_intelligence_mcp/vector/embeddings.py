"""Embedding service for code representations."""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ..core import BaseService
from ..config import VectorConfig

logger = logging.getLogger(__name__)


class EmbeddingService(BaseService):
    """Service for generating code embeddings."""
    
    def __init__(self, config: VectorConfig):
        super().__init__()
        self.config = config
        self.model: Optional[SentenceTransformer] = None
    
    async def _initialize(self) -> None:
        """Initialize embedding model."""
        self._logger.info(f"Loading embedding model: {self.config.embedding_model}")
        try:
            self.model = SentenceTransformer(self.config.embedding_model)
            self._logger.info("Embedding model loaded successfully")
        except Exception as e:
            self._logger.error(f"Failed to load embedding model: {e}")
            # Use a smaller model as fallback
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._logger.info("Using fallback model: all-MiniLM-L6-v2")
    
    async def _shutdown(self) -> None:
        """Shutdown embedding service."""
        self.model = None
    
    def embed_code(self, code_snippets: List[str]) -> np.ndarray:
        """Generate embeddings for code snippets."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        # Prepare code for embedding
        prepared_snippets = [self._prepare_code_for_embedding(code) for code in code_snippets]
        
        # Generate embeddings
        embeddings = self.model.encode(prepared_snippets)
        return embeddings
    
    def embed_single(self, code: str) -> List[float]:
        """Generate embedding for a single code snippet."""
        embeddings = self.embed_code([code])
        return embeddings[0].tolist()
    
    def _prepare_code_for_embedding(self, code: str) -> str:
        """Prepare code for embedding by extracting semantic information."""
        # Simple preparation - can be enhanced with AST info
        lines = code.strip().split('\n')
        
        # Remove empty lines and excessive whitespace
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                cleaned_lines.append(line)
        
        # Limit to reasonable size for embedding
        if len(cleaned_lines) > 20:
            cleaned_lines = cleaned_lines[:20]
        
        return ' '.join(cleaned_lines)
    
    def create_symbol_embedding_text(self, symbol: Dict[str, Any]) -> str:
        """Create text representation of a symbol for embedding."""
        parts = []
        
        # Add type and name
        parts.append(f"{symbol.get('symbol_type', 'unknown')} {symbol.get('name', '')}")
        
        # Add signature if available
        if symbol.get('signature'):
            parts.append(symbol['signature'])
        
        # Add docstring if available
        if symbol.get('docstring'):
            # Limit docstring length
            docstring = symbol['docstring'][:200]
            parts.append(docstring)
        
        # Add qualified name for context
        if symbol.get('qualified_name'):
            parts.append(f"qualified: {symbol['qualified_name']}")
        
        return ' '.join(parts)