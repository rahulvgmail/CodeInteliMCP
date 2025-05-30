"""Graph database service for Code Intelligence MCP."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

from ..core import BaseService
from ..config import GraphConfig
from .duckdb_graph import DuckDBGraph


class GraphService(BaseService):
    """Service for graph database operations."""
    
    def __init__(self, db_path: Path, config: GraphConfig):
        super().__init__(config=config)
        self.db_path = db_path
        self._graph: Optional[DuckDBGraph] = None
    
    async def _initialize(self) -> None:
        """Initialize graph database."""
        self._graph = DuckDBGraph(self.db_path)
        # Graph initializes its own schema
        self._logger.info(f"GraphService initialized with database at {self.db_path}")
    
    async def _shutdown(self) -> None:
        """Shutdown graph database."""
        if self._graph:
            self._graph.close()
            self._graph = None
            self._logger.info("GraphService shut down")
    
    # Delegate methods to DuckDBGraph
    async def add_symbol(self, **kwargs) -> int:
        """Add a symbol to the graph."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.add_symbol(**kwargs)
    
    async def add_relationship(self, **kwargs) -> int:
        """Add a relationship between symbols."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.add_relationship(**kwargs)
    
    async def find_symbol(
        self,
        name: Optional[str] = None,
        qualified_name: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find symbols by criteria."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.find_symbol(name, qualified_name, file_path)
    
    async def find_usages(self, symbol_id: int) -> List[Dict[str, Any]]:
        """Find all usages of a symbol."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.find_usages(symbol_id)
    
    async def find_dependencies(
        self,
        symbol_id: int,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find dependencies of a symbol."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.find_dependencies(symbol_id, relationship_types)
    
    async def find_circular_dependencies(self) -> List[Dict[str, Any]]:
        """Find circular dependencies in the codebase."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.find_circular_dependencies()
    
    async def get_file_symbols(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all symbols in a file."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.get_file_symbols(file_path)
    
    async def get_call_graph(
        self,
        root_symbol_id: int,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Get the call graph starting from a symbol."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.get_call_graph(root_symbol_id, max_depth)
    
    async def update_file(self, file_path: str, new_hash: str) -> None:
        """Update file information."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.update_file(file_path, new_hash)
    
    async def delete_file_symbols(self, file_path: str) -> None:
        """Delete all symbols from a file."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.delete_file_symbols(file_path)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.get_statistics()
    
    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """Execute a raw query (use with caution)."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.execute_query(query, params)
    
    async def get_symbol_details(self, symbol_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a symbol."""
        symbols = await self.find_symbol(qualified_name=None, file_path=None)
        for symbol in symbols:
            if symbol.get('id') == symbol_id:
                return symbol
        return None
    
    async def find_relationships(
        self,
        from_symbol_id: Optional[int] = None,
        to_symbol_id: Optional[int] = None,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find relationships matching criteria."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        
        # TODO: Implement in DuckDBGraph
        # For now, return empty list
        return []
    
    async def get_indexed_files(
        self,
        language: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get list of indexed files."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        
        # TODO: Implement in DuckDBGraph
        # For now, return empty list
        return []
    
    async def add_repository(self, name: str, path: str, description: Optional[str] = None) -> int:
        """Add a new repository to track."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.add_repository(name, path, description)
    
    async def list_repositories(self) -> List[Dict[str, Any]]:
        """List all tracked repositories."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.list_repositories()
    
    async def get_repository(self, name: str) -> Optional[Dict[str, Any]]:
        """Get repository by name."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        repositories = await self.list_repositories()
        return next((repo for repo in repositories if repo['name'] == name), None)
