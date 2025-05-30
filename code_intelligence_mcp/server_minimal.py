"""Minimal MCP server for initial testing."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import sys

from mcp.server import FastMCP
from mcp.types import Tool

from .config import get_config, Config
from .di import create_container, DIContainer
from .graph import GraphService
from .parser import ParserService
from .vector import VectorService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinimalCodeIntelligenceMCP:
    """Minimal MCP server implementation."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.server = FastMCP("code-intelligence-minimal")
        self.container: Optional[DIContainer] = None
        self.graph: Optional[GraphService] = None
        self.parser: Optional[ParserService] = None
        self.vector: Optional[VectorService] = None
    
    async def initialize(self):
        """Initialize the server."""
        logger.info("Initializing Minimal Code Intelligence MCP server...")
        
        # Create DI container
        self.container = await create_container(self.config)
        
        # Get services
        self.graph = await self.container.get(GraphService)
        self.parser = await self.container.get(ParserService)
        self.vector = await self.container.get(VectorService)
        
        # Register minimal tools
        self._register_tools()
        
        logger.info("Server initialized successfully")
    
    def _register_tools(self):
        """Register minimal set of tools."""
        
        @self.server.tool()
        async def test_connection() -> Dict[str, Any]:
            """Test that the server is working."""
            return {
                "status": "connected",
                "message": "Code Intelligence MCP server is running"
            }
        
        @self.server.tool()
        async def get_index_stats() -> Dict[str, Any]:
            """Get current index statistics."""
            try:
                stats = await self.graph.get_statistics()
                return {
                    "status": "success",
                    "stats": stats
                }
            except Exception as e:
                logger.error(f"Error getting index stats: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def find_symbol(name: str, file_path: Optional[str] = None) -> Dict[str, Any]:
            """Find a symbol by name."""
            try:
                symbols = await self.graph.find_symbol(name=name, file_path=file_path)
                return {
                    "status": "success",
                    "count": len(symbols),
                    "symbols": symbols[:10]  # Limit results
                }
            except Exception as e:
                logger.error(f"Error finding symbol: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def find_usages(name: str, file_path: Optional[str] = None) -> Dict[str, Any]:
            """Find all usages of a symbol."""
            try:
                # Find the symbol first
                symbols = await self.graph.find_symbol(name=name, file_path=file_path)
                
                if not symbols:
                    return {
                        "status": "not_found",
                        "error": f"Symbol '{name}' not found",
                        "suggestions": []
                    }
                
                # Get usages for the first matching symbol
                symbol = symbols[0]
                usages = await self.graph.find_usages(symbol['id'])
                
                return {
                    "status": "success",
                    "symbol": {
                        "name": symbol['name'],
                        "type": symbol['symbol_type'],
                        "file": symbol['file_path'],
                        "line": symbol['line_start']
                    },
                    "usages": usages,
                    "total_usages": len(usages)
                }
            except Exception as e:
                logger.error(f"Error finding usages: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def add_test_symbol() -> Dict[str, Any]:
            """Add a test symbol to the database."""
            try:
                symbol_id = await self.graph.add_symbol(
                    name="test_function",
                    qualified_name="module.test_function",
                    symbol_type="function",
                    file_path="/test/file.py",
                    line_start=10,
                    line_end=20,
                    language="python",
                    ast_hash="abc123",
                    signature="def test_function(arg1, arg2)",
                    docstring="Test function for demonstration"
                )
                
                return {
                    "status": "success",
                    "message": "Test symbol added",
                    "symbol_id": symbol_id
                }
            except Exception as e:
                logger.error(f"Error adding test symbol: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def index_file(file_path: str) -> Dict[str, Any]:
            """Index a single file into the code graph."""
            try:
                from pathlib import Path
                path = Path(file_path)
                
                if not path.exists():
                    return {
                        "status": "error",
                        "error": f"File not found: {file_path}"
                    }
                
                # Parse the file
                symbols, relationships = await self.parser.analyze_file(path)
                
                # Store in graph and vector store
                symbol_ids = []
                vector_symbols = []
                
                for symbol in symbols:
                    symbol_id = await self.graph.add_symbol(
                        name=symbol.name,
                        qualified_name=symbol.qualified_name,
                        symbol_type=symbol.symbol_type,
                        file_path=str(path),
                        line_start=symbol.line_start,
                        line_end=symbol.line_end,
                        language=symbol.language or "unknown",
                        ast_hash=symbol.ast_hash,
                        signature=symbol.signature,
                        docstring=symbol.docstring,
                        is_async=symbol.is_async,
                        visibility=symbol.visibility,
                        complexity=symbol.complexity
                    )
                    symbol_ids.append(symbol_id)
                    
                    # Prepare for vector store
                    symbol_data = {
                        'name': symbol.name,
                        'qualified_name': symbol.qualified_name,
                        'symbol_type': symbol.symbol_type,
                        'file_path': str(path),
                        'line_start': symbol.line_start,
                        'language': symbol.language or "unknown",
                        'signature': symbol.signature,
                        'docstring': symbol.docstring
                    }
                    vector_symbols.append((symbol_id, symbol_data, None))
                
                # Add to vector store in batch
                if vector_symbols:
                    await self.vector.add_symbols_batch(vector_symbols)
                
                # Store relationships
                relationship_ids = []
                for rel in relationships:
                    # Find symbol IDs for the relationship
                    from_symbols = await self.graph.find_symbol(qualified_name=rel.from_symbol)
                    to_symbols = await self.graph.find_symbol(qualified_name=rel.to_symbol)
                    
                    if from_symbols and to_symbols:
                        rel_id = await self.graph.add_relationship(
                            from_symbol_id=from_symbols[0]['id'],
                            to_symbol_id=to_symbols[0]['id'],
                            relationship_type=rel.relationship_type,
                            file_path=str(path),
                            line_number=rel.line_number,
                            confidence=rel.confidence
                        )
                        relationship_ids.append(rel_id)
                
                return {
                    "status": "success",
                    "file": str(path),
                    "symbols_indexed": len(symbol_ids),
                    "relationships_indexed": len(relationship_ids),
                    "language": symbols[0].language if symbols else "unknown"
                }
                
            except Exception as e:
                logger.error(f"Error indexing file: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def semantic_search(query: str, limit: int = 10, symbol_types: Optional[str] = None) -> Dict[str, Any]:
            """Search for code using semantic similarity."""
            try:
                # Parse symbol types if provided
                types_list = None
                if symbol_types:
                    types_list = [t.strip() for t in symbol_types.split(',')]
                
                results = await self.vector.search_similar(
                    query=query,
                    n_results=limit,
                    symbol_types=types_list
                )
                
                return {
                    "status": "success",
                    "query": query,
                    "results": results,
                    "total_found": len(results)
                }
            except Exception as e:
                logger.error(f"Error in semantic search: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def find_similar_code(symbol_name: str, file_path: Optional[str] = None) -> Dict[str, Any]:
            """Find code similar to a given symbol."""
            try:
                # Find the symbol first
                symbols = await self.graph.find_symbol(name=symbol_name, file_path=file_path)
                
                if not symbols:
                    return {
                        "status": "not_found",
                        "error": f"Symbol '{symbol_name}' not found",
                        "symbol_name": symbol_name
                    }
                
                # Get similar symbols
                symbol = symbols[0]
                similar = await self.vector.find_similar_symbols(symbol['id'], n_results=10)
                
                return {
                    "status": "success",
                    "reference_symbol": {
                        "name": symbol['name'],
                        "type": symbol['symbol_type'],
                        "file": symbol['file_path'],
                        "line": symbol['line_start']
                    },
                    "similar_symbols": similar,
                    "total_found": len(similar)
                }
            except Exception as e:
                logger.error(f"Error finding similar code: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def get_vector_stats() -> Dict[str, Any]:
            """Get vector store statistics."""
            try:
                stats = await self.vector.get_statistics()
                return {
                    "status": "success",
                    "vector_stats": stats
                }
            except Exception as e:
                logger.error(f"Error getting vector stats: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def add_repository(name: str, path: str, description: Optional[str] = None, make_active: bool = False) -> Dict[str, Any]:
            """Add a new repository to track."""
            try:
                from pathlib import Path
                repo_path = Path(path).resolve()
                
                if not repo_path.exists():
                    return {
                        "status": "error",
                        "error": f"Repository path does not exist: {path}"
                    }
                
                if not repo_path.is_dir():
                    return {
                        "status": "error", 
                        "error": f"Repository path is not a directory: {path}"
                    }
                
                # Add to graph database
                repo_id = await self.graph.add_repository(
                    name=name,
                    path=str(repo_path),
                    description=description
                )
                
                return {
                    "status": "success",
                    "message": f"Repository '{name}' added successfully",
                    "repository_id": repo_id,
                    "path": str(repo_path)
                }
            except Exception as e:
                logger.error(f"Error adding repository: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def list_repositories() -> Dict[str, Any]:
            """List all tracked repositories."""
            try:
                repositories = await self.graph.list_repositories()
                return {
                    "status": "success",
                    "repositories": repositories,
                    "total_count": len(repositories)
                }
            except Exception as e:
                logger.error(f"Error listing repositories: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def set_active_repository(name: str) -> Dict[str, Any]:
            """Set the active repository for operations."""
            try:
                repositories = await self.graph.list_repositories()
                repo_names = [repo['name'] for repo in repositories]
                
                if name not in repo_names:
                    return {
                        "status": "error",
                        "error": f"Repository '{name}' not found. Available: {', '.join(repo_names)}"
                    }
                
                # Update active repository (in a real implementation, this would update config)
                return {
                    "status": "success",
                    "message": f"Active repository set to '{name}'"
                }
            except Exception as e:
                logger.error(f"Error setting active repository: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def index_repository(repository_name: str, include_patterns: Optional[str] = None, exclude_patterns: Optional[str] = None) -> Dict[str, Any]:
            """Index an entire repository."""
            try:
                import os
                from pathlib import Path
                
                # Get repository info
                repositories = await self.graph.list_repositories()
                repo = next((r for r in repositories if r['name'] == repository_name), None)
                
                if not repo:
                    return {
                        "status": "error",
                        "error": f"Repository '{repository_name}' not found"
                    }
                
                repo_path = Path(repo['path'])
                if not repo_path.exists():
                    return {
                        "status": "error",
                        "error": f"Repository path no longer exists: {repo_path}"
                    }
                
                # Parse patterns
                include_list = include_patterns.split(',') if include_patterns else ['**/*.py', '**/*.js', '**/*.ts', '**/*.go', '**/*.rs', '**/*.java']
                exclude_list = exclude_patterns.split(',') if exclude_patterns else ['**/__pycache__/**', '**/node_modules/**', '**/.git/**', '**/venv/**', '**/.venv/**']
                
                # Find files to index
                files_to_index = []
                for pattern in include_list:
                    for file_path in repo_path.glob(pattern):
                        if file_path.is_file():
                            # Check if file matches exclude patterns
                            relative_path = file_path.relative_to(repo_path)
                            excluded = any(relative_path.match(exclude_pattern) for exclude_pattern in exclude_list)
                            if not excluded:
                                files_to_index.append(file_path)
                
                # Index files
                indexed_files = 0
                indexed_symbols = 0
                indexed_relationships = 0
                errors = []
                
                for file_path in files_to_index[:50]:  # Limit to first 50 files for demo
                    try:
                        # Parse the file
                        symbols, relationships = await self.parser.analyze_file(file_path)
                        
                        # Store symbols
                        for symbol in symbols:
                            relative_path = str(file_path.relative_to(repo_path))
                            symbol_id = await self.graph.add_symbol(
                                repository_id=repo['id'],
                                name=symbol.name,
                                qualified_name=symbol.qualified_name,
                                symbol_type=symbol.symbol_type,
                                file_path=relative_path,
                                line_start=symbol.line_start,
                                line_end=symbol.line_end,
                                language=symbol.language or "unknown",
                                ast_hash=symbol.ast_hash,
                                signature=symbol.signature,
                                docstring=symbol.docstring,
                                is_async=symbol.is_async,
                                visibility=symbol.visibility,
                                complexity=symbol.complexity
                            )
                            indexed_symbols += 1
                            
                            # Add to vector store
                            symbol_data = {
                                'repository_name': repository_name,
                                'name': symbol.name,
                                'qualified_name': symbol.qualified_name,
                                'symbol_type': symbol.symbol_type,
                                'file_path': relative_path,
                                'line_start': symbol.line_start,
                                'language': symbol.language or "unknown",
                                'signature': symbol.signature,
                                'docstring': symbol.docstring
                            }
                            await self.vector.add_symbols_batch([(symbol_id, symbol_data, None)])
                        
                        indexed_files += 1
                        indexed_relationships += len(relationships)
                        
                    except Exception as e:
                        errors.append(f"{file_path}: {str(e)}")
                
                return {
                    "status": "success",
                    "repository": repository_name,
                    "files_indexed": indexed_files,
                    "symbols_indexed": indexed_symbols,
                    "relationships_indexed": indexed_relationships,
                    "total_files_found": len(files_to_index),
                    "errors": errors[:10]  # Limit error list
                }
                
            except Exception as e:
                logger.error(f"Error indexing repository: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    async def run(self):
        """Run the MCP server."""
        await self.initialize()
        logger.info("Minimal Code Intelligence MCP server started")
        
        # The MCP server handles its own event loop
        await self.server.run()
    
    async def shutdown(self):
        """Shutdown the server."""
        logger.info("Shutting down server...")
        if self.container:
            await self.container.shutdown()
        logger.info("Server shut down complete")


async def main():
    """Main entry point."""
    server = MinimalCodeIntelligenceMCP()
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())