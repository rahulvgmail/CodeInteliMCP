"""Main MCP server for code intelligence."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

from mcp.server import Server
from mcp.types import (
    Tool, Resource, TextContent, 
    ListResourcesResult, ReadResourceResult
)

from .config import get_config, Config
from .graph import DuckDBGraph
from .tools import (
    SearchTools, AnalysisTools, IndexTools, VizTools
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeIntelligenceMCP:
    """Main MCP server for code intelligence."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the MCP server."""
        self.config = config or get_config()
        self.server = Server(self.config.mcp_server_name)
        
        # Initialize components
        self.graph = DuckDBGraph()
        
        # Tool instances
        self.search_tools = SearchTools(self.graph)
        self.analysis_tools = AnalysisTools(self.graph)
        self.index_tools = IndexTools(self.graph)
        self.viz_tools = VizTools(self.graph)
        
        # Register handlers
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        """Register all MCP tools."""
        # Search tools
        @self.server.tool()
        async def find_usages(
            name: str,
            file_path: Optional[str] = None
        ) -> Dict[str, Any]:
            """Find all usages of a function, class, or variable.
            
            Args:
                name: Name of the symbol to find usages for
                file_path: Optional file path to narrow search
            
            Returns:
                List of locations where the symbol is used
            """
            return await self.search_tools.find_usages(name, file_path)
        
        @self.server.tool()
        async def find_similar_code(
            code_snippet: str,
            limit: int = 10
        ) -> List[Dict[str, Any]]:
            """Find code similar to the given snippet.
            
            Args:
                code_snippet: Code to find similar patterns for
                limit: Maximum number of results
            
            Returns:
                List of similar code segments with similarity scores
            """
            return await self.search_tools.find_similar_code(code_snippet, limit)
        
        @self.server.tool()
        async def find_dependencies(
            name: str,
            depth: int = 3
        ) -> Dict[str, Any]:
            """Find all dependencies of a symbol.
            
            Args:
                name: Name of the symbol
                depth: Maximum depth to traverse
            
            Returns:
                Dependency graph with symbols and relationships
            """
            return await self.analysis_tools.find_dependencies(name, depth)
        
        @self.server.tool()
        async def analyze_impact(
            name: str,
            change_type: str = "modify"
        ) -> Dict[str, Any]:
            """Analyze the impact of changing a symbol.
            
            Args:
                name: Name of the symbol to analyze
                change_type: Type of change (modify, delete, rename)
            
            Returns:
                Impact analysis including affected files and symbols
            """
            return await self.analysis_tools.analyze_impact(name, change_type)
        
        @self.server.tool()
        async def update_index(
            paths: Optional[List[str]] = None,
            full_rebuild: bool = False
        ) -> Dict[str, Any]:
            """Update the code index.
            
            Args:
                paths: Specific paths to update (None for smart update)
                full_rebuild: Force complete rebuild
            
            Returns:
                Update statistics
            """
            return await self.index_tools.update_index(paths, full_rebuild)
        
        @self.server.tool()
        async def get_index_stats() -> Dict[str, Any]:
            """Get current index statistics.
            
            Returns:
                Statistics about indexed symbols, relationships, etc.
            """
            return await self.graph.get_statistics()
        
        @self.server.tool()
        async def find_circular_dependencies() -> List[Dict[str, Any]]:
            """Find circular dependencies in the codebase.
            
            Returns:
                List of circular dependency chains
            """
            return await self.graph.find_circular_dependencies()
        
        @self.server.tool()
        async def get_call_graph(
            name: str,
            max_depth: int = 3
        ) -> Dict[str, Any]:
            """Get the call graph for a function.
            
            Args:
                name: Function name
                max_depth: Maximum depth to traverse
            
            Returns:
                Call graph with nodes and edges
            """
            # Find the symbol first
            symbols = await self.graph.find_symbol(name=name)
            if not symbols:
                return {"error": f"Symbol '{name}' not found"}
            
            return await self.graph.get_call_graph(symbols[0]['id'], max_depth)
        
        @self.server.tool()
        async def search_by_pattern(
            pattern: str,
            language: Optional[str] = None
        ) -> List[Dict[str, Any]]:
            """Search code using tree-sitter patterns.
            
            Args:
                pattern: Tree-sitter query pattern
                language: Optional language filter
            
            Returns:
                Matching code segments
            """
            return await self.search_tools.search_by_pattern(pattern, language)
    
    def _register_resources(self):
        """Register MCP resources."""
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available resources."""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri="index://statistics",
                        name="Index Statistics",
                        mimeType="application/json",
                        description="Current code index statistics"
                    ),
                    Resource(
                        uri="index://config",
                        name="Configuration",
                        mimeType="application/json",
                        description="Current server configuration"
                    )
                ]
            )
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read a resource."""
            if uri == "index://statistics":
                stats = await self.graph.get_statistics()
                return ReadResourceResult(
                    contents=[TextContent(
                        uri=uri,
                        mimeType="application/json",
                        text=str(stats)
                    )]
                )
            elif uri == "index://config":
                return ReadResourceResult(
                    contents=[TextContent(
                        uri=uri,
                        mimeType="application/json",
                        text=self.config.model_dump_json(indent=2)
                    )]
                )
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def run(self):
        """Run the MCP server."""
        logger.info(f"Starting {self.config.mcp_server_name} MCP server...")
        
        # Initialize index if needed
        stats = await self.graph.get_statistics()
        if stats['total_symbols'] == 0:
            logger.info("No symbols in index. Run 'update_index' to index your codebase.")
        else:
            logger.info(f"Loaded index with {stats['total_symbols']} symbols")
        
        # Run the server
        await self.server.run()
    
    def close(self):
        """Clean up resources."""
        self.graph.close()


def main():
    """Main entry point."""
    # Check if we should run with custom config
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
        config = Config(project_root=project_root)
    else:
        config = get_config()
    
    # Create and run server
    server = CodeIntelligenceMCP(config)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.close()


if __name__ == "__main__":
    main()
