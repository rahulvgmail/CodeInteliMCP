# 🚀 Quick Start Implementation Guide

## Overview

This guide provides step-by-step instructions for Claude Code to implement the Code Intelligence MCP server. Follow these steps in order.

## Phase 1: Core Infrastructure (Day 1)

### Step 1: Create Base Service Classes

```python
# code_intelligence_mcp/core/__init__.py
from .service import BaseService, CacheableService, ServiceProtocol
from .errors import ServiceError, ConfigurationError

__all__ = [
    'BaseService',
    'CacheableService', 
    'ServiceProtocol',
    'ServiceError',
    'ConfigurationError'
]
```

```python
# code_intelligence_mcp/core/service.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Protocol
import logging
import asyncio


class ServiceProtocol(Protocol):
    """Protocol for all services."""
    
    async def initialize(self) -> None:
        """Initialize the service."""
        ...
    
    async def shutdown(self) -> None:
        """Cleanup service resources."""
        ...
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        ...


class BaseService(ABC):
    """Base class for all services."""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            self._logger.warning(f"{self.__class__.__name__} already initialized")
            return
        
        self._logger.info(f"Initializing {self.__class__.__name__}")
        try:
            await self._initialize()
            self._initialized = True
            self._logger.info(f"{self.__class__.__name__} initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            raise
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Service-specific initialization."""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the service."""
        if not self._initialized:
            return
        
        self._logger.info(f"Shutting down {self.__class__.__name__}")
        self._shutdown_event.set()
        
        try:
            await self._shutdown()
            self._initialized = False
            self._logger.info(f"{self.__class__.__name__} shut down successfully")
        except Exception as e:
            self._logger.error(f"Error during shutdown of {self.__class__.__name__}: {e}")
            raise
    
    async def _shutdown(self) -> None:
        """Service-specific shutdown."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


class CacheableService(BaseService):
    """Base for services that support caching."""
    
    def __init__(self, cache_service: Optional['CacheService'] = None, **kwargs):
        super().__init__(**kwargs)
        self._cache = cache_service
    
    async def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self._cache and self._cache.is_initialized:
            return await self._cache.get(key)
        return None
    
    async def _set_cached(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if self._cache and self._cache.is_initialized:
            await self._cache.set(key, value, ttl)
    
    async def _invalidate_cache(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern."""
        if self._cache and self._cache.is_initialized:
            await self._cache.invalidate(pattern)
```

```python
# code_intelligence_mcp/core/errors.py
from typing import Optional, Dict, Any


class ServiceError(Exception):
    """Base exception for service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code or "SERVICE_ERROR"
        self.details = details or {}


class ConfigurationError(ServiceError):
    """Configuration related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key} if config_key else {}
        )


class InitializationError(ServiceError):
    """Service initialization errors."""
    
    def __init__(self, service_name: str, reason: str):
        super().__init__(
            f"Failed to initialize {service_name}: {reason}",
            error_code="INITIALIZATION_ERROR",
            details={"service": service_name, "reason": reason}
        )
```

### Step 2: Create DI Container

```python
# code_intelligence_mcp/di/__init__.py
from .container import DIContainer, ScopedContainer
from .bootstrap import create_container, register_services

__all__ = [
    'DIContainer',
    'ScopedContainer',
    'create_container',
    'register_services'
]
```

```python
# code_intelligence_mcp/di/container.py
from typing import Dict, Any, TypeVar, Type, Optional, Callable, Union
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager
import inspect

from ..core import ServiceProtocol, ServiceError


T = TypeVar('T')


class DIContainer:
    """Central dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Optional[Any]] = {}
        self._scoped_types: set = set()
        self._lock = asyncio.Lock()
        self._initialized = False
    
    def register_factory(
        self,
        interface: Type[T],
        factory: Callable[[DIContainer], Union[T, asyncio.Future[T]]],
        singleton: bool = True,
        scoped: bool = False
    ) -> None:
        """Register a factory function for creating instances."""
        self._factories[interface] = factory
        
        if singleton:
            self._singletons[interface] = None
        
        if scoped:
            self._scoped_types.add(interface)
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register an existing instance."""
        self._services[interface] = instance
        self._singletons[interface] = instance
    
    async def get(self, interface: Type[T]) -> T:
        """Get an instance of the requested interface."""
        async with self._lock:
            # Check if we have a direct instance
            if interface in self._services:
                return self._services[interface]
            
            # Check if it's a singleton
            if interface in self._singletons:
                if self._singletons[interface] is None:
                    # Create singleton instance
                    instance = await self._create_instance(interface)
                    self._singletons[interface] = instance
                    self._services[interface] = instance
                return self._singletons[interface]
            
            # Create new instance
            return await self._create_instance(interface)
    
    async def _create_instance(self, interface: Type[T]) -> T:
        """Create an instance using the registered factory."""
        if interface not in self._factories:
            raise ServiceError(
                f"No factory registered for {interface.__name__}",
                error_code="NO_FACTORY_REGISTERED",
                details={"interface": interface.__name__}
            )
        
        factory = self._factories[interface]
        
        try:
            # Get factory signature to inject dependencies
            sig = inspect.signature(factory)
            kwargs = {}
            
            # Auto-inject container if requested
            if 'container' in sig.parameters:
                kwargs['container'] = self
            
            # Call factory
            if asyncio.iscoroutinefunction(factory):
                instance = await factory(**kwargs)
            else:
                instance = factory(**kwargs)
            
            # Initialize if it's a service
            if hasattr(instance, 'initialize') and not instance.is_initialized:
                await instance.initialize()
            
            return instance
            
        except Exception as e:
            raise ServiceError(
                f"Failed to create instance of {interface.__name__}: {str(e)}",
                error_code="INSTANCE_CREATION_FAILED",
                details={"interface": interface.__name__, "error": str(e)}
            )
    
    def is_scoped(self, interface: Type) -> bool:
        """Check if interface is scoped."""
        return interface in self._scoped_types
    
    @asynccontextmanager
    async def scope(self):
        """Create a scoped container for request-level dependencies."""
        scoped_container = ScopedContainer(self)
        try:
            yield scoped_container
        finally:
            await scoped_container.cleanup()
    
    async def initialize_all(self):
        """Initialize all registered singletons."""
        if self._initialized:
            return
        
        # Initialize all singletons
        for interface in list(self._singletons.keys()):
            if self._singletons[interface] is None:
                await self.get(interface)
        
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown all services."""
        # Shutdown in reverse order of creation
        for service in reversed(list(self._services.values())):
            if hasattr(service, 'shutdown'):
                try:
                    await service.shutdown()
                except Exception as e:
                    # Log error but continue shutdown
                    print(f"Error shutting down {service.__class__.__name__}: {e}")
        
        self._services.clear()
        self._singletons.clear()
        self._initialized = False


class ScopedContainer(DIContainer):
    """Scoped container for request-level dependencies."""
    
    def __init__(self, parent: DIContainer):
        super().__init__()
        self._parent = parent
        self._scoped_instances: Dict[Type, Any] = {}
    
    async def get(self, interface: Type[T]) -> T:
        """Get scoped instance or delegate to parent."""
        # Check if we already have a scoped instance
        if interface in self._scoped_instances:
            return self._scoped_instances[interface]
        
        # Check if it's a scoped service
        if self._parent.is_scoped(interface):
            instance = await self._parent._create_instance(interface)
            self._scoped_instances[interface] = instance
            return instance
        
        # Delegate to parent
        return await self._parent.get(interface)
    
    async def cleanup(self):
        """Cleanup scoped instances."""
        for instance in self._scoped_instances.values():
            if hasattr(instance, 'shutdown'):
                await instance.shutdown()
        self._scoped_instances.clear()
```

### Step 3: Implement GraphService Starter

```python
# code_intelligence_mcp/graph/__init__.py
from .service import GraphService
from .duckdb_graph import DuckDBGraph

__all__ = ['GraphService', 'DuckDBGraph']
```

```python
# code_intelligence_mcp/graph/service.py
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
    
    async def _shutdown(self) -> None:
        """Shutdown graph database."""
        if self._graph:
            self._graph.close()
            self._graph = None
    
    # Delegate methods to DuckDBGraph
    async def add_symbol(self, **kwargs) -> int:
        """Add a symbol to the graph."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.add_symbol(**kwargs)
    
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
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._graph:
            raise RuntimeError("GraphService not initialized")
        return await self._graph.get_statistics()
    
    # Add more delegated methods as needed
```

## Phase 2: Create Minimal Working Server

### Step 4: Create Simplified Bootstrap

```python
# code_intelligence_mcp/di/bootstrap.py
from pathlib import Path
from typing import Optional

from .container import DIContainer
from ..config import Config, get_config
from ..graph import GraphService, DuckDBGraph
from ..tools import SearchTools


async def create_container(config: Optional[Config] = None) -> DIContainer:
    """Create and configure the DI container with minimal services."""
    if config is None:
        config = get_config()
    
    container = DIContainer()
    
    # Register configuration
    container.register_instance(Config, config)
    
    # Register GraphService
    container.register_factory(
        GraphService,
        lambda c: GraphService(
            db_path=config.graph.db_path,
            config=config.graph
        ),
        singleton=True
    )
    
    # Register basic tools
    container.register_factory(
        SearchTools,
        lambda c: SearchTools(
            graph=c.get(GraphService),
            vector=None,  # Add later
            parser=None,  # Add later
            cache=None    # Add later
        ),
        singleton=False
    )
    
    # Initialize all services
    await container.initialize_all()
    
    return container


async def register_services(container: DIContainer, config: Config):
    """Register additional services."""
    # This will be expanded as we add more services
    pass
```

### Step 5: Create Minimal MCP Server

```python
# code_intelligence_mcp/server_minimal.py
"""Minimal MCP server for initial testing."""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import sys

from mcp.server import Server
from mcp.types import Tool

from .config import get_config, Config
from .di import create_container
from .graph import GraphService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinimalCodeIntelligenceMCP:
    """Minimal MCP server implementation."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.server = Server("code-intelligence-minimal")
        self.container = None
    
    async def initialize(self):
        """Initialize the server."""
        # Create DI container
        self.container = await create_container(self.config)
        
        # Get services
        self.graph = await self.container.get(GraphService)
        
        # Register minimal tools
        self._register_tools()
    
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
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def find_symbol(name: str) -> Dict[str, Any]:
            """Find a symbol by name."""
            try:
                symbols = await self.graph.find_symbol(name=name)
                return {
                    "status": "success",
                    "count": len(symbols),
                    "symbols": symbols[:10]  # Limit results
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    async def run(self):
        """Run the MCP server."""
        await self.initialize()
        logger.info("Minimal Code Intelligence MCP server started")
        await self.server.run()
    
    async def shutdown(self):
        """Shutdown the server."""
        if self.container:
            await self.container.shutdown()


async def main():
    """Main entry point."""
    server = MinimalCodeIntelligenceMCP()
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

## Phase 3: Test the Minimal Server

### Step 6: Create Test Script

```python
# test_minimal_server.py
"""Test script for minimal server."""

import asyncio
from pathlib import Path
import tempfile

from code_intelligence_mcp.config import Config, GraphConfig
from code_intelligence_mcp.server_minimal import MinimalCodeIntelligenceMCP


async def test_server():
    """Test the minimal server."""
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test configuration
        config = Config(
            project_root=Path(__file__).parent,
            data_directory=Path(tmpdir) / "data",
            graph=GraphConfig(
                db_path=Path(tmpdir) / "data" / "test.duckdb"
            )
        )
        
        # Create server
        server = MinimalCodeIntelligenceMCP(config)
        
        # Initialize
        await server.initialize()
        
        # Test tools
        print("Testing connection...")
        result = await server.server._tools["test_connection"]()
        print(f"Result: {result}")
        
        print("\nGetting index stats...")
        result = await server.server._tools["get_index_stats"]()
        print(f"Result: {result}")
        
        print("\nSearching for symbol...")
        result = await server.server._tools["find_symbol"](name="test")
        print(f"Result: {result}")
        
        # Shutdown
        await server.shutdown()
        print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_server())
```

## Phase 4: Incremental Implementation

### Step 7: Add Parser Service Stub

```python
# code_intelligence_mcp/parser/service.py
from typing import List, Dict, Any, Tuple
from pathlib import Path

from ..core import BaseService
from ..config import ParserConfig


class ParserService(BaseService):
    """Stub parser service for testing."""
    
    def __init__(self, config: ParserConfig):
        super().__init__(config=config)
    
    async def _initialize(self) -> None:
        """Initialize parser."""
        self._logger.info("ParserService initialized (stub)")
    
    async def analyze_file(
        self,
        file_path: Path,
        content: bytes
    ) -> Tuple[List[Any], List[Any]]:
        """Stub implementation."""
        # Return empty results for now
        return [], []
    
    async def get_context_lines(
        self,
        file_path: str,
        line: int,
        before: int = 3,
        after: int = 3
    ) -> List[str]:
        """Get context lines around a line number."""
        # Stub implementation
        return [f"Line {i}" for i in range(line - before, line + after + 1)]
```

### Step 8: Add Tools Module Structure

```python
# code_intelligence_mcp/tools/__init__.py
from .search_tools import SearchTools
from .base import BaseTool

__all__ = ['SearchTools', 'BaseTool']
```

```python
# code_intelligence_mcp/tools/base.py
from typing import Any, Dict
from abc import ABC, abstractmethod

from ..core import BaseService


class BaseTool(BaseService):
    """Base class for MCP tools."""
    
    async def _initialize(self) -> None:
        """Tools don't need initialization."""
        pass
    
    async def validate_inputs(self, **kwargs) -> None:
        """Validate tool inputs."""
        # Override in subclasses
        pass
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with validation."""
        await self.validate_inputs(**kwargs)
        return await self._execute(**kwargs)
    
    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """Tool-specific execution."""
        pass
```

```python
# code_intelligence_mcp/tools/search_tools.py
from typing import Optional, Dict, Any, List

from .base import BaseTool


class SearchTools:
    """Container for search tools."""
    
    def __init__(
        self,
        graph,
        vector=None,
        parser=None,
        cache=None
    ):
        self.graph = graph
        self.vector = vector
        self.parser = parser
        self.cache = cache
        
        # Initialize individual tools
        self.find_usages = FindUsagesStub(graph)


class FindUsagesStub(BaseTool):
    """Stub implementation of find usages."""
    
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
    
    async def _execute(
        self,
        name: str,
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Find usages of a symbol."""
        # Simple implementation for testing
        symbols = await self.graph.find_symbol(name=name, file_path=file_path)
        
        if not symbols:
            return {
                "error": f"Symbol '{name}' not found",
                "suggestions": []
            }
        
        symbol = symbols[0]
        
        # For now, just return the symbol info
        return {
            "symbol": symbol,
            "usages": [],  # Will implement actual usage finding later
            "message": "Usage finding not yet implemented"
        }
```

## Next Steps for Claude Code

1. **Run the test script** to verify basic functionality
2. **Implement DuckDB schema** if not already present
3. **Add more tools incrementally**
4. **Implement parser with tree-sitter**
5. **Add vector store support**
6. **Implement full indexing**

Each component can be tested independently before integration.

## Common Issues and Solutions

### Issue: Import Errors
```python
# Add to the root __init__.py files
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### Issue: Async Context
```python
# Always use proper async context
async def main():
    # Your code here
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

### Issue: Database Lock
```python
# Ensure single connection per operation
with self.connection() as conn:
    # Do all operations
    pass
# Connection automatically closed
```

This guide provides a solid foundation for implementing the Code Intelligence MCP server incrementally.
