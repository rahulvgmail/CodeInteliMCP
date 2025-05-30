# 🏗️ Dependency Injection Architecture

## Overview

This project uses a comprehensive dependency injection (DI) pattern to ensure testability, modularity, and clean separation of concerns. All components should follow this pattern strictly.

## Core DI Container

### Implementation Pattern

```python
# code_intelligence_mcp/di/container.py
from typing import Dict, Any, TypeVar, Type, Optional, Protocol
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager

T = TypeVar('T')


class ServiceProtocol(Protocol):
    """Base protocol for all services."""
    async def initialize(self) -> None:
        """Initialize the service."""
        ...
    
    async def shutdown(self) -> None:
        """Cleanup service resources."""
        ...


class DIContainer:
    """Central dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = asyncio.Lock()
    
    def register_factory(self, interface: Type[T], factory):
        """Register a factory function for creating instances."""
        self._factories[interface] = factory
    
    def register_singleton(self, interface: Type[T], factory):
        """Register a singleton factory."""
        self._factories[interface] = factory
        self._singletons[interface] = None
    
    async def get(self, interface: Type[T]) -> T:
        """Get an instance of the requested interface."""
        async with self._lock:
            # Check if it's a singleton
            if interface in self._singletons:
                if self._singletons[interface] is None:
                    # Create singleton instance
                    instance = await self._create_instance(interface)
                    self._singletons[interface] = instance
                return self._singletons[interface]
            
            # Create new instance
            return await self._create_instance(interface)
    
    async def _create_instance(self, interface: Type[T]) -> T:
        """Create an instance using the registered factory."""
        if interface not in self._factories:
            raise ValueError(f"No factory registered for {interface}")
        
        factory = self._factories[interface]
        
        # Handle async factories
        if asyncio.iscoroutinefunction(factory):
            instance = await factory(self)
        else:
            instance = factory(self)
        
        # Initialize if it's a service
        if hasattr(instance, 'initialize'):
            await instance.initialize()
        
        return instance
    
    @asynccontextmanager
    async def scope(self):
        """Create a scoped container for request-level dependencies."""
        scoped_container = ScopedContainer(self)
        try:
            yield scoped_container
        finally:
            await scoped_container.cleanup()
    
    async def shutdown(self):
        """Shutdown all services."""
        for service in self._singletons.values():
            if service and hasattr(service, 'shutdown'):
                await service.shutdown()


class ScopedContainer(DIContainer):
    """Scoped container for request-level dependencies."""
    
    def __init__(self, parent: DIContainer):
        super().__init__()
        self._parent = parent
        self._scoped_instances: Dict[Type, Any] = {}
    
    async def get(self, interface: Type[T]) -> T:
        """Get scoped instance or delegate to parent."""
        if interface in self._scoped_instances:
            return self._scoped_instances[interface]
        
        # Check if it's a scoped service
        if self._is_scoped(interface):
            instance = await super().get(interface)
            self._scoped_instances[interface] = instance
            return instance
        
        # Delegate to parent
        return await self._parent.get(interface)
    
    def _is_scoped(self, interface: Type) -> bool:
        """Check if interface should be scoped."""
        # Add logic to determine scoped services
        return getattr(interface, '__scoped__', False)
    
    async def cleanup(self):
        """Cleanup scoped instances."""
        for instance in self._scoped_instances.values():
            if hasattr(instance, 'shutdown'):
                await instance.shutdown()
```

## Service Registration

### Bootstrap Module

```python
# code_intelligence_mcp/di/bootstrap.py
from .container import DIContainer
from ..graph import DuckDBGraph, GraphService
from ..vector import ChromaStore, VectorService
from ..parser import ParserService, LanguageRegistry
from ..indexing import IndexingService, FileWatcher
from ..cache import CacheService
from ..config import Config


async def create_container(config: Config) -> DIContainer:
    """Create and configure the DI container."""
    container = DIContainer()
    
    # Register configuration
    container.register_singleton(Config, lambda _: config)
    
    # Register core services
    container.register_singleton(
        GraphService,
        lambda c: GraphService(
            db_path=config.graph.db_path,
            config=config.graph
        )
    )
    
    container.register_singleton(
        VectorService,
        lambda c: VectorService(
            persist_directory=config.vector.persist_directory,
            config=config.vector
        )
    )
    
    container.register_singleton(
        LanguageRegistry,
        lambda c: LanguageRegistry(config.parser.supported_languages)
    )
    
    container.register_singleton(
        ParserService,
        lambda c: ParserService(
            language_registry=c.get(LanguageRegistry),
            config=config.parser
        )
    )
    
    container.register_singleton(
        CacheService,
        lambda c: CacheService(
            cache_dir=config.cache.cache_directory,
            config=config.cache
        )
    )
    
    container.register_singleton(
        FileWatcher,
        lambda c: FileWatcher(
            graph=c.get(GraphService),
            vector=c.get(VectorService),
            parser=c.get(ParserService),
            config=config.indexing
        )
    )
    
    container.register_singleton(
        IndexingService,
        lambda c: IndexingService(
            graph=c.get(GraphService),
            vector=c.get(VectorService),
            parser=c.get(ParserService),
            cache=c.get(CacheService),
            config=config.indexing
        )
    )
    
    # Register tool services
    container.register_factory(
        SearchTools,
        lambda c: SearchTools(
            graph=c.get(GraphService),
            vector=c.get(VectorService),
            parser=c.get(ParserService)
        )
    )
    
    container.register_factory(
        AnalysisTools,
        lambda c: AnalysisTools(
            graph=c.get(GraphService),
            indexing=c.get(IndexingService)
        )
    )
    
    return container
```

## Service Base Classes

### Abstract Service Base

```python
# code_intelligence_mcp/core/service.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging


class BaseService(ABC):
    """Base class for all services."""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return
        
        self._logger.info(f"Initializing {self.__class__.__name__}")
        await self._initialize()
        self._initialized = True
        self._logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Service-specific initialization."""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the service."""
        if not self._initialized:
            return
        
        self._logger.info(f"Shutting down {self.__class__.__name__}")
        await self._shutdown()
        self._initialized = False
        self._logger.info(f"{self.__class__.__name__} shut down")
    
    async def _shutdown(self) -> None:
        """Service-specific shutdown."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized


class CacheableService(BaseService):
    """Base for services that support caching."""
    
    def __init__(self, cache_service: Optional['CacheService'] = None, **kwargs):
        super().__init__(**kwargs)
        self._cache = cache_service
    
    async def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self._cache:
            return await self._cache.get(key)
        return None
    
    async def _set_cached(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if self._cache:
            await self._cache.set(key, value, ttl)
```

## Tool Implementation Pattern

### Tool Base with DI

```python
# code_intelligence_mcp/tools/base.py
from typing import Protocol, runtime_checkable
from ..core.service import BaseService


@runtime_checkable
class Tool(Protocol):
    """Protocol for MCP tools."""
    async def execute(self, **kwargs) -> Any:
        """Execute the tool."""
        ...


class BaseTool(BaseService):
    """Base class for tools with DI support."""
    
    def __init__(self, **services):
        """Initialize with injected services."""
        super().__init__()
        self._services = services
    
    async def _initialize(self) -> None:
        """Tools typically don't need initialization."""
        pass
    
    async def validate_inputs(self, **kwargs) -> None:
        """Validate tool inputs."""
        # Override in subclasses
        pass
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with validation."""
        await self.validate_inputs(**kwargs)
        return await self._execute(**kwargs)
    
    async def _execute(self, **kwargs) -> Any:
        """Tool-specific execution."""
        raise NotImplementedError
```

### Example Tool Implementation

```python
# code_intelligence_mcp/tools/search_tools.py
from typing import List, Dict, Any, Optional
from .base import BaseTool
from ..graph import GraphService
from ..vector import VectorService
from ..parser import ParserService


class FindUsagesTool(BaseTool):
    """Tool for finding symbol usages."""
    
    def __init__(
        self,
        graph: GraphService,
        vector: VectorService,
        parser: ParserService
    ):
        super().__init__(
            graph=graph,
            vector=vector,
            parser=parser
        )
        self.graph = graph
        self.vector = vector
        self.parser = parser
    
    async def validate_inputs(self, name: str, **kwargs) -> None:
        """Validate inputs."""
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
    
    async def _execute(
        self,
        name: str,
        file_path: Optional[str] = None,
        include_indirect: bool = False,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Find all usages of a symbol."""
        # Find the symbol
        symbols = await self.graph.find_symbol(
            name=name,
            file_path=file_path
        )
        
        if not symbols:
            return {
                "error": f"Symbol '{name}' not found",
                "suggestions": await self._find_similar_symbols(name)
            }
        
        # Get primary symbol (handle multiple matches)
        symbol = self._select_best_match(symbols, file_path)
        
        # Find usages
        usages = await self.graph.find_usages(
            symbol['id'],
            include_indirect=include_indirect,
            limit=limit
        )
        
        # Enhance with context
        enhanced_usages = []
        for usage in usages:
            context = await self._get_usage_context(usage)
            enhanced_usages.append({
                **usage,
                'context': context
            })
        
        return {
            'symbol': symbol,
            'usages': enhanced_usages,
            'total_count': len(enhanced_usages),
            'truncated': len(usages) >= limit
        }
    
    async def _find_similar_symbols(self, name: str) -> List[str]:
        """Find similar symbol names."""
        # Use vector similarity or fuzzy matching
        similar = await self.vector.find_similar_names(name, limit=5)
        return [s['name'] for s in similar]
    
    def _select_best_match(
        self,
        symbols: List[Dict],
        preferred_file: Optional[str]
    ) -> Dict:
        """Select best matching symbol."""
        # Prefer symbol from specified file
        if preferred_file:
            for symbol in symbols:
                if symbol['file_path'] == preferred_file:
                    return symbol
        
        # Otherwise, prefer public symbols
        for symbol in symbols:
            if symbol.get('visibility') == 'public':
                return symbol
        
        # Default to first
        return symbols[0]
    
    async def _get_usage_context(self, usage: Dict) -> Dict[str, Any]:
        """Get context around usage."""
        # Get surrounding code
        file_path = usage['file_path']
        line = usage['line_start']
        
        context_lines = await self.parser.get_context_lines(
            file_path,
            line,
            before=2,
            after=2
        )
        
        return {
            'code': '\n'.join(context_lines),
            'function': usage.get('parent_function'),
            'class': usage.get('parent_class')
        }
```

## Testing with DI

### Test Container Setup

```python
# tests/conftest.py
import pytest
from code_intelligence_mcp.di.container import DIContainer
from code_intelligence_mcp.di.bootstrap import create_container
from code_intelligence_mcp.config import Config


@pytest.fixture
async def test_config():
    """Test configuration."""
    return Config(
        project_root="/tmp/test_project",
        data_directory="/tmp/test_data"
    )


@pytest.fixture
async def test_container(test_config):
    """Test DI container."""
    container = await create_container(test_config)
    yield container
    await container.shutdown()


@pytest.fixture
async def mock_graph_service():
    """Mock graph service for unit tests."""
    from unittest.mock import AsyncMock
    
    mock = AsyncMock()
    mock.find_symbol.return_value = [
        {
            'id': 1,
            'name': 'test_function',
            'file_path': 'test.py'
        }
    ]
    mock.find_usages.return_value = []
    
    return mock


@pytest.fixture
async def find_usages_tool(mock_graph_service):
    """Create tool with mocked dependencies."""
    from code_intelligence_mcp.tools.search_tools import FindUsagesTool
    
    tool = FindUsagesTool(
        graph=mock_graph_service,
        vector=AsyncMock(),
        parser=AsyncMock()
    )
    return tool
```

### Example Test

```python
# tests/test_find_usages.py
import pytest


@pytest.mark.asyncio
async def test_find_usages_basic(find_usages_tool):
    """Test basic usage finding."""
    result = await find_usages_tool.execute(
        name="test_function"
    )
    
    assert result['symbol']['name'] == 'test_function'
    assert result['usages'] == []
    assert result['total_count'] == 0


@pytest.mark.asyncio
async def test_find_usages_not_found(find_usages_tool):
    """Test when symbol is not found."""
    find_usages_tool.graph.find_symbol.return_value = []
    
    result = await find_usages_tool.execute(
        name="nonexistent"
    )
    
    assert 'error' in result
    assert 'suggestions' in result
```

## Key Principles

### 1. Constructor Injection
- All dependencies passed through constructor
- No global imports or singletons
- Services don't create their dependencies

### 2. Interface Segregation
- Define protocols for each service interface
- Depend on abstractions, not concretions
- Keep interfaces focused and minimal

### 3. Lifecycle Management
- Services have clear initialization/shutdown
- Container manages lifecycle
- Proper cleanup on shutdown

### 4. Testability First
- Every component easily mockable
- No hidden dependencies
- Clear separation of concerns

### 5. Async by Default
- All service methods async
- Proper async context managers
- No blocking operations

## Common Patterns

### Repository Pattern

```python
class SymbolRepository:
    """Repository for symbol operations."""
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    async def find_by_name(self, name: str) -> List[Symbol]:
        """Find symbols by name."""
        return await self.db.query(
            "SELECT * FROM symbols WHERE name = ?",
            [name]
        )
```

### Unit of Work Pattern

```python
class UnitOfWork:
    """Transactional unit of work."""
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self._transaction = None
    
    async def __aenter__(self):
        self._transaction = await self.db.begin()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self._transaction.rollback()
        else:
            await self._transaction.commit()
```

### Factory Pattern

```python
class ParserFactory:
    """Factory for creating language parsers."""
    
    def __init__(self, registry: LanguageRegistry):
        self.registry = registry
    
    def create_parser(self, language: str) -> Parser:
        """Create parser for language."""
        parser_class = self.registry.get_parser_class(language)
        return parser_class()
```

## Anti-Patterns to Avoid

### ❌ Service Locator
```python
# BAD
class BadService:
    def __init__(self):
        self.db = ServiceLocator.get(DatabaseService)  # Anti-pattern!
```

### ❌ Hidden Dependencies
```python
# BAD
class BadService:
    def process(self):
        from ..utils import helper  # Hidden dependency!
        return helper.do_something()
```

### ❌ Tight Coupling
```python
# BAD
class BadService:
    def __init__(self):
        self.db = DuckDBGraph()  # Direct instantiation!
```

### ✅ Correct Pattern
```python
# GOOD
class GoodService:
    def __init__(self, db: GraphService, cache: CacheService):
        self.db = db
        self.cache = cache
```

## Implementation Checklist

- [ ] Every service extends BaseService
- [ ] All dependencies injected through constructor
- [ ] Services registered in bootstrap module
- [ ] Protocols defined for service interfaces
- [ ] Unit tests use mocked dependencies
- [ ] No circular dependencies
- [ ] Proper async/await usage
- [ ] Lifecycle methods implemented
- [ ] Logging uses class name
- [ ] Configuration injected, not imported
