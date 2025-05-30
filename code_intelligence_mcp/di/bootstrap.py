"""Bootstrap configuration for Dependency Injection."""

from pathlib import Path
from typing import Optional

from .container import DIContainer
from ..config import Config, get_config
from ..graph import GraphService
from ..parser import ParserService
from ..vector import VectorService, EmbeddingService
from ..tools import SearchTools, AnalysisTools, IndexTools


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
        lambda container: GraphService(
            db_path=config.graph.db_path,
            config=config.graph
        ),
        singleton=True
    )
    
    # Register ParserService
    container.register_factory(
        ParserService,
        lambda container: ParserService(
            config=config.parser,
            cache_service=None  # TODO: Add when CacheService is implemented
        ),
        singleton=True
    )
    
    # Register EmbeddingService
    container.register_factory(
        EmbeddingService,
        lambda container: EmbeddingService(config=config.vector),
        singleton=True
    )
    
    # Register VectorService
    async def vector_factory(container):
        embedding_service = await container.get(EmbeddingService)
        return VectorService(config=config.vector, embedding_service=embedding_service)
    
    container.register_factory(
        VectorService,
        vector_factory,
        singleton=True
    )
    
    # TODO: Add these services as they are implemented
    # - CacheService
    # - IndexingService
    # - FileWatcher
    
    # Register tool factories (not singletons)
    container.register_factory(
        SearchTools,
        lambda container: create_search_tools(container),
        singleton=False
    )
    
    # Initialize all services
    await container.initialize_all()
    
    return container


async def create_vector_service(container: DIContainer, config) -> VectorService:
    """Factory for creating vector service."""
    embedding_service = await container.get(EmbeddingService)
    return VectorService(config=config, embedding_service=embedding_service)


async def create_search_tools(container: DIContainer) -> SearchTools:
    """Factory for creating search tools."""
    # Get services asynchronously
    graph_service = await container.get(GraphService)
    parser_service = await container.get(ParserService)
    vector_service = await container.get(VectorService)
    
    return SearchTools(
        graph=graph_service,
        vector=vector_service,
        parser=parser_service,
        cache=None    # TODO: Add when CacheService is implemented
    )


async def create_full_container(config: Optional[Config] = None) -> DIContainer:
    """Create container with all services (for future use)."""
    if config is None:
        config = get_config()
    
    container = DIContainer()
    
    # Register configuration
    container.register_instance(Config, config)
    
    # Core services
    container.register_factory(
        GraphService,
        lambda c: GraphService(
            db_path=config.graph.db_path,
            config=config.graph
        ),
        singleton=True
    )
    
    # TODO: Register all other services as they are implemented
    # This function will be expanded as services are added
    
    # Initialize all services
    await container.initialize_all()
    
    return container
