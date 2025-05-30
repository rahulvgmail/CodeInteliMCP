"""Base service classes for Code Intelligence MCP."""

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
