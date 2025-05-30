"""Dependency Injection Container implementation."""

from __future__ import annotations
from typing import Dict, Any, TypeVar, Type, Optional, Callable, Union
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager
import inspect

from ..core import ServiceProtocol, ServiceError, DependencyError


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
    
    def get_sync(self, interface: Type[T]) -> T:
        """Get an already-created instance synchronously (for use in lambdas)."""
        if interface in self._services:
            return self._services[interface]
        if interface in self._singletons and self._singletons[interface] is not None:
            return self._singletons[interface]
        raise DependencyError(f"Instance not available synchronously: {interface.__name__}")
    
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
            raise DependencyError(
                f"No factory registered for {interface.__name__}",
                dependency=interface.__name__
            )
        
        factory = self._factories[interface]
        
        try:
            # Get factory signature to inject dependencies
            sig = inspect.signature(factory)
            kwargs = {}
            
            # Auto-inject container if requested
            if 'container' in sig.parameters:
                kwargs['container'] = self
            
            # Auto-inject other dependencies based on type hints
            for param_name, param in sig.parameters.items():
                if param_name == 'container':
                    continue
                
                # Check if parameter has a type annotation
                if param.annotation != inspect.Parameter.empty:
                    param_type = param.annotation
                    
                    # Try to resolve the dependency
                    if param_type in self._factories or param_type in self._services:
                        kwargs[param_name] = await self.get(param_type)
            
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
