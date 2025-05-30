"""Dependency Injection system for Code Intelligence MCP."""

from .container import DIContainer, ScopedContainer
from .bootstrap import create_container

__all__ = [
    'DIContainer',
    'ScopedContainer',
    'create_container'
]
