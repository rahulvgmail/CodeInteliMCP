"""
Code Intelligence MCP Server

An advanced MCP server that combines Tree-sitter parsing with graph databases
and vector search for lightning-fast code intelligence.
"""

__version__ = "0.1.0"
__author__ = "Code Intelligence Team"

# Public API
from .config import Config, get_config, set_config
from .server import CodeIntelligenceMCP

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "CodeIntelligenceMCP",
    "__version__"
]
