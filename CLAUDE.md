# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Claude Code Intelligence Guide

## 🧠 Overview

This project is an advanced MCP (Model Control Protocol) server that combines Tree-sitter parsing with graph databases (DuckDB) and vector search (ChromaDB) for lightning-fast code intelligence. It provides instant code analysis capabilities to Claude Code.

## 🛠️ Development Commands

### Essential Commands

```bash
# Install dependencies with dev tools
pip install -e ".[dev]"

# Run the server
python -m code_intelligence_mcp.server

# Run minimal test server
python -m code_intelligence_mcp.server_minimal

# Run tests
pytest tests/ -v

# Linting and formatting
black code_intelligence_mcp/
ruff check code_intelligence_mcp/

# Type checking  
mypy code_intelligence_mcp/

# Test coverage
pytest --cov=code_intelligence_mcp tests/
```

### MCP Integration

```bash
# Add to Claude Code
claude mcp add code-intel python -m code_intelligence_mcp

# Or run directly
python -m code_intelligence_mcp.server
```

## 🏗️ Architecture Overview

### Core Components

1. **Dependency Injection System** (`di/`)
   - `DIContainer`: Central DI container managing service lifecycles
   - `ScopedContainer`: Request-scoped dependencies
   - All services follow `BaseService` pattern with async initialization

2. **Graph Database** (`graph/`)
   - DuckDB-based graph storage for code relationships
   - Schema in `graph/schema.sql`
   - Stores symbols, relationships, and metadata
   - Optimized for recursive queries (call chains, dependencies)

3. **Parser System** (`parser/`)
   - Tree-sitter based multi-language parsing
   - Query files in `parser/queries/` for each language
   - Extracts symbols, relationships, and semantic information

4. **Vector Store** (planned)
   - ChromaDB for semantic code search
   - Code embeddings using sentence-transformers
   - Natural language code queries

5. **MCP Tools** (`tools/`)
   - Search tools: find usages, similar code, dependencies
   - Analysis tools: impact analysis, pattern detection
   - Index tools: incremental updates, full reindex

### Service Architecture Pattern

All services follow this pattern:
```python
class MyService(BaseService):
    async def _initialize(self) -> None:
        # Service-specific initialization
        
    async def _shutdown(self) -> None:
        # Cleanup resources
```

Services are registered in the DI container and auto-initialized.

## 🚀 Implementation Status

### ✅ Completed
- Project structure and architecture
- Configuration system with environment overrides
- DuckDB graph database implementation
- Dependency injection container
- Base service patterns
- Documentation framework

### 🚧 In Progress
- MCP server with basic tools
- Tree-sitter parser integration
- Search tools implementation

### 📋 TODO
- ChromaDB vector store integration
- Full indexing system
- File watcher for auto-updates
- Advanced analysis tools
- Performance optimizations

## 💡 Key Design Decisions

1. **DuckDB over Neo4j**: Zero setup, embedded, fast analytical queries, SQL interface
2. **Async Everything**: All services use async/await for performance
3. **DI Container**: Clean dependency management and testability
4. **Incremental Updates**: Smart change detection to avoid full reindexing
5. **Multi-language Support**: Extensible parser system for any tree-sitter grammar

## 🔍 Working with the Codebase

### Adding a New Service

1. Create service class inheriting from `BaseService`
2. Implement `_initialize()` and `_shutdown()` methods
3. Register in `di/bootstrap.py`
4. Add configuration in `config.py` if needed

### Adding a New Tool

1. Create tool class in `tools/` directory
2. Implement tool logic with proper error handling
3. Register in MCP server (`server.py`)
4. Add tests in `tests/`

### Testing Strategy

- Unit tests for individual services
- Integration tests for tool workflows
- Performance benchmarks for large codebases
- Mock external dependencies in tests

## 📊 Performance Considerations

- **Lazy Loading**: Index only what's needed initially
- **Caching**: LRU cache for parsed ASTs and query results
- **Batch Operations**: Process multiple files concurrently
- **Connection Pooling**: Reuse database connections

## 🚨 Common Pitfalls

1. **Database Locks**: Always use context managers for DB connections
2. **Async Context**: Ensure proper async/await usage throughout
3. **Service Initialization**: Services must be initialized before use
4. **Path Handling**: Use `pathlib.Path` consistently

## 🎯 Quick Implementation Guide

When implementing new features:

1. Start with the service layer (`core/service.py` pattern)
2. Register in DI container (`di/bootstrap.py`)
3. Add configuration if needed (`config.py`)
4. Implement tools that use the service
5. Write tests alongside implementation
6. Update documentation in relevant `.md` files

## 🔧 Debugging Tips

- Enable debug logging: `LOG_LEVEL=DEBUG`
- Check service initialization order in container
- Verify database schema matches expected structure
- Use `test_minimal_server.py` for quick testing

Remember: This is a high-performance code intelligence system. Keep operations fast, use caching wisely, and leverage the graph database for complex queries.