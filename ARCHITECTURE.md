# 🏗️ Project Structure

```
CodeInteliMCP/
├── README.md
├── ARCHITECTURE.md          # This file
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
├── CLAUDE.md              # Instructions for Claude
│
├── code_intelligence_mcp/  # Main package
│   ├── __init__.py
│   ├── server.py          # MCP server entry point
│   ├── config.py          # Configuration management
│   │
│   ├── indexing/          # Indexing subsystem
│   │   ├── __init__.py
│   │   ├── incremental.py # Incremental update logic
│   │   ├── full_index.py  # Full indexing
│   │   ├── watcher.py     # File system watcher
│   │   └── git_sync.py    # Git-aware updates
│   │
│   ├── graph/             # Graph database layer
│   │   ├── __init__.py
│   │   ├── duckdb_graph.py # DuckDB graph implementation
│   │   ├── schema.sql      # Graph schema
│   │   └── queries.py      # Pre-built queries
│   │
│   ├── vector/            # Vector database layer
│   │   ├── __init__.py
│   │   ├── chroma_store.py # ChromaDB implementation
│   │   ├── embeddings.py   # Code embeddings
│   │   └── semantic.py     # Semantic search
│   │
│   ├── parser/            # Tree-sitter parsing
│   │   ├── __init__.py
│   │   ├── language_registry.py
│   │   ├── ast_analyzer.py
│   │   └── symbol_extractor.py
│   │
│   ├── tools/             # MCP tools
│   │   ├── __init__.py
│   │   ├── search_tools.py
│   │   ├── analysis_tools.py
│   │   ├── index_tools.py
│   │   └── viz_tools.py   # Visualization tools
│   │
│   └── utils/
│       ├── __init__.py
│       ├── cache.py
│       ├── logger.py
│       └── progress.py
│
├── tests/
│   ├── __init__.py
│   ├── test_graph.py
│   ├── test_vector.py
│   └── test_integration.py
│
├── scripts/
│   ├── benchmark.py       # Performance testing
│   └── migrate.py         # Migration from other tools
│
└── data/                  # Local storage
    ├── code_graph.duckdb  # Graph database
    ├── chroma/           # Vector embeddings
    └── cache/            # Parse cache
```

## 🗃️ Database Choice: DuckDB for Graphs

### Why DuckDB instead of Neo4j/ArangoDB?

1. **Zero Setup**: No server to start, just a file
2. **Instant Startup**: < 10ms to open database
3. **Graph Queries**: Recursive CTEs work perfectly for code graphs
4. **Analytical Speed**: Columnar storage ideal for code analysis
5. **SQL Interface**: Easier to maintain than Cypher
6. **Embedded**: Ships as a single Python package

### Graph Schema in DuckDB

```sql
-- Entities
CREATE TABLE symbols (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    type VARCHAR NOT NULL,  -- 'function', 'class', 'variable', 'module'
    file_path VARCHAR NOT NULL,
    line_number INTEGER,
    column_number INTEGER,
    signature TEXT,
    docstring TEXT,
    ast_hash VARCHAR,       -- For change detection
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON          -- Language-specific data
);

-- Relationships
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    from_symbol_id INTEGER REFERENCES symbols(id),
    to_symbol_id INTEGER REFERENCES symbols(id),
    relationship_type VARCHAR NOT NULL, -- 'calls', 'imports', 'inherits', 'uses'
    file_path VARCHAR NOT NULL,
    line_number INTEGER,
    confidence FLOAT DEFAULT 1.0,      -- For fuzzy matches
    metadata JSON
);

-- Indexes for performance
CREATE INDEX idx_symbols_name ON symbols(name);
CREATE INDEX idx_symbols_file ON symbols(file_path);
CREATE INDEX idx_rel_from ON relationships(from_symbol_id);
CREATE INDEX idx_rel_to ON relationships(to_symbol_id);
CREATE INDEX idx_rel_type ON relationships(relationship_type);

-- Example recursive query for finding all callers
WITH RECURSIVE callers AS (
    SELECT s.* FROM symbols s
    JOIN relationships r ON s.id = r.from_symbol_id
    WHERE r.to_symbol_id = ? AND r.relationship_type = 'calls'
    
    UNION ALL
    
    SELECT s.* FROM symbols s
    JOIN relationships r ON s.id = r.from_symbol_id
    JOIN callers c ON r.to_symbol_id = c.id
    WHERE r.relationship_type = 'calls'
)
SELECT DISTINCT * FROM callers;
```

## 🧠 Vector Storage: ChromaDB

### Configuration

```python
import chromadb
from chromadb.config import Settings

# Lightweight local setup
client = chromadb.PersistentClient(
    path="./data/chroma",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True,
    )
)

# Collections for different embedding types
code_collection = client.create_collection(
    name="code_embeddings",
    metadata={"hnsw:space": "cosine"}
)

doc_collection = client.create_collection(
    name="documentation", 
    metadata={"hnsw:space": "cosine"}
)
```

## 🔥 Killer Features to Implement

### 1. Smart Context Window Management
```python
@tool
async def get_relevant_context(file_path: str, line_number: int) -> dict:
    """Get only the most relevant code context for current cursor position"""
    # Uses graph to find related symbols
    # Uses vectors to find similar patterns
    # Returns optimized context that fits in Claude's window
```

### 2. Architectural Pattern Detection
```python
@tool  
async def detect_patterns(project_path: str) -> dict:
    """Detect design patterns and architectural styles"""
    # MVC, Repository, Factory patterns
    # Microservices vs monolith
    # Testing patterns
```

### 3. Code Evolution Timeline
```python
@tool
async def function_history(function_name: str) -> dict:
    """Show how a function evolved over time"""
    # Git integration
    # Complexity trends
    # Breaking change detection
```

### 4. Intelligent Refactoring Suggestions
```python
@tool
async def suggest_refactoring(file_path: str) -> dict:
    """AI-powered refactoring suggestions"""
    # Duplicate code detection via vectors
    # Extract method opportunities
    # Dead code identification
```

### 5. Cross-Language Intelligence
```python
@tool
async def find_implementations(interface_name: str) -> dict:
    """Find implementations across different languages"""
    # Python protocol → TypeScript interface
    # Go interface → Rust trait
```

## 🚀 Performance Optimizations

### 1. Lazy Loading Strategy
- Index only opened files initially
- Background index rest of project
- Prioritize frequently accessed paths

### 2. Smart Caching
- LRU cache for AST parsing
- Memoize graph queries
- Vector similarity cache

### 3. Incremental Updates
- File watcher with debouncing
- Git diff-based updates
- Hash-based change detection

### 4. Query Optimization
- Pre-computed common queries
- Materialized views for hot paths
- Parallel processing for bulk operations

## 🎯 Development Phases

### Phase 1: Core Infrastructure (Week 1)
- [x] Project structure
- [ ] Basic MCP server
- [ ] DuckDB integration
- [ ] ChromaDB setup
- [ ] Tree-sitter multi-language support

### Phase 2: Essential Features (Week 2)
- [ ] Find usages (graph query)
- [ ] Find similar code (vector search)
- [ ] Basic incremental indexing
- [ ] Claude Code integration

### Phase 3: Advanced Features (Week 3)
- [ ] Git integration
- [ ] Pattern detection
- [ ] Smart context management
- [ ] Performance optimization

### Phase 4: Polish (Week 4)
- [ ] UI/UX improvements
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Example workflows
