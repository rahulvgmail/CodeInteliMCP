# TreeSitter MCP - Intelligent Code Analysis Server

An advanced MCP server that combines Tree-sitter parsing with graph databases and vector search for lightning-fast code intelligence.

## 🚀 Features

- **Instant Usage Search**: Find where functions/classes are used in milliseconds
- **Semantic Code Search**: Find similar code patterns using vector embeddings
- **Dependency Analysis**: Visualize and query code dependencies
- **Impact Analysis**: See what breaks when you change something
- **Auto-indexing**: Smart incremental updates as you code
- **Multi-language Support**: Python, JavaScript, TypeScript, Go, Rust, and more

## 🏗 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Claude Code    │────▶│   MCP Server     │────▶│   Tree-sitter   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────┐          ┌────────────────┐
                        │  DuckDB      │          │  ChromaDB      │
                        │  (Graph)     │          │  (Vectors)     │
                        └──────────────┘          └────────────────┘
```

## 🎯 Core Capabilities

### 1. Graph Queries (via DuckDB)
- Function call relationships
- Import dependencies  
- Class inheritance trees
- Variable usage tracking

### 2. Vector Search (via ChromaDB)
- Semantic code similarity
- Natural language code search
- Pattern detection
- Code duplication finding

### 3. Real-time Analysis (via Tree-sitter)
- On-demand parsing
- Syntax validation
- Code structure analysis
- AST manipulation

## 🚦 Quick Start

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd CodeInteliMCP

# Run the setup script
python setup_for_claude.py
```

The setup script will:
- Install all dependencies
- Configure Claude Desktop/Code
- Create data directories
- Set up environment variables

### Manual Setup

```bash
# Install dependencies
pip install mcp duckdb chromadb sentence-transformers aiofiles
pip install tree-sitter tree-sitter-python tree-sitter-javascript

# Add to Claude Desktop config (~/.config/claude/claude_desktop_config.json):
{
  "mcpServers": {
    "code-intelligence": {
      "command": "python",
      "args": ["/path/to/CodeInteliMCP/code_intelligence_mcp/server_minimal.py"],
      "env": {
        "CODE_INTEL_PROJECT_ROOT": "/path/to/your/project"
      }
    }
  }
}
```

## 📖 Usage

Once configured, use these commands in Claude:

```
# Index files
index_file: src/main.py

# Semantic search
semantic_search: "authentication functions"

# Find symbol usage
find_usages: UserService

# Find similar code
find_similar_code: login_user
```

## 📚 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)**: Complete setup and usage guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Technical architecture details
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)**: Development roadmap

## 📊 Performance Targets

- Initial indexing: < 1 minute for 100k LOC
- Incremental updates: < 1 second per file
- Usage queries: < 50ms
- Semantic search: < 200ms
