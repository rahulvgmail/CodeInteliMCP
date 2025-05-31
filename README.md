# CodeInteliMCP - Intelligent Code Analysis Server

An advanced MCP (Model Context Protocol) server that combines Tree-sitter parsing with graph databases and vector search for lightning-fast code intelligence across multiple repositories.

## 🚀 Features

- **Multi-Repository Support**: Manage and analyze multiple codebases simultaneously
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
git clone https://github.com/rahulvgmail/CodeInteliMCP.git
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

## 🛠️ Available Tools

This MCP server provides the following tools:

### Repository Management
- **`add_repository`** - Add a new repository to track
  - `name` (string, required): Repository name
  - `path` (string, required): Path to repository
  - `description` (string, optional): Repository description
  - `make_active` (boolean, optional): Set as active repository

- **`list_repositories`** - List all tracked repositories
  
- **`set_active_repository`** - Set the active repository for operations
  - `name` (string, required): Repository name to make active

- **`index_repository`** - Index an entire repository
  - `repository_name` (string, required): Name of repository to index
  - `include_patterns` (string, optional): Comma-separated file patterns to include
  - `exclude_patterns` (string, optional): Comma-separated file patterns to exclude

### Code Analysis
- **`index_file`** - Index a single file into the code graph
  - `file_path` (string, required): Path to the file to index

- **`find_symbol`** - Find a symbol by name
  - `name` (string, required): Symbol name to search for
  - `file_path` (string, optional): Limit search to specific file

- **`find_usages`** - Find all usages of a symbol
  - `name` (string, required): Symbol name to find usages for
  - `file_path` (string, optional): Limit search to specific file

- **`semantic_search`** - Search for code using semantic similarity
  - `query` (string, required): Natural language search query
  - `limit` (integer, optional): Maximum number of results (default: 10)
  - `symbol_types` (string, optional): Comma-separated list of symbol types to filter

- **`find_similar_code`** - Find code similar to a given symbol
  - `symbol_name` (string, required): Reference symbol name
  - `file_path` (string, optional): File containing the reference symbol

### System Tools
- **`test_connection`** - Test that the server is working

- **`get_index_stats`** - Get current index statistics

- **`get_vector_stats`** - Get vector store statistics

## 📖 Usage Examples

Once configured, use these tools in Claude:

```
# Add a new repository
add_repository(name="my-app", path="/path/to/my-app", description="Main application")

# Index the repository
index_repository(repository_name="my-app", include_patterns="**/*.py,**/*.js")

# Search for authentication functions
semantic_search(query="authentication and login functions", limit=20)

# Find where a class is used
find_usages(name="UserService")

# Find similar implementations
find_similar_code(symbol_name="login_user")
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
