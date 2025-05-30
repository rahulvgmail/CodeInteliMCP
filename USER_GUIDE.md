# Code Intelligence MCP - User Guide

## 🚀 Quick Start

This guide will help you set up and use the Code Intelligence MCP server with Claude Desktop and Claude Code.

## 📋 Prerequisites

- Python 3.9 or higher
- Claude Desktop or Claude Code
- Git (optional, for installation from source)

## 🛠️ Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd CodeInteliMCP

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Option 2: Install Required Dependencies Only

```bash
# Create virtual environment
python -m venv code-intelligence-venv
source code-intelligence-venv/bin/activate

# Install core dependencies
pip install mcp duckdb chromadb sentence-transformers aiofiles tree-sitter
pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript
```

## 🔧 Configuration

### Claude Desktop Configuration

Add the following to your Claude Desktop MCP configuration file:

**Location of config file:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "code-intelligence": {
      "command": "python",
      "args": ["/path/to/CodeInteliMCP/code_intelligence_mcp/server_minimal.py"],
      "env": {
        "CODE_INTEL_PROJECT_ROOT": "/path/to/your/codebase",
        "CODE_INTEL_DATA_DIR": "/path/to/data/storage"
      }
    }
  }
}
```

**Replace the paths:**
- `/path/to/CodeInteliMCP/code_intelligence_mcp/server_minimal.py` → Full path to the server script
- `/path/to/your/codebase` → The directory you want to analyze
- `/path/to/data/storage` → Where to store the index data (optional)

### Claude Code Configuration

Create a `.clauderc` file in your project root:

```json
{
  "mcp": {
    "servers": {
      "code-intelligence": {
        "command": "python",
        "args": ["/path/to/CodeInteliMCP/code_intelligence_mcp/server_minimal.py"],
        "env": {
          "CODE_INTEL_PROJECT_ROOT": ".",
          "CODE_INTEL_DATA_DIR": "./.code-intelligence"
        }
      }
    }
  }
}
```

## 📂 Project Setup

### 1. Initialize for Your Codebase

```bash
# Navigate to your project
cd /your/project/directory

# Create data directory
mkdir .code-intelligence

# Set environment variables (optional)
export CODE_INTEL_PROJECT_ROOT=$(pwd)
export CODE_INTEL_DATA_DIR=$(pwd)/.code-intelligence
```

### 2. Start the Server (for testing)

```bash
# From the CodeInteliMCP directory
source venv/bin/activate
python code_intelligence_mcp/server_minimal.py
```

## 🎯 Using with Claude

Once configured, restart Claude Desktop/Code and you'll have access to these tools:

### Basic Operations

```
# Test connection
test_connection

# Get index statistics
get_index_stats

# Get vector store statistics  
get_vector_stats
```

### Indexing Files

```
# Index a single file
index_file: /path/to/your/file.py

# Index multiple files (repeat for each)
index_file: src/main.py
index_file: src/utils.py
index_file: src/models.py
```

### Search Operations

```
# Find symbols by name
find_symbol: MyClass
find_symbol: process_data

# Find where symbols are used
find_usages: authenticate
find_usages: DatabaseConnection

# Semantic search (natural language)
semantic_search: "authentication and login functions"
semantic_search: "database query operations"
semantic_search: "error handling patterns"

# Find similar code
find_similar_code: login_user
find_similar_code: DatabaseService
```

## 💡 Example Workflows

### Workflow 1: Understanding a New Codebase

```
1. "Index the main application files in this project"
   → Use index_file for key files

2. "Show me authentication-related code"
   → semantic_search: "user authentication login verification"

3. "Find all database operations"  
   → semantic_search: "database query insert update delete"

4. "What uses the UserService class?"
   → find_usages: UserService
```

### Workflow 2: Refactoring

```
1. "Find all functions similar to process_payment"
   → find_similar_code: process_payment

2. "Show me error handling patterns"
   → semantic_search: "error handling exception try catch"

3. "What would break if I change the User class?"
   → find_usages: User
```

### Workflow 3: Code Review

```
1. "Find authentication functions in this codebase"
   → semantic_search: "authentication security login"

2. "Show me all API endpoints"
   → semantic_search: "API endpoint route handler"

3. "Find similar validation patterns"
   → find_similar_code: validate_input
```

## 🔍 Available Search Tools

### 1. `find_symbol`
**Purpose**: Find symbols by exact name
**Usage**: `find_symbol: ClassName` or `find_symbol: function_name`
**Returns**: Symbol definitions with file locations

### 2. `find_usages`  
**Purpose**: Find all places where a symbol is used
**Usage**: `find_usages: ClassName`
**Returns**: All usage locations and call sites

### 3. `semantic_search`
**Purpose**: Natural language code search
**Usage**: `semantic_search: "description of what you're looking for"`
**Examples**:
- `"authentication and security functions"`
- `"database connection and queries"`
- `"error handling and logging"`
- `"API endpoints and routes"`

### 4. `find_similar_code`
**Purpose**: Find code similar to a given symbol
**Usage**: `find_similar_code: symbol_name`
**Returns**: Semantically similar functions/classes

### 5. `index_file`
**Purpose**: Add a file to the search index
**Usage**: `index_file: /path/to/file.py`
**Note**: Must be done before searching

## 📊 Understanding Results

### Symbol Information
- **name**: Symbol identifier
- **type**: function, class, method, variable
- **file**: Source file location  
- **line**: Line number in file
- **signature**: Function/method signature
- **docstring**: Documentation

### Usage Information
- **caller**: What calls this symbol
- **location**: Where the call happens
- **context**: Surrounding code context

### Similarity Scores
- **distance**: Lower = more similar (0.0 = identical)
- **document**: Text used for similarity matching

## ⚡ Performance Tips

### Indexing Strategy
```
# Index core files first
index_file: src/main.py
index_file: src/models.py  
index_file: src/services.py

# Then add supporting files
index_file: src/utils.py
index_file: src/helpers.py
```

### Search Optimization
- Use specific terms for better semantic search results
- Combine symbol search with semantic search for comprehensive results
- Use type-specific searches when needed

### Data Management
- Index data is stored in `.code-intelligence/` directory
- Safe to delete for fresh start
- Incremental updates planned for future versions

## 🐛 Troubleshooting

### Common Issues

**"No module named 'mcp'"**
```bash
pip install mcp
```

**"Symbol not found"**
- Ensure the file is indexed first with `index_file`
- Check that the symbol name is exact (case-sensitive)

**"Semantic search returns no results"**
- Try broader search terms
- Index more files to improve corpus
- Check `get_vector_stats` to verify vector data

**Server won't start**
- Check Python path in configuration
- Verify virtual environment activation
- Check file permissions

### Getting Help

1. **Check connection**: Use `test_connection` tool
2. **View statistics**: Use `get_index_stats` and `get_vector_stats`
3. **Check logs**: Server logs show initialization status
4. **Restart**: Restart Claude Desktop/Code after config changes

## 🚀 Advanced Usage

### Environment Variables

```bash
# Project root (where to search for files)
export CODE_INTEL_PROJECT_ROOT=/path/to/project

# Data storage location  
export CODE_INTEL_DATA_DIR=/path/to/data

# Performance tuning
export CODE_INTEL_MAX_WORKERS=4
```

### Configuration File

Create `code_intelligence_config.json`:

```json
{
  "project_root": "/path/to/project",
  "data_directory": "/path/to/data",
  "parser": {
    "supported_languages": {
      "python": [".py"],
      "javascript": [".js", ".jsx"],
      "typescript": [".ts", ".tsx"]
    }
  },
  "vector": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_threshold": 0.7
  }
}
```

## 📚 Best Practices

### 1. Start Small
- Index 5-10 key files initially
- Test search functionality
- Gradually add more files

### 2. Use Descriptive Searches
- ✅ "user authentication and password validation"
- ❌ "auth stuff"

### 3. Combine Search Methods
- Use `find_symbol` for exact matches
- Use `semantic_search` for exploration
- Use `find_usages` for impact analysis

### 4. Regular Maintenance
- Re-index files after major changes
- Clean data directory occasionally
- Monitor index statistics

## 🔮 Future Features
- Automatic file watching and re-indexing
- Cross-language code search
- Code metrics and complexity analysis
- Integration with Git for change tracking
- Custom embedding models for domain-specific code

---

**Need help?** Check the logs, verify your configuration, and ensure all dependencies are installed. The server provides detailed error messages to help with troubleshooting.