# 📚 API Documentation

## Overview

This document provides comprehensive API documentation for all MCP tools provided by the Code Intelligence server.

## Tool Categories

1. **Search Tools** - Finding symbols, usages, and similar code
2. **Analysis Tools** - Impact analysis, dependencies, and patterns
3. **Index Tools** - Managing and updating the code index
4. **Visualization Tools** - Getting graph data for visualization

## 1. Search Tools

### find_usages

Find all places where a symbol (function, class, variable) is used.

**Parameters:**
- `name` (string, required): Name of the symbol to find usages for
- `file_path` (string, optional): Limit search to symbols in this file
- `include_indirect` (boolean, optional): Include indirect usages through call chains. Default: false
- `include_tests` (boolean, optional): Include usages in test files. Default: true
- `limit` (integer, optional): Maximum number of results. Default: 100

**Returns:**
```json
{
  "symbol": {
    "name": "process_data",
    "qualified_name": "utils.data.process_data",
    "type": "function",
    "file": "src/utils/data.py",
    "line": 45,
    "signature": "def process_data(input: dict) -> dict",
    "docstring": "Process input data and return results."
  },
  "usages": [
    {
      "id": 123,
      "name": "handle_request",
      "file_path": "src/api/handlers.py",
      "line_start": 67,
      "context": {
        "code": "    data = load_data(request)\n    result = process_data(data)\n    return jsonify(result)",
        "function": "handle_request",
        "class": "DataHandler"
      },
      "is_indirect": false
    }
  ],
  "grouped_by_file": {
    "src/api/handlers.py": [...],
    "src/cli/commands.py": [...]
  },
  "summary": {
    "total_usages": 5,
    "direct_usages": 3,
    "indirect_usages": 2,
    "files_affected": 3,
    "truncated": false
  }
}
```

**Example:**
```python
# Find all usages of a function
result = await find_usages(name="process_data")

# Find usages in a specific file
result = await find_usages(
    name="validate",
    file_path="src/validators.py"
)

# Include indirect usages
result = await find_usages(
    name="critical_function",
    include_indirect=True,
    include_tests=False
)
```

### find_similar_code

Find code snippets similar to a given example using semantic search.

**Parameters:**
- `code_snippet` (string, required): Code to find similar patterns for
- `limit` (integer, optional): Maximum results. Default: 10
- `language` (string, optional): Filter by programming language
- `min_similarity` (float, optional): Minimum similarity score (0-1). Default: 0.7

**Returns:**
```json
{
  "query": "def validate_email(email):\n    return '@' in email",
  "results": [
    {
      "id": "abc123",
      "code": "def validate_email_address(addr):\n    return re.match(r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', addr)",
      "metadata": {
        "file": "src/validators.py",
        "line_start": 23,
        "line_end": 25,
        "language": "python"
      },
      "similarity": 0.85,
      "symbols": ["validate_email_address"],
      "complexity": 2,
      "patterns": ["validation", "function_definition"]
    }
  ],
  "total_found": 3,
  "search_metadata": {
    "min_similarity": 0.7,
    "language_filter": null
  }
}
```

### search_by_pattern

Search code using tree-sitter query patterns.

**Parameters:**
- `pattern` (string, required): Tree-sitter query pattern
- `language` (string, optional): Language for the pattern
- `file_pattern` (string, optional): File name pattern to search
- `limit` (integer, optional): Maximum results. Default: 50

**Returns:**
```json
{
  "pattern": "(function_declaration name: (identifier) @name)",
  "results": [
    {
      "file": "src/utils.py",
      "language": "python",
      "match": {
        "name": "process_data",
        "line_start": 10,
        "line_end": 20
      },
      "context": "def process_data(input):\n    # Process the input\n    return result"
    }
  ],
  "files_searched": 150,
  "total_matches": 45
}
```

### find_definitions

Find where symbols are defined.

**Parameters:**
- `name` (string, required): Symbol name to find
- `symbol_type` (string, optional): Filter by type (function, class, variable)
- `exact_match` (boolean, optional): Require exact name match. Default: true

**Returns:**
```json
{
  "definitions": [
    {
      "name": "UserModel",
      "type": "class",
      "file": "src/models/user.py",
      "line": 15,
      "qualified_name": "models.user.UserModel"
    }
  ],
  "count": 1
}
```

## 2. Analysis Tools

### analyze_impact

Analyze the impact of changing a symbol.

**Parameters:**
- `name` (string, required): Name of the symbol to analyze
- `change_type` (string, optional): Type of change - "modify", "delete", "rename", "signature_change". Default: "modify"
- `include_tests` (boolean, optional): Include test files in analysis. Default: true
- `max_depth` (integer, optional): Maximum dependency depth. Default: 5

**Returns:**
```json
{
  "symbol": {
    "name": "critical_function",
    "type": "function",
    "file": "src/core/engine.py"
  },
  "change_type": "delete",
  "impact": {
    "directly_affected": [
      {
        "name": "process_request",
        "file": "src/api/handler.py",
        "line": 45
      }
    ],
    "transitively_affected": [...],
    "total_affected": 15,
    "files_affected": 8,
    "would_break": true
  },
  "risk_score": {
    "score": 0.85,
    "level": "high",
    "factors": {
      "change_type": "delete",
      "symbols_affected": 15,
      "files_affected": 8
    }
  },
  "recommendations": [
    "⚠️ High-risk change. Consider breaking into smaller changes.",
    "🔍 Review all usages before deletion.",
    "💡 Consider deprecation instead of immediate deletion."
  ]
}
```

### find_dependencies

Find what a symbol depends on or what depends on it.

**Parameters:**
- `name` (string, required): Symbol name
- `direction` (string, optional): "both", "depends_on", "depended_by". Default: "both"
- `max_depth` (integer, optional): Maximum traversal depth. Default: 3
- `include_types` (array, optional): Relationship types to include

**Returns:**
```json
{
  "symbol": {
    "name": "DataProcessor",
    "type": "class",
    "file": "src/processing/data.py"
  },
  "depends_on": {
    "all": [...],
    "by_depth": {
      "1": [
        {
          "name": "BaseProcessor",
          "type": "class",
          "relationship": "inherits"
        }
      ]
    },
    "direct": [...]
  },
  "depended_by": {
    "all": [...],
    "direct": [...]
  },
  "circular_dependencies": [],
  "metrics": {
    "fan_in": 5,
    "fan_out": 3,
    "instability": 0.375
  }
}
```

### detect_patterns

Detect design patterns and anti-patterns in the codebase.

**Parameters:**
- `scope` (string, optional): "project", "module", "file". Default: "project"
- `pattern_types` (array, optional): Specific patterns to detect

**Returns:**
```json
{
  "design_patterns": {
    "singleton": [
      {
        "class": "ConfigManager",
        "file": "src/config/manager.py",
        "confidence": 0.95
      }
    ],
    "factory": [...],
    "observer": [...]
  },
  "anti_patterns": {
    "god_classes": [
      {
        "class": "SuperManager",
        "file": "src/legacy/manager.py",
        "method_count": 47,
        "line_count": 2500
      }
    ],
    "circular_dependencies": [...],
    "duplicate_code": [...]
  },
  "summary": {
    "patterns_found": 8,
    "anti_patterns_found": 3,
    "recommendations": [...]
  }
}
```

### find_dead_code

Find potentially unused code.

**Parameters:**
- `scope` (string, optional): Scope of analysis
- `include_private` (boolean, optional): Include private methods/functions. Default: false

**Returns:**
```json
{
  "dead_code": [
    {
      "symbol": "unused_helper",
      "type": "function",
      "file": "src/utils/helpers.py",
      "line": 234,
      "reason": "No references found"
    }
  ],
  "total_found": 12,
  "by_type": {
    "function": 8,
    "class": 2,
    "variable": 2
  }
}
```

## 3. Index Tools

### update_index

Update the code index.

**Parameters:**
- `paths` (array, optional): Specific paths to update. None for smart update
- `full_rebuild` (boolean, optional): Force complete rebuild. Default: false
- `watch` (boolean, optional): Start watching for changes. Default: true

**Returns:**
```json
{
  "operation": "incremental",
  "stats": {
    "files_processed": 25,
    "symbols_extracted": 150,
    "relationships_found": 320,
    "errors": [],
    "duration_seconds": 3.5,
    "files_per_second": 7.14,
    "symbols_per_second": 42.86
  },
  "watching": true,
  "timestamp": "2024-03-15T10:30:00Z"
}
```

### get_index_stats

Get current index statistics.

**Parameters:**
- `detailed` (boolean, optional): Include detailed metrics. Default: false

**Returns:**
```json
{
  "symbols": {
    "total": 5420,
    "by_type": {
      "function": 2100,
      "class": 450,
      "method": 1800,
      "variable": 1070
    },
    "unique_names": 3200
  },
  "relationships": {
    "total": 12500,
    "by_type": {
      "calls": 8000,
      "imports": 2500,
      "inherits": 500,
      "uses": 1500
    }
  },
  "files": {
    "total": 350,
    "by_language": {
      "python": 200,
      "javascript": 100,
      "typescript": 50
    }
  },
  "storage": {
    "graph_size_mb": 45.2,
    "vector_size_mb": 120.5,
    "total_size_mb": 165.7
  }
}
```

### validate_index

Validate index integrity and optionally fix issues.

**Parameters:**
- `fix_issues` (boolean, optional): Automatically fix found issues. Default: false

**Returns:**
```json
{
  "valid": false,
  "issues": [
    {
      "type": "orphaned_symbols",
      "count": 5,
      "severity": "medium",
      "description": "Symbols without valid file references"
    },
    {
      "type": "stale_files",
      "count": 3,
      "severity": "high",
      "description": "Files changed since last index"
    }
  ],
  "fixed": true,
  "recommendations": [
    "Run incremental index update to refresh stale files"
  ]
}
```

## 4. Visualization Tools

### get_call_graph

Get call graph data for visualization.

**Parameters:**
- `name` (string, required): Function name to start from
- `max_depth` (integer, optional): Maximum depth. Default: 3
- `include_external` (boolean, optional): Include external calls. Default: false
- `layout` (string, optional): Layout algorithm - "hierarchical", "force", "circular". Default: "hierarchical"

**Returns:**
```json
{
  "root": {
    "id": 1,
    "name": "main",
    "file": "src/main.py"
  },
  "nodes": [
    {
      "id": 1,
      "name": "main",
      "type": "function",
      "file": "src/main.py",
      "line": 10,
      "x": 0,
      "y": 0,
      "is_root": true
    },
    {
      "id": 2,
      "name": "process_data",
      "type": "function",
      "file": "src/processor.py",
      "line": 25,
      "x": -100,
      "y": 100
    }
  ],
  "edges": [
    {
      "from": 1,
      "to": 2,
      "type": "calls",
      "depth": 1
    }
  ],
  "stats": {
    "total_nodes": 15,
    "total_edges": 18,
    "max_depth_reached": 3
  }
}
```

### get_dependency_graph

Get dependency graph for files or modules.

**Parameters:**
- `scope` (string, optional): "file", "module", "project". Default: "file"
- `file_path` (string, optional): For file scope
- `module_path` (string, optional): For module scope
- `max_nodes` (integer, optional): Limit nodes. Default: 50

**Returns:**
```json
{
  "nodes": [
    {
      "id": "src/main.py",
      "name": "main.py",
      "type": "file",
      "is_root": true
    }
  ],
  "edges": [
    {
      "from": "src/main.py",
      "to": "src/utils.py",
      "type": "imports"
    }
  ],
  "scope": "file",
  "root": "src/main.py"
}
```

## Error Handling

All tools follow a consistent error format:

```json
{
  "error": "Symbol 'nonexistent' not found",
  "error_code": "SYMBOL_NOT_FOUND",
  "details": {
    "symbol_name": "nonexistent",
    "suggestions": ["existing_func", "another_func"]
  }
}
```

Common error codes:
- `SYMBOL_NOT_FOUND` - Requested symbol doesn't exist
- `INVALID_PATTERN` - Invalid tree-sitter pattern
- `INDEX_NOT_READY` - Index not built or corrupted
- `INTERNAL_ERROR` - Unexpected server error

## Rate Limiting

The server implements rate limiting for expensive operations:

- Index updates: 1 per minute
- Full rebuilds: 1 per hour
- Search operations: 100 per minute
- Analysis operations: 50 per minute

## Best Practices

1. **Use specific queries** - More specific queries return faster and more relevant results
2. **Limit results** - Use the `limit` parameter to avoid overwhelming responses
3. **Cache results** - Results are cached for 5 minutes by default
4. **Batch operations** - Use batch APIs when available
5. **Monitor index health** - Regularly check index statistics and validity

## Examples

### Complete Workflow Example

```python
# 1. Update the index
await update_index(paths=["src/"])

# 2. Find a function
symbols = await find_definitions(name="process_data")

# 3. Analyze its impact
if symbols:
    impact = await analyze_impact(
        name=symbols[0]["name"],
        change_type="signature_change"
    )
    
    # 4. Find its dependencies
    deps = await find_dependencies(
        name=symbols[0]["name"],
        direction="both"
    )
    
    # 5. Get visualization data
    graph = await get_call_graph(
        name=symbols[0]["name"],
        max_depth=2
    )
```

### Error Handling Example

```python
try:
    result = await find_usages(name="some_function")
    
    if "error" in result:
        if result["error_code"] == "SYMBOL_NOT_FOUND":
            # Try suggestions
            suggestions = result["details"]["suggestions"]
            if suggestions:
                result = await find_usages(name=suggestions[0])
except Exception as e:
    print(f"Unexpected error: {e}")
```

This comprehensive API documentation covers all available MCP tools and their usage patterns.
