# 🧪 Testing Strategy Guide

## Overview

This guide outlines the comprehensive testing strategy for the Code Intelligence MCP server. All components should have thorough test coverage following these patterns.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_graph/
│   ├── test_vector/
│   ├── test_parser/
│   ├── test_tools/
│   └── test_indexing/
├── integration/             # Integration tests
│   ├── test_full_indexing.py
│   ├── test_search_flow.py
│   └── test_mcp_server.py
├── performance/             # Performance tests
│   ├── test_large_codebase.py
│   └── test_query_performance.py
└── fixtures/                # Test data
    ├── sample_projects/
    └── mock_data/
```

## 1. Test Fixtures and Mocks

### Core Test Fixtures

```python
# tests/conftest.py
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import tempfile
import shutil

from code_intelligence_mcp.di.container import DIContainer
from code_intelligence_mcp.config import Config
from code_intelligence_mcp.graph import GraphService
from code_intelligence_mcp.vector import VectorService
from code_intelligence_mcp.parser import ParserService
from code_intelligence_mcp.cache import CacheService


# Event loop fixture
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Configuration fixtures
@pytest.fixture
def test_config(tmp_path):
    """Test configuration with temp directories."""
    return Config(
        project_root=tmp_path / "project",
        data_directory=tmp_path / "data",
        graph=GraphConfig(
            db_path=tmp_path / "data" / "test.duckdb"
        ),
        vector=VectorConfig(
            persist_directory=tmp_path / "data" / "chroma"
        ),
        cache=CacheConfig(
            cache_directory=tmp_path / "data" / "cache"
        )
    )


@pytest.fixture
async def test_container(test_config):
    """Test DI container with real services."""
    from code_intelligence_mcp.di.bootstrap import create_container
    
    container = await create_container(test_config)
    yield container
    await container.shutdown()


# Mock fixtures
@pytest.fixture
def mock_graph_service():
    """Mock GraphService for unit tests."""
    mock = AsyncMock(spec=GraphService)
    
    # Setup default behaviors
    mock.find_symbol.return_value = []
    mock.find_usages.return_value = []
    mock.get_statistics.return_value = {
        'total_symbols': 0,
        'total_relationships': 0,
        'total_files': 0
    }
    
    return mock


@pytest.fixture
def mock_vector_service():
    """Mock VectorService for unit tests."""
    mock = AsyncMock(spec=VectorService)
    
    mock.search_similar_code.return_value = []
    mock.find_similar_symbols.return_value = []
    
    return mock


@pytest.fixture
def mock_parser_service():
    """Mock ParserService for unit tests."""
    mock = AsyncMock(spec=ParserService)
    
    mock.analyze_file.return_value = ([], [])
    mock.get_context_lines.return_value = []
    
    return mock


# Sample data fixtures
@pytest.fixture
def sample_symbol():
    """Sample symbol data."""
    return {
        'id': 1,
        'name': 'process_data',
        'qualified_name': 'module.process_data',
        'symbol_type': 'function',
        'file_path': 'src/processor.py',
        'line_start': 10,
        'line_end': 20,
        'signature': 'def process_data(input: str) -> dict',
        'docstring': 'Process input data and return results.'
    }


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def process_data(input_str: str) -> dict:
    """Process input data and return results."""
    data = json.loads(input_str)
    result = transform(data)
    return result

class DataProcessor:
    def __init__(self):
        self.cache = {}
    
    def process(self, data):
        if data in self.cache:
            return self.cache[data]
        result = self._process_internal(data)
        self.cache[data] = result
        return result
'''


# Test project fixtures
@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project structure."""
    project_root = tmp_path / "sample_project"
    
    # Create directory structure
    (project_root / "src").mkdir(parents=True)
    (project_root / "tests").mkdir(parents=True)
    (project_root / "src" / "utils").mkdir(parents=True)
    
    # Create sample files
    files = {
        "src/main.py": '''
from src.utils.helpers import process_data
from src.models import UserModel

def main():
    data = load_data()
    result = process_data(data)
    user = UserModel(result)
    return user
''',
        "src/utils/helpers.py": '''
def process_data(data):
    """Process data and return results."""
    return {"processed": data}

def validate_data(data):
    """Validate input data."""
    if not data:
        raise ValueError("Empty data")
    return True
''',
        "src/models.py": '''
class UserModel:
    def __init__(self, data):
        self.data = data
    
    def save(self):
        """Save user to database."""
        pass
''',
        "tests/test_main.py": '''
from src.main import main
from src.utils.helpers import process_data

def test_main():
    result = main()
    assert result is not None

def test_process_data():
    result = process_data({"test": 123})
    assert result["processed"]["test"] == 123
'''
    }
    
    for file_path, content in files.items():
        full_path = project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    yield project_root
    
    # Cleanup
    shutil.rmtree(project_root)


# Database fixtures
@pytest.fixture
async def populated_graph(test_container, sample_project):
    """Graph database populated with sample data."""
    graph = await test_container.get(GraphService)
    indexing = await test_container.get(IndexingService)
    
    # Index the sample project
    await indexing.index_project(sample_project)
    
    return graph
```

## 2. Unit Test Patterns

### Testing Services

```python
# tests/unit/test_graph/test_graph_service.py
import pytest
from code_intelligence_mcp.graph import GraphService


class TestGraphService:
    """Test GraphService functionality."""
    
    @pytest.mark.asyncio
    async def test_add_symbol(self, test_container):
        """Test adding a symbol."""
        graph = await test_container.get(GraphService)
        
        # Add symbol
        symbol_id = await graph.add_symbol(
            name="test_func",
            qualified_name="module.test_func",
            symbol_type="function",
            file_path="test.py",
            line_start=1,
            line_end=5,
            language="python",
            ast_hash="abc123"
        )
        
        assert symbol_id > 0
        
        # Verify it was added
        symbols = await graph.find_symbol(name="test_func")
        assert len(symbols) == 1
        assert symbols[0]['name'] == "test_func"
    
    @pytest.mark.asyncio
    async def test_find_usages(self, populated_graph):
        """Test finding symbol usages."""
        # Find the process_data function
        symbols = await populated_graph.find_symbol(name="process_data")
        assert len(symbols) > 0
        
        # Find its usages
        usages = await populated_graph.find_usages(symbols[0]['id'])
        
        # Should find usage in main.py
        assert any(u['file_path'].endswith('main.py') for u in usages)
    
    @pytest.mark.asyncio
    async def test_circular_dependencies(self, test_container):
        """Test circular dependency detection."""
        graph = await test_container.get(GraphService)
        
        # Create circular dependency
        id1 = await graph.add_symbol(
            name="func1",
            qualified_name="func1",
            symbol_type="function",
            file_path="file1.py",
            line_start=1,
            line_end=5,
            language="python",
            ast_hash="hash1"
        )
        
        id2 = await graph.add_symbol(
            name="func2",
            qualified_name="func2",
            symbol_type="function",
            file_path="file2.py",
            line_start=1,
            line_end=5,
            language="python",
            ast_hash="hash2"
        )
        
        # Create circular relationship
        await graph.add_relationship(
            from_symbol_id=id1,
            to_symbol_id=id2,
            relationship_type="calls",
            file_path="file1.py",
            line_number=3
        )
        
        await graph.add_relationship(
            from_symbol_id=id2,
            to_symbol_id=id1,
            relationship_type="calls",
            file_path="file2.py",
            line_number=3
        )
        
        # Detect cycles
        cycles = await graph.find_circular_dependencies()
        assert len(cycles) > 0
```

### Testing Tools

```python
# tests/unit/test_tools/test_search_tools.py
import pytest
from unittest.mock import AsyncMock
from code_intelligence_mcp.tools.search_tools import FindUsagesTool


class TestFindUsagesTool:
    """Test FindUsagesTool."""
    
    @pytest.fixture
    def tool(self, mock_graph_service, mock_vector_service, mock_parser_service):
        """Create tool with mocked dependencies."""
        return FindUsagesTool(
            graph=mock_graph_service,
            vector=mock_vector_service,
            parser=mock_parser_service,
            cache=None
        )
    
    @pytest.mark.asyncio
    async def test_find_usages_success(self, tool, sample_symbol):
        """Test successful usage finding."""
        # Setup mocks
        tool.graph.find_symbol.return_value = [sample_symbol]
        tool.graph.find_usages.return_value = [
            {
                'id': 2,
                'name': 'caller_function',
                'file_path': 'caller.py',
                'line_start': 15
            }
        ]
        
        tool.parser.get_context_lines.return_value = [
            "def caller_function():",
            "    data = get_data()",
            "    result = process_data(data)",  # Usage line
            "    return result"
        ]
        
        # Execute
        result = await tool.execute(name="process_data")
        
        # Verify
        assert 'error' not in result
        assert result['symbol']['name'] == 'process_data'
        assert len(result['usages']) == 1
        assert result['usages'][0]['name'] == 'caller_function'
        assert 'context' in result['usages'][0]
    
    @pytest.mark.asyncio
    async def test_find_usages_not_found(self, tool):
        """Test when symbol is not found."""
        # Setup mocks
        tool.graph.find_symbol.return_value = []
        tool.vector.find_similar_symbols.return_value = [
            {'name': 'process_data_v2', 'type': 'function', 'similarity': 0.85}
        ]
        
        # Execute
        result = await tool.execute(name="process_data")
        
        # Verify
        assert 'error' in result
        assert 'suggestions' in result
        assert len(result['suggestions']) > 0
    
    @pytest.mark.asyncio
    async def test_find_usages_multiple_matches(self, tool):
        """Test when multiple symbols match."""
        # Setup mocks
        tool.graph.find_symbol.return_value = [
            {'name': 'process_data', 'file_path': 'file1.py'},
            {'name': 'process_data', 'file_path': 'file2.py'}
        ]
        
        # Execute without file_path
        result = await tool.execute(name="process_data")
        
        # Verify
        assert result.get('multiple_matches') is True
        assert len(result['symbols']) == 2
    
    @pytest.mark.asyncio
    async def test_input_validation(self, tool):
        """Test input validation."""
        # Test empty name
        with pytest.raises(ValueError):
            await tool.validate_inputs(name="")
        
        # Test invalid type
        with pytest.raises(ValueError):
            await tool.validate_inputs(name=123)
```

### Testing Parser

```python
# tests/unit/test_parser/test_python_extractor.py
import pytest
import tree_sitter
from pathlib import Path
from code_intelligence_mcp.parser.extractors import PythonExtractor


class TestPythonExtractor:
    """Test Python code extraction."""
    
    @pytest.fixture
    def extractor(self, mock_parser_service):
        """Create Python extractor."""
        return PythonExtractor(mock_parser_service)
    
    @pytest.mark.asyncio
    async def test_extract_function(self, extractor, sample_python_code):
        """Test extracting functions from Python code."""
        # Parse code
        parser = tree_sitter.Parser()
        parser.set_language(tree_sitter_python.language())
        tree = parser.parse(sample_python_code.encode())
        
        # Extract symbols
        symbols, relationships = await extractor.extract(
            tree,
            Path("test.py"),
            sample_python_code.encode()
        )
        
        # Verify functions found
        function_names = [s.name for s in symbols if s.symbol_type == 'function']
        assert 'process_data' in function_names
        
        # Verify function details
        process_func = next(s for s in symbols if s.name == 'process_data')
        assert process_func.signature == 'def process_data(input_str: str) -> dict'
        assert process_func.docstring == 'Process input data and return results.'
    
    @pytest.mark.asyncio
    async def test_extract_class(self, extractor, sample_python_code):
        """Test extracting classes from Python code."""
        tree = self._parse_python(sample_python_code)
        
        symbols, _ = await extractor.extract(
            tree,
            Path("test.py"),
            sample_python_code.encode()
        )
        
        # Verify class found
        classes = [s for s in symbols if s.symbol_type == 'class']
        assert len(classes) == 1
        assert classes[0].name == 'DataProcessor'
        
        # Verify methods found
        methods = [s for s in symbols if s.symbol_type == 'method']
        assert any(m.name == '__init__' for m in methods)
        assert any(m.name == 'process' for m in methods)
    
    @pytest.mark.asyncio
    async def test_extract_relationships(self, extractor):
        """Test extracting function calls."""
        code = '''
def caller():
    result = process_data(input)
    validate_data(result)
    return result

def process_data(data):
    return transform(data)
'''
        
        tree = self._parse_python(code)
        symbols, relationships = await extractor.extract(
            tree,
            Path("test.py"),
            code.encode()
        )
        
        # Verify relationships found
        assert len(relationships) > 0
        
        # Check specific calls
        calls = [r for r in relationships if r.relationship_type == 'calls']
        assert any(
            r.from_symbol == 'test.caller' and 
            r.to_symbol == 'test.process_data'
            for r in calls
        )
```

## 3. Integration Tests

### Full Indexing Test

```python
# tests/integration/test_full_indexing.py
import pytest
from pathlib import Path


class TestFullIndexing:
    """Test complete indexing workflow."""
    
    @pytest.mark.asyncio
    async def test_index_project(self, test_container, sample_project):
        """Test indexing an entire project."""
        indexing = await test_container.get(IndexingService)
        
        # Index the project
        stats = await indexing.index_project(sample_project)
        
        # Verify stats
        assert stats['files_processed'] > 0
        assert stats['symbols_extracted'] > 0
        assert stats['relationships_found'] > 0
        assert len(stats['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_incremental_update(self, test_container, populated_graph, sample_project):
        """Test incremental indexing."""
        indexing = await test_container.get(IndexingService)
        
        # Modify a file
        file_path = sample_project / "src" / "utils" / "helpers.py"
        original_content = file_path.read_text()
        
        new_content = original_content + '''

def new_function():
    """Newly added function."""
    return 42
'''
        file_path.write_text(new_content)
        
        # Update index
        stats = await indexing.update_files([file_path])
        
        # Verify new symbol was added
        graph = await test_container.get(GraphService)
        symbols = await graph.find_symbol(name="new_function")
        assert len(symbols) == 1
    
    @pytest.mark.asyncio
    async def test_file_deletion_handling(self, test_container, populated_graph, sample_project):
        """Test handling file deletion."""
        graph = await test_container.get(GraphService)
        indexing = await test_container.get(IndexingService)
        
        # Get initial symbol count
        initial_stats = await graph.get_statistics()
        initial_count = initial_stats['total_symbols']
        
        # Delete a file
        file_to_delete = sample_project / "src" / "models.py"
        file_to_delete.unlink()
        
        # Update index
        await indexing.handle_file_deletion(str(file_to_delete))
        
        # Verify symbols were removed
        final_stats = await graph.get_statistics()
        assert final_stats['total_symbols'] < initial_count
        
        # Verify specific symbol removed
        symbols = await graph.find_symbol(name="UserModel")
        assert len(symbols) == 0
```

### Search Flow Integration

```python
# tests/integration/test_search_flow.py
import pytest


class TestSearchFlow:
    """Test complete search workflows."""
    
    @pytest.mark.asyncio
    async def test_find_usages_flow(self, test_container, populated_graph):
        """Test complete find usages flow."""
        # Get tools
        search_tools = await test_container.get(SearchTools)
        
        # Find usages of process_data
        result = await search_tools.find_usages.execute(
            name="process_data"
        )
        
        # Verify results
        assert 'error' not in result
        assert result['symbol']['name'] == 'process_data'
        assert len(result['usages']) > 0
        
        # Verify context is included
        for usage in result['usages']:
            assert 'context' in usage
            assert 'code' in usage['context']
    
    @pytest.mark.asyncio
    async def test_semantic_search_flow(self, test_container, populated_graph):
        """Test semantic code search."""
        search_tools = await test_container.get(SearchTools)
        
        # Search for similar code
        code_snippet = '''
def validate_input(data):
    if not data:
        raise ValueError("Invalid data")
    return True
'''
        
        result = await search_tools.find_similar_code.execute(
            code_snippet=code_snippet,
            limit=5
        )
        
        # Should find validate_data function
        assert len(result['results']) > 0
        
        # Check similarity scores
        for res in result['results']:
            assert 'similarity' in res
            assert res['similarity'] > 0.5
```

## 4. Performance Tests

### Large Codebase Performance

```python
# tests/performance/test_large_codebase.py
import pytest
import time
from pathlib import Path


class TestLargeCodebasePerformance:
    """Test performance with large codebases."""
    
    @pytest.fixture
    def large_project(self, tmp_path):
        """Generate a large test project."""
        project_root = tmp_path / "large_project"
        
        # Generate 1000 Python files
        for i in range(100):
            module_dir = project_root / f"module_{i}"
            module_dir.mkdir(parents=True, exist_ok=True)
            
            for j in range(10):
                file_path = module_dir / f"file_{j}.py"
                content = self._generate_python_file(i, j)
                file_path.write_text(content)
        
        return project_root
    
    def _generate_python_file(self, module_idx: int, file_idx: int) -> str:
        """Generate Python file content."""
        return f'''
# Module {module_idx}, File {file_idx}

class Class_{module_idx}_{file_idx}:
    def __init__(self):
        self.value = {module_idx * 10 + file_idx}
    
    def method_1(self):
        return self.value * 2
    
    def method_2(self, param):
        result = self.method_1()
        return result + param

def function_{module_idx}_{file_idx}(input_data):
    """Process function for module {module_idx}."""
    obj = Class_{module_idx}_{file_idx}()
    return obj.method_2(input_data)

def helper_function():
    """Helper function."""
    return 42
'''
    
    @pytest.mark.asyncio
    async def test_indexing_performance(self, test_container, large_project):
        """Test indexing performance on large codebase."""
        indexing = await test_container.get(IndexingService)
        
        start_time = time.time()
        stats = await indexing.index_project(large_project)
        duration = time.time() - start_time
        
        # Performance assertions
        files_per_second = stats['files_processed'] / duration
        assert files_per_second > 10  # Should process >10 files/sec
        
        print(f"\nIndexing Performance:")
        print(f"  Files: {stats['files_processed']}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Files/sec: {files_per_second:.2f}")
        print(f"  Symbols: {stats['symbols_extracted']}")
    
    @pytest.mark.asyncio
    async def test_query_performance(self, test_container, populated_graph):
        """Test query performance."""
        graph = await test_container.get(GraphService)
        
        # Test symbol search performance
        start_time = time.time()
        for i in range(100):
            await graph.find_symbol(name=f"function_{i}_0")
        symbol_search_time = (time.time() - start_time) / 100
        
        assert symbol_search_time < 0.01  # <10ms per search
        
        # Test usage finding performance
        symbols = await graph.find_symbol(name="function_0_0")
        if symbols:
            start_time = time.time()
            await graph.find_usages(symbols[0]['id'])
            usage_time = time.time() - start_time
            
            assert usage_time < 0.1  # <100ms
```

## 5. Test Data Management

### Test Data Builder

```python
# tests/fixtures/builders.py
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class SymbolBuilder:
    """Builder for test symbols."""
    name: str = "test_symbol"
    qualified_name: str = ""
    symbol_type: str = "function"
    file_path: str = "test.py"
    line_start: int = 1
    line_end: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.qualified_name:
            self.qualified_name = f"module.{self.name}"
    
    def with_name(self, name: str) -> 'SymbolBuilder':
        self.name = name
        return self
    
    def with_type(self, symbol_type: str) -> 'SymbolBuilder':
        self.symbol_type = symbol_type
        return self
    
    def build(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'qualified_name': self.qualified_name,
            'symbol_type': self.symbol_type,
            'file_path': self.file_path,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'metadata': self.metadata
        }


@dataclass
class ProjectBuilder:
    """Builder for test projects."""
    root_path: Path
    files: Dict[str, str] = field(default_factory=dict)
    
    def add_file(self, path: str, content: str) -> 'ProjectBuilder':
        self.files[path] = content
        return self
    
    def add_python_module(self, name: str) -> 'ProjectBuilder':
        content = f'''
"""Module {name}"""

def {name}_function():
    """Function in {name} module."""
    return "{name}"

class {name.title()}Class:
    """Class in {name} module."""
    pass
'''
        self.files[f"{name}.py"] = content
        return self
    
    def build(self) -> Path:
        """Build the project structure."""
        for file_path, content in self.files.items():
            full_path = self.root_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        return self.root_path
```

## 6. Mock Strategies

### Smart Mocks

```python
# tests/mocks/smart_mocks.py
from unittest.mock import AsyncMock
from typing import Dict, List, Any


class SmartGraphMock(AsyncMock):
    """Smart mock for GraphService with state."""
    
    def __init__(self):
        super().__init__()
        self._symbols: Dict[int, Dict[str, Any]] = {}
        self._relationships: List[Dict[str, Any]] = []
        self._next_id = 1
        
        # Wire up methods
        self.add_symbol.side_effect = self._add_symbol
        self.find_symbol.side_effect = self._find_symbol
        self.find_usages.side_effect = self._find_usages
    
    async def _add_symbol(self, **kwargs) -> int:
        """Add symbol to mock state."""
        symbol_id = self._next_id
        self._next_id += 1
        
        self._symbols[symbol_id] = {
            'id': symbol_id,
            **kwargs
        }
        
        return symbol_id
    
    async def _find_symbol(self, name: str = None, **kwargs) -> List[Dict[str, Any]]:
        """Find symbol in mock state."""
        results = []
        
        for symbol in self._symbols.values():
            if name and symbol.get('name') == name:
                results.append(symbol)
        
        return results
    
    async def _find_usages(self, symbol_id: int) -> List[Dict[str, Any]]:
        """Find usages in mock state."""
        usages = []
        
        for rel in self._relationships:
            if rel['to_symbol_id'] == symbol_id:
                from_symbol = self._symbols.get(rel['from_symbol_id'])
                if from_symbol:
                    usages.append(from_symbol)
        
        return usages
```

## 7. Testing Best Practices

### 1. Test Organization

```python
# Good: Clear test class and method names
class TestSymbolExtraction:
    """Test symbol extraction from source code."""
    
    async def test_extracts_function_with_docstring(self):
        """Test that functions with docstrings are properly extracted."""
        # Test implementation
        
    async def test_handles_nested_classes(self):
        """Test extraction of nested class definitions."""
        # Test implementation
```

### 2. Async Testing

```python
# Always use pytest.mark.asyncio for async tests
@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_function()
    assert result is not None

# Use async fixtures
@pytest.fixture
async def async_resource():
    resource = await create_resource()
    yield resource
    await cleanup_resource(resource)
```

### 3. Assertions

```python
# Use specific assertions
assert result['name'] == 'expected_name'  # Good
assert result  # Too vague

# Use pytest features
with pytest.raises(ValueError, match="Invalid input"):
    await process_invalid_data()

# Use approx for floating point
assert result['score'] == pytest.approx(0.85, rel=0.01)
```

### 4. Test Data

```python
# Use fixtures for test data
@pytest.fixture
def valid_symbol_data():
    return {
        'name': 'test_func',
        'type': 'function',
        # ... complete data
    }

# Use parametrize for multiple cases
@pytest.mark.parametrize("input,expected", [
    ("valid_name", True),
    ("", False),
    ("_private", True),
    ("123invalid", False),
])
async def test_name_validation(input, expected):
    assert is_valid_name(input) == expected
```

## 8. Coverage Requirements

### Minimum Coverage Targets

- Overall: 90%
- Core services: 95%
- Tools: 90%
- Utilities: 85%

### Coverage Configuration

```ini
# .coveragerc
[run]
source = code_intelligence_mcp
omit = 
    */tests/*
    */migrations/*
    */config.py

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

### Running Coverage

```bash
# Run tests with coverage
pytest --cov=code_intelligence_mcp --cov-report=html --cov-report=term

# Run specific test suite
pytest tests/unit --cov=code_intelligence_mcp.tools

# Generate coverage report
coverage html
```

## 9. Continuous Testing

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/unit -v
        language: system
        pass_filenames: false
        always_run: true
```

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest --cov=code_intelligence_mcp --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

This comprehensive testing strategy ensures high quality and reliability for the Code Intelligence MCP server.
