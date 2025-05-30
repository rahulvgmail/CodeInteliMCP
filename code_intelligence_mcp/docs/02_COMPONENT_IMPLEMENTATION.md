# 🔧 Component Implementation Guide

## Overview

This guide provides detailed implementation instructions for each component of the Code Intelligence MCP server. Follow these in order for best results.

## 1. Parser Service Implementation

### Language Registry

```python
# code_intelligence_mcp/parser/language_registry.py
from typing import Dict, List, Optional, Any
from pathlib import Path
import tree_sitter
import logging
from ..core.service import BaseService


class LanguageRegistry(BaseService):
    """Registry for tree-sitter language parsers."""
    
    def __init__(self, language_config: Dict[str, List[str]]):
        super().__init__()
        self.language_config = language_config
        self.parsers: Dict[str, tree_sitter.Parser] = {}
        self.languages: Dict[str, tree_sitter.Language] = {}
        self._extension_map: Dict[str, str] = {}
    
    async def _initialize(self) -> None:
        """Load all language parsers."""
        import tree_sitter_python
        import tree_sitter_javascript
        import tree_sitter_typescript
        import tree_sitter_go
        import tree_sitter_rust
        
        # Map languages to their tree-sitter modules
        language_modules = {
            'python': tree_sitter_python,
            'javascript': tree_sitter_javascript,
            'typescript': tree_sitter_typescript,
            'go': tree_sitter_go,
            'rust': tree_sitter_rust,
        }
        
        for lang, module in language_modules.items():
            if lang in self.language_config:
                try:
                    # Get the language object
                    language = module.language()
                    self.languages[lang] = language
                    
                    # Create parser
                    parser = tree_sitter.Parser()
                    parser.set_language(language)
                    self.parsers[lang] = parser
                    
                    # Map extensions
                    for ext in self.language_config[lang]:
                        self._extension_map[ext] = lang
                    
                    self._logger.info(f"Loaded {lang} parser")
                except Exception as e:
                    self._logger.error(f"Failed to load {lang} parser: {e}")
    
    def get_language_for_file(self, file_path: Path) -> Optional[str]:
        """Get language for a file based on extension."""
        ext = file_path.suffix.lower()
        return self._extension_map.get(ext)
    
    def get_parser(self, language: str) -> Optional[tree_sitter.Parser]:
        """Get parser for a language."""
        return self.parsers.get(language)
    
    def is_supported(self, file_path: Path) -> bool:
        """Check if file type is supported."""
        return self.get_language_for_file(file_path) is not None
```

### AST Analyzer

```python
# code_intelligence_mcp/parser/ast_analyzer.py
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tree_sitter
from dataclasses import dataclass
import hashlib
from ..core.service import BaseService


@dataclass
class Symbol:
    """Represents a code symbol."""
    name: str
    qualified_name: str
    symbol_type: str
    file_path: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    visibility: str = 'public'
    is_async: bool = False
    is_generator: bool = False
    parent_id: Optional[int] = None
    metadata: Dict[str, Any] = None


@dataclass
class Relationship:
    """Represents a relationship between symbols."""
    from_symbol: str  # qualified name
    to_symbol: str    # qualified name
    relationship_type: str
    file_path: str
    line_number: int
    column_number: int
    confidence: float = 1.0
    context: Optional[str] = None


class ASTAnalyzer(BaseService):
    """Analyzes AST to extract symbols and relationships."""
    
    def __init__(self, language_registry: 'LanguageRegistry'):
        super().__init__()
        self.language_registry = language_registry
        self._query_cache: Dict[str, tree_sitter.Query] = {}
    
    async def analyze_file(
        self,
        file_path: Path,
        content: bytes
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Analyze a file and extract symbols and relationships."""
        language = self.language_registry.get_language_for_file(file_path)
        if not language:
            return [], []
        
        parser = self.language_registry.get_parser(language)
        if not parser:
            return [], []
        
        # Parse the file
        tree = parser.parse(content)
        
        # Extract based on language
        extractor = self._get_extractor(language)
        symbols, relationships = await extractor.extract(
            tree,
            file_path,
            content
        )
        
        return symbols, relationships
    
    def _get_extractor(self, language: str) -> 'LanguageExtractor':
        """Get language-specific extractor."""
        extractors = {
            'python': PythonExtractor(self),
            'javascript': JavaScriptExtractor(self),
            'typescript': TypeScriptExtractor(self),
            'go': GoExtractor(self),
            'rust': RustExtractor(self),
        }
        return extractors.get(language, GenericExtractor(self))
    
    def _get_query(self, language: str, query_name: str) -> tree_sitter.Query:
        """Get or create a tree-sitter query."""
        cache_key = f"{language}:{query_name}"
        if cache_key not in self._query_cache:
            query_text = self._load_query(language, query_name)
            lang_obj = self.language_registry.languages[language]
            self._query_cache[cache_key] = lang_obj.query(query_text)
        return self._query_cache[cache_key]
    
    def _load_query(self, language: str, query_name: str) -> str:
        """Load query from file."""
        # Load from queries directory
        query_path = Path(__file__).parent / 'queries' / language / f"{query_name}.scm"
        if query_path.exists():
            return query_path.read_text()
        
        # Return default query
        return self._get_default_query(language, query_name)
    
    def _get_default_query(self, language: str, query_name: str) -> str:
        """Get default query for common patterns."""
        if query_name == 'symbols':
            if language == 'python':
                return '''
                (function_definition
                    name: (identifier) @function.name
                    parameters: (parameters) @function.params
                    body: (block) @function.body) @function
                
                (class_definition
                    name: (identifier) @class.name
                    body: (block) @class.body) @class
                '''
            # Add more languages
        
        return ""
```

### Language-Specific Extractors

```python
# code_intelligence_mcp/parser/extractors/python_extractor.py
from typing import List, Tuple, Optional
import tree_sitter
from pathlib import Path
from ..ast_analyzer import Symbol, Relationship, LanguageExtractor


class PythonExtractor(LanguageExtractor):
    """Python-specific symbol and relationship extractor."""
    
    async def extract(
        self,
        tree: tree_sitter.Tree,
        file_path: Path,
        content: bytes
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Extract symbols and relationships from Python AST."""
        symbols = []
        relationships = []
        
        # Track current context
        context = {
            'module': file_path.stem,
            'class_stack': [],
            'function_stack': [],
            'imports': {}
        }
        
        # First pass: collect imports
        await self._collect_imports(tree.root_node, content, context)
        
        # Second pass: extract symbols and relationships
        await self._visit_node(
            tree.root_node,
            content,
            file_path,
            context,
            symbols,
            relationships
        )
        
        return symbols, relationships
    
    async def _collect_imports(
        self,
        node: tree_sitter.Node,
        content: bytes,
        context: dict
    ):
        """Collect all imports for relationship resolution."""
        if node.type == 'import_statement':
            # import module
            for child in node.children:
                if child.type == 'dotted_name':
                    module_name = self._get_node_text(child, content)
                    context['imports'][module_name] = module_name
        
        elif node.type == 'import_from_statement':
            # from module import name
            module_node = node.child_by_field_name('module_name')
            if module_node:
                module_name = self._get_node_text(module_node, content)
                
                # Get imported names
                for child in node.children:
                    if child.type == 'dotted_name' and child != module_node:
                        name = self._get_node_text(child, content)
                        context['imports'][name] = f"{module_name}.{name}"
        
        # Recurse
        for child in node.children:
            await self._collect_imports(child, content, context)
    
    async def _visit_node(
        self,
        node: tree_sitter.Node,
        content: bytes,
        file_path: Path,
        context: dict,
        symbols: List[Symbol],
        relationships: List[Relationship]
    ):
        """Visit AST node and extract information."""
        if node.type == 'function_definition':
            await self._extract_function(
                node, content, file_path, context, symbols, relationships
            )
        
        elif node.type == 'class_definition':
            await self._extract_class(
                node, content, file_path, context, symbols, relationships
            )
        
        elif node.type == 'call':
            await self._extract_call(
                node, content, file_path, context, relationships
            )
        
        # Continue traversal for children
        for child in node.children:
            await self._visit_node(
                child, content, file_path, context, symbols, relationships
            )
    
    async def _extract_function(
        self,
        node: tree_sitter.Node,
        content: bytes,
        file_path: Path,
        context: dict,
        symbols: List[Symbol],
        relationships: List[Relationship]
    ):
        """Extract function symbol."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return
        
        name = self._get_node_text(name_node, content)
        
        # Build qualified name
        qualified_parts = [context['module']]
        qualified_parts.extend(context['class_stack'])
        qualified_parts.append(name)
        qualified_name = '.'.join(qualified_parts)
        
        # Get parameters
        params_node = node.child_by_field_name('parameters')
        signature = self._extract_signature(name, params_node, content)
        
        # Check decorators
        decorators = []
        is_async = False
        for child in node.children:
            if child.type == 'decorator':
                dec_name = self._get_decorator_name(child, content)
                decorators.append(dec_name)
                if dec_name == 'async':
                    is_async = True
        
        # Get docstring
        docstring = self._extract_docstring(node, content)
        
        # Determine visibility
        visibility = 'private' if name.startswith('_') else 'public'
        
        # Create symbol
        symbol = Symbol(
            name=name,
            qualified_name=qualified_name,
            symbol_type='method' if context['class_stack'] else 'function',
            file_path=str(file_path),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            column_start=node.start_point[1],
            column_end=node.end_point[1],
            signature=signature,
            docstring=docstring,
            visibility=visibility,
            is_async=is_async,
            metadata={'decorators': decorators}
        )
        
        symbols.append(symbol)
        
        # Update context for nested traversal
        context['function_stack'].append(name)
        
        # Process function body
        body_node = node.child_by_field_name('body')
        if body_node:
            for child in body_node.children:
                await self._visit_node(
                    child, content, file_path, context, symbols, relationships
                )
        
        # Restore context
        context['function_stack'].pop()
    
    def _extract_signature(
        self,
        name: str,
        params_node: Optional[tree_sitter.Node],
        content: bytes
    ) -> str:
        """Extract function signature."""
        if not params_node:
            return f"{name}()"
        
        params_text = self._get_node_text(params_node, content)
        return f"{name}{params_text}"
    
    def _extract_docstring(
        self,
        node: tree_sitter.Node,
        content: bytes
    ) -> Optional[str]:
        """Extract docstring from function or class."""
        body_node = node.child_by_field_name('body')
        if not body_node or not body_node.children:
            return None
        
        first_stmt = body_node.children[0]
        if first_stmt.type == 'expression_statement':
            expr = first_stmt.children[0] if first_stmt.children else None
            if expr and expr.type == 'string':
                # Remove quotes and clean up
                docstring = self._get_node_text(expr, content)
                return docstring.strip('"\'').strip()
        
        return None
    
    def _get_node_text(self, node: tree_sitter.Node, content: bytes) -> str:
        """Get text content of a node."""
        return content[node.start_byte:node.end_byte].decode('utf-8')
```

## 2. Vector Store Implementation

### ChromaDB Integration

```python
# code_intelligence_mcp/vector/chroma_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ..core.service import BaseService, CacheableService
from ..config import VectorConfig


class ChromaStore(CacheableService):
    """Vector store using ChromaDB for semantic search."""
    
    def __init__(
        self,
        persist_directory: Path,
        config: VectorConfig,
        cache_service: Optional['CacheService'] = None
    ):
        super().__init__(cache_service=cache_service, config=config)
        self.persist_directory = persist_directory
        self._client: Optional[chromadb.Client] = None
        self._collections: Dict[str, chromadb.Collection] = {}
        self._embedder: Optional[SentenceTransformer] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    async def _initialize(self) -> None:
        """Initialize ChromaDB and collections."""
        # Create ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Load embedding model
        self._embedder = SentenceTransformer(
            self.config.embedding_model
        )
        
        # Create collections
        await self._create_collections()
    
    async def _create_collections(self):
        """Create or get collections."""
        collection_configs = {
            'code_snippets': {
                'metadata': {'hnsw:space': 'cosine'},
                'embedding_function': None  # We'll handle embeddings
            },
            'symbols': {
                'metadata': {'hnsw:space': 'cosine'},
                'embedding_function': None
            },
            'documentation': {
                'metadata': {'hnsw:space': 'cosine'},
                'embedding_function': None
            }
        }
        
        for name, config in collection_configs.items():
            try:
                collection = self._client.get_collection(name)
            except:
                collection = self._client.create_collection(
                    name=name,
                    **config
                )
            self._collections[name] = collection
    
    async def add_code_snippet(
        self,
        code: str,
        metadata: Dict[str, Any],
        snippet_id: Optional[str] = None
    ) -> str:
        """Add a code snippet to the vector store."""
        # Generate embedding
        embedding = await self._generate_embedding(code)
        
        # Generate ID if not provided
        if not snippet_id:
            import uuid
            snippet_id = str(uuid.uuid4())
        
        # Add to collection
        def _add():
            self._collections['code_snippets'].add(
                embeddings=[embedding.tolist()],
                documents=[code],
                metadatas=[metadata],
                ids=[snippet_id]
            )
        
        await asyncio.get_event_loop().run_in_executor(
            self._executor, _add
        )
        
        return snippet_id
    
    async def add_symbol(
        self,
        symbol_data: Dict[str, Any]
    ) -> str:
        """Add a symbol to the vector store."""
        # Create searchable text
        text_parts = [
            symbol_data['name'],
            symbol_data.get('qualified_name', ''),
            symbol_data.get('signature', ''),
            symbol_data.get('docstring', '')
        ]
        text = ' '.join(filter(None, text_parts))
        
        # Generate embedding
        embedding = await self._generate_embedding(text)
        
        # Add to collection
        symbol_id = str(symbol_data.get('id', ''))
        
        def _add():
            self._collections['symbols'].add(
                embeddings=[embedding.tolist()],
                documents=[text],
                metadatas=[{
                    'name': symbol_data['name'],
                    'type': symbol_data['symbol_type'],
                    'file': symbol_data['file_path'],
                    'line': symbol_data['line_start']
                }],
                ids=[symbol_id]
            )
        
        await asyncio.get_event_loop().run_in_executor(
            self._executor, _add
        )
        
        return symbol_id
    
    async def search_similar_code(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar code snippets."""
        # Check cache
        cache_key = f"similar_code:{query}:{limit}:{filters}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Search
        def _search():
            results = self._collections['code_snippets'].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                where=filters
            )
            
            # Format results
            formatted = []
            for idx in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][idx],
                    'code': results['documents'][0][idx],
                    'metadata': results['metadatas'][0][idx],
                    'similarity': 1 - results['distances'][0][idx]
                })
            
            return formatted
        
        results = await asyncio.get_event_loop().run_in_executor(
            self._executor, _search
        )
        
        # Cache results
        await self._set_cached(cache_key, results, ttl=300)
        
        return results
    
    async def find_similar_symbols(
        self,
        name: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find symbols with similar names."""
        # Generate embedding for the name
        embedding = await self._generate_embedding(name)
        
        def _search():
            results = self._collections['symbols'].query(
                query_embeddings=[embedding.tolist()],
                n_results=limit
            )
            
            formatted = []
            for idx in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][idx],
                    'name': results['metadatas'][0][idx]['name'],
                    'type': results['metadatas'][0][idx]['type'],
                    'file': results['metadatas'][0][idx]['file'],
                    'similarity': 1 - results['distances'][0][idx]
                })
            
            return formatted
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _search
        )
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        def _embed():
            return self._embedder.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _embed
        )
    
    async def delete_by_file(self, file_path: str):
        """Delete all embeddings for a file."""
        def _delete():
            for collection in self._collections.values():
                collection.delete(
                    where={"file": file_path}
                )
        
        await asyncio.get_event_loop().run_in_executor(
            self._executor, _delete
        )
    
    async def _shutdown(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=True)
```

## 3. Indexing Service Implementation

### Core Indexing Service

```python
# code_intelligence_mcp/indexing/indexing_service.py
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import asyncio
from datetime import datetime
import hashlib
from ..core.service import BaseService
from ..graph import GraphService
from ..vector import VectorService
from ..parser import ParserService
from ..cache import CacheService
from ..config import IndexingConfig


class IndexingService(BaseService):
    """Service for indexing code into graph and vector stores."""
    
    def __init__(
        self,
        graph: GraphService,
        vector: VectorService,
        parser: ParserService,
        cache: CacheService,
        config: IndexingConfig
    ):
        super().__init__(config=config)
        self.graph = graph
        self.vector = vector
        self.parser = parser
        self.cache = cache
        self._indexing_lock = asyncio.Lock()
        self._pending_files: Set[Path] = set()
    
    async def index_project(
        self,
        project_path: Path,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """Index an entire project."""
        async with self._indexing_lock:
            start_time = datetime.now()
            stats = {
                'files_processed': 0,
                'symbols_extracted': 0,
                'relationships_found': 0,
                'errors': [],
                'duration_seconds': 0
            }
            
            # Collect files to index
            files_to_index = await self._collect_files(
                project_path,
                force_reindex
            )
            
            self._logger.info(f"Indexing {len(files_to_index)} files")
            
            # Process in batches
            batch_size = self.config.incremental_batch_size
            for i in range(0, len(files_to_index), batch_size):
                batch = files_to_index[i:i + batch_size]
                batch_stats = await self._process_batch(batch)
                
                # Update stats
                stats['files_processed'] += batch_stats['files']
                stats['symbols_extracted'] += batch_stats['symbols']
                stats['relationships_found'] += batch_stats['relationships']
                stats['errors'].extend(batch_stats['errors'])
                
                # Progress update
                progress = (i + len(batch)) / len(files_to_index) * 100
                self._logger.info(f"Progress: {progress:.1f}%")
            
            # Update stats
            stats['duration_seconds'] = (
                datetime.now() - start_time
            ).total_seconds()
            
            return stats
    
    async def _collect_files(
        self,
        project_path: Path,
        force_reindex: bool
    ) -> List[Path]:
        """Collect files that need indexing."""
        files = []
        
        for pattern in ['**/*.py', '**/*.js', '**/*.ts', '**/*.go', '**/*.rs']:
            for file_path in project_path.glob(pattern):
                # Skip ignored patterns
                if self._should_ignore(file_path):
                    continue
                
                # Check if needs indexing
                if force_reindex or await self._needs_indexing(file_path):
                    files.append(file_path)
        
        return files[:self.config.full_index_file_limit]
    
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        path_str = str(file_path)
        
        for pattern in self.config.ignore_patterns:
            if pattern in path_str:
                return True
        
        # Check file size
        try:
            size_mb = file_path.stat().st_size / 1024 / 1024
            if size_mb > self.config.file_size_limit_mb:
                return True
        except:
            pass
        
        return False
    
    async def _needs_indexing(self, file_path: Path) -> bool:
        """Check if file needs indexing."""
        # Get file hash
        current_hash = await self._get_file_hash(file_path)
        
        # Check cache
        cached_hash = await self.cache.get(f"file_hash:{file_path}")
        
        return cached_hash != current_hash
    
    async def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content."""
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            self._logger.error(f"Error hashing {file_path}: {e}")
            return ""
    
    async def _process_batch(
        self,
        files: List[Path]
    ) -> Dict[str, Any]:
        """Process a batch of files."""
        stats = {
            'files': 0,
            'symbols': 0,
            'relationships': 0,
            'errors': []
        }
        
        # Process files concurrently
        tasks = [
            self._index_file(file_path)
            for file_path in files
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for file_path, result in zip(files, results):
            if isinstance(result, Exception):
                stats['errors'].append({
                    'file': str(file_path),
                    'error': str(result)
                })
            else:
                stats['files'] += 1
                stats['symbols'] += result['symbols']
                stats['relationships'] += result['relationships']
        
        return stats
    
    async def _index_file(self, file_path: Path) -> Dict[str, Any]:
        """Index a single file."""
        try:
            # Read file content
            content = file_path.read_bytes()
            
            # Parse and analyze
            symbols, relationships = await self.parser.analyze_file(
                file_path,
                content
            )
            
            # Clear existing data
            await self.graph.delete_file_symbols(str(file_path))
            await self.vector.delete_by_file(str(file_path))
            
            # Store symbols
            symbol_map = {}
            for symbol in symbols:
                symbol_id = await self.graph.add_symbol(**symbol.__dict__)
                symbol_map[symbol.qualified_name] = symbol_id
                
                # Add to vector store
                await self.vector.add_symbol({
                    'id': symbol_id,
                    **symbol.__dict__
                })
            
            # Store relationships
            for rel in relationships:
                # Resolve symbol IDs
                from_id = symbol_map.get(rel.from_symbol)
                to_id = symbol_map.get(rel.to_symbol)
                
                if from_id and to_id:
                    await self.graph.add_relationship(
                        from_symbol_id=from_id,
                        to_symbol_id=to_id,
                        relationship_type=rel.relationship_type,
                        file_path=rel.file_path,
                        line_number=rel.line_number,
                        column_number=rel.column_number,
                        confidence=rel.confidence,
                        context=rel.context
                    )
            
            # Update file info
            file_hash = await self._get_file_hash(file_path)
            await self.graph.update_file(str(file_path), file_hash)
            await self.cache.set(f"file_hash:{file_path}", file_hash)
            
            return {
                'symbols': len(symbols),
                'relationships': len(relationships)
            }
            
        except Exception as e:
            self._logger.error(f"Error indexing {file_path}: {e}")
            raise
    
    async def update_files(
        self,
        file_paths: List[Path]
    ) -> Dict[str, Any]:
        """Update specific files in the index."""
        async with self._indexing_lock:
            return await self._process_batch(file_paths)
    
    async def smart_update(self) -> Dict[str, Any]:
        """Smart incremental update based on pending changes."""
        if not self._pending_files:
            return {'message': 'No pending updates'}
        
        files = list(self._pending_files)
        self._pending_files.clear()
        
        return await self.update_files(files)
    
    def mark_file_changed(self, file_path: Path):
        """Mark a file as changed for next update."""
        self._pending_files.add(file_path)
        
        # Auto-update if threshold reached
        if len(self._pending_files) >= self.config.auto_index_threshold:
            asyncio.create_task(self.smart_update())
```

### File Watcher

```python
# code_intelligence_mcp/indexing/file_watcher.py
from pathlib import Path
from typing import Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import asyncio
from ..core.service import BaseService
from .indexing_service import IndexingService


class CodeFileHandler(FileSystemEventHandler):
    """Handler for file system events."""
    
    def __init__(self, indexing_service: IndexingService):
        self.indexing_service = indexing_service
        self._debounce_tasks: Dict[str, asyncio.Task] = {}
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if not event.is_directory:
            self._handle_file_change(event.src_path)
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if not event.is_directory:
            self._handle_file_change(event.src_path)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if not event.is_directory:
            asyncio.create_task(
                self._handle_file_deletion(event.src_path)
            )
    
    def _handle_file_change(self, path: str):
        """Handle file change with debouncing."""
        # Cancel existing task
        if path in self._debounce_tasks:
            self._debounce_tasks[path].cancel()
        
        # Create new debounced task
        self._debounce_tasks[path] = asyncio.create_task(
            self._debounced_update(path)
        )
    
    async def _debounced_update(self, path: str):
        """Update file after debounce delay."""
        await asyncio.sleep(1.0)  # 1 second debounce
        
        file_path = Path(path)
        if file_path.exists():
            self.indexing_service.mark_file_changed(file_path)
        
        # Clean up task
        self._debounce_tasks.pop(path, None)
    
    async def _handle_file_deletion(self, path: str):
        """Handle file deletion."""
        file_path = Path(path)
        await self.indexing_service.graph.delete_file_symbols(str(file_path))
        await self.indexing_service.vector.delete_by_file(str(file_path))


class FileWatcher(BaseService):
    """Service for watching file changes."""
    
    def __init__(self, indexing_service: IndexingService):
        super().__init__()
        self.indexing_service = indexing_service
        self._observer: Optional[Observer] = None
        self._watched_paths: Set[Path] = set()
    
    async def _initialize(self):
        """Initialize file watcher."""
        self._observer = Observer()
    
    async def watch_directory(self, path: Path):
        """Start watching a directory."""
        if path in self._watched_paths:
            return
        
        handler = CodeFileHandler(self.indexing_service)
        self._observer.schedule(
            handler,
            str(path),
            recursive=True
        )
        
        self._watched_paths.add(path)
        
        if not self._observer.is_alive():
            self._observer.start()
        
        self._logger.info(f"Watching directory: {path}")
    
    async def stop_watching(self, path: Path):
        """Stop watching a directory."""
        # Watchdog doesn't support unscheduling easily
        # Would need to restart observer
        self._watched_paths.discard(path)
    
    async def _shutdown(self):
        """Stop the observer."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
```

## 4. Cache Service Implementation

```python
# code_intelligence_mcp/cache/cache_service.py
from typing import Any, Optional, Dict
from pathlib import Path
import asyncio
import pickle
import hashlib
from datetime import datetime, timedelta
import aiofiles
from ..core.service import BaseService


class CacheService(BaseService):
    """Service for caching various data."""
    
    def __init__(self, cache_dir: Path, config: Any):
        super().__init__(config=config)
        self.cache_dir = cache_dir
        self._memory_cache: Dict[str, Any] = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    async def _initialize(self):
        """Initialize cache directory."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not self._is_expired(entry):
                self._cache_stats['hits'] += 1
                return entry['value']
            else:
                del self._memory_cache[key]
        
        # Check disk cache
        file_path = self._get_cache_path(key)
        if file_path.exists():
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    data = pickle.loads(await f.read())
                
                if not self._is_expired(data):
                    # Add to memory cache
                    self._memory_cache[key] = data
                    self._cache_stats['hits'] += 1
                    return data['value']
                else:
                    # Clean up expired file
                    file_path.unlink()
            except Exception as e:
                self._logger.error(f"Cache read error: {e}")
        
        self._cache_stats['misses'] += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in cache."""
        ttl = ttl or self.config.default_ttl
        
        entry = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=ttl),
            'created_at': datetime.now()
        }
        
        # Add to memory cache
        self._memory_cache[key] = entry
        
        # Persist to disk for important data
        if self._should_persist(key, value):
            file_path = self._get_cache_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(pickle.dumps(entry))
        
        # Check memory limits
        if len(self._memory_cache) > self.config.max_memory_entries:
            await self._evict_lru()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash the key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / key_hash[:2] / f"{key_hash}.cache"
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > entry['expires_at']
    
    def _should_persist(self, key: str, value: Any) -> bool:
        """Determine if value should be persisted to disk."""
        # Persist query results and parsed ASTs
        return key.startswith(('query:', 'ast:', 'graph:'))
    
    async def _evict_lru(self):
        """Evict least recently used entries."""
        # Simple implementation - evict oldest
        if not self._memory_cache:
            return
        
        oldest_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k]['created_at']
        )
        
        del self._memory_cache[oldest_key]
        self._cache_stats['evictions'] += 1
    
    async def _cleanup_task(self):
        """Periodic cleanup of expired entries."""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Clean memory cache
            expired_keys = [
                k for k, v in self._memory_cache.items()
                if self._is_expired(v)
            ]
            
            for key in expired_keys:
                del self._memory_cache[key]
            
            # Clean disk cache
            for cache_file in self.cache_dir.rglob('*.cache'):
                try:
                    async with aiofiles.open(cache_file, 'rb') as f:
                        data = pickle.loads(await f.read())
                    
                    if self._is_expired(data):
                        cache_file.unlink()
                except:
                    # Remove corrupted files
                    cache_file.unlink()
    
    async def clear(self):
        """Clear all cache."""
        self._memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.rglob('*.cache'):
            cache_file.unlink()
        
        self._logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = (
            self._cache_stats['hits'] +
            self._cache_stats['misses']
        )
        
        hit_rate = (
            self._cache_stats['hits'] / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            **self._cache_stats,
            'hit_rate': hit_rate,
            'memory_entries': len(self._memory_cache)
        }
```

## Next Steps

With these components in place, you need to:

1. **Create the DI container module** (`di/`)
2. **Implement language-specific extractors** (`parser/extractors/`)
3. **Create Tree-sitter queries** (`parser/queries/`)
4. **Implement remaining tools** (`tools/`)
5. **Add comprehensive tests** (`tests/`)
6. **Create integration scripts** (`scripts/`)

Each component follows the same patterns:
- Dependency injection through constructor
- Async/await throughout
- Proper error handling and logging
- Testable design with clear interfaces
