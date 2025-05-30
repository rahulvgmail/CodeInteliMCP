# 🚀 Performance Optimization Guide

## Overview

This guide covers performance optimization strategies for the Code Intelligence MCP server to handle large codebases efficiently.

## 1. Database Optimizations

### DuckDB Performance Tuning

```python
# code_intelligence_mcp/graph/performance.py
from typing import Dict, Any, List
import duckdb
from pathlib import Path


class OptimizedDuckDBGraph:
    """Performance-optimized DuckDB graph implementation."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_optimized_db()
    
    def _init_optimized_db(self):
        """Initialize with performance optimizations."""
        with self.connection() as conn:
            # Performance settings
            conn.execute("SET memory_limit='1GB'")
            conn.execute("SET threads=8")
            conn.execute("SET enable_profiling=false")
            conn.execute("SET enable_progress_bar=false")
            
            # Create optimized indexes
            self._create_optimized_indexes(conn)
            
            # Create materialized views for common queries
            self._create_materialized_views(conn)
    
    def _create_optimized_indexes(self, conn):
        """Create performance-critical indexes."""
        # Composite indexes for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbols_name_type 
            ON symbols(name, symbol_type)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbols_file_line 
            ON symbols(file_path, line_start)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_composite 
            ON relationships(from_symbol_id, to_symbol_id, relationship_type)
        """)
        
        # Covering indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbols_covering 
            ON symbols(name, id, qualified_name, file_path, line_start)
        """)
    
    def _create_materialized_views(self, conn):
        """Create materialized views for expensive queries."""
        # Direct dependencies view
        conn.execute("""
            CREATE OR REPLACE VIEW direct_dependencies AS
            SELECT 
                s1.id as symbol_id,
                s1.name as symbol_name,
                s2.id as dependency_id,
                s2.name as dependency_name,
                r.relationship_type
            FROM relationships r
            JOIN symbols s1 ON r.from_symbol_id = s1.id
            JOIN symbols s2 ON r.to_symbol_id = s2.id
            WHERE r.is_direct = true
        """)
        
        # File dependency summary
        conn.execute("""
            CREATE OR REPLACE VIEW file_dependencies AS
            SELECT 
                s1.file_path as from_file,
                s2.file_path as to_file,
                COUNT(*) as dependency_count
            FROM relationships r
            JOIN symbols s1 ON r.from_symbol_id = s1.id
            JOIN symbols s2 ON r.to_symbol_id = s2.id
            WHERE s1.file_path != s2.file_path
            GROUP BY s1.file_path, s2.file_path
        """)
```

### Query Optimization Patterns

```python
# code_intelligence_mcp/graph/query_optimizer.py
class QueryOptimizer:
    """Optimize graph queries for performance."""
    
    def __init__(self, graph: DuckDBGraph):
        self.graph = graph
        self._query_cache = {}
        self._prepared_statements = {}
    
    async def find_usages_optimized(
        self,
        symbol_id: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Optimized usage finding with prepared statements."""
        # Use prepared statement
        if 'find_usages' not in self._prepared_statements:
            with self.graph.connection() as conn:
                self._prepared_statements['find_usages'] = conn.prepare("""
                    WITH RECURSIVE usage_tree AS (
                        -- Anchor: direct usages
                        SELECT 
                            s.*, 
                            1 as depth,
                            ARRAY[s.id] as path
                        FROM symbols s
                        JOIN relationships r ON s.id = r.from_symbol_id
                        WHERE r.to_symbol_id = $1
                        AND r.relationship_type = 'calls'
                        
                        UNION ALL
                        
                        -- Recursive: indirect usages
                        SELECT 
                            s.*,
                            ut.depth + 1,
                            array_append(ut.path, s.id)
                        FROM symbols s
                        JOIN relationships r ON s.id = r.from_symbol_id
                        JOIN usage_tree ut ON r.to_symbol_id = ut.id
                        WHERE ut.depth < 3
                        AND s.id != ALL(ut.path)  -- Prevent cycles
                    )
                    SELECT * FROM usage_tree
                    ORDER BY depth, name
                    LIMIT $2
                """)
        
        # Execute with parameters
        result = self._prepared_statements['find_usages'].execute(
            [symbol_id, limit]
        ).fetchall()
        
        return self._format_results(result)
    
    async def batch_find_symbols(
        self,
        names: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Batch symbol lookup for efficiency."""
        with self.graph.connection() as conn:
            # Use VALUES clause for batch lookup
            placeholders = ','.join(['(?)' for _ in names])
            query = f"""
                WITH search_names(name) AS (
                    VALUES {placeholders}
                )
                SELECT s.*, sn.name as search_name
                FROM search_names sn
                LEFT JOIN symbols s ON s.name = sn.name
                ORDER BY sn.name, s.qualified_name
            """
            
            results = conn.execute(query, names).fetchall()
            
            # Group by search name
            grouped = {}
            for row in results:
                search_name = row[-1]  # Last column is search_name
                if search_name not in grouped:
                    grouped[search_name] = []
                if row[0] is not None:  # Symbol found
                    grouped[search_name].append(dict(row[:-1]))
            
            return grouped
```

## 2. Vector Store Optimizations

### Batch Embedding Generation

```python
# code_intelligence_mcp/vector/embedding_optimizer.py
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor


class EmbeddingOptimizer:
    """Optimized embedding generation and storage."""
    
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Enable optimizations
        self.model.max_seq_length = 256  # Limit sequence length
        self.model.eval()  # Set to eval mode
    
    async def batch_generate_embeddings(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """Generate embeddings in batches."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Generate embeddings in thread pool
            batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._generate_batch,
                batch
            )
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def _generate_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch."""
        # Truncate long texts
        truncated = [t[:512] for t in texts]
        
        # Generate embeddings with optimizations
        with torch.no_grad():  # Disable gradient computation
            embeddings = self.model.encode(
                truncated,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize
                show_progress_bar=False
            )
        
        return embeddings
```

### ChromaDB Optimization

```python
# code_intelligence_mcp/vector/chroma_optimizer.py
class OptimizedChromaStore:
    """Optimized ChromaDB operations."""
    
    def __init__(self, persist_directory: Path):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                # Performance settings
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(persist_directory),
                chroma_server_cors_allow_origins=["*"]
            )
        )
        
        self._init_collections()
    
    async def bulk_add(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """Bulk add with optimizations."""
        collection = self._collections[collection_name]
        
        # Add in chunks to avoid memory issues
        chunk_size = 1000
        
        for i in range(0, len(documents), chunk_size):
            end_idx = min(i + chunk_size, len(documents))
            
            collection.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
    
    async def parallel_search(
        self,
        queries: List[str],
        collection_name: str,
        n_results: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Parallel similarity search."""
        collection = self._collections[collection_name]
        
        # Generate query embeddings in batch
        query_embeddings = await self.embedding_optimizer.batch_generate_embeddings(
            queries
        )
        
        # Search in parallel
        results = collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(queries)):
            query_results = []
            for j in range(len(results['ids'][i])):
                query_results.append({
                    'id': results['ids'][i][j],
                    'document': results['documents'][i][j],
                    'metadata': results['metadatas'][i][j],
                    'distance': results['distances'][i][j]
                })
            formatted_results.append(query_results)
        
        return formatted_results
```

## 3. Parsing Optimizations

### Parallel File Processing

```python
# code_intelligence_mcp/parser/parallel_parser.py
import asyncio
from typing import List, Tuple, Dict, Any
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


class ParallelParser:
    """Parallel parsing for large codebases."""
    
    def __init__(self, language_registry, max_workers: int = None):
        self.language_registry = language_registry
        self.max_workers = max_workers or mp.cpu_count()
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
    
    async def parse_files_parallel(
        self,
        file_paths: List[Path]
    ) -> Dict[Path, Tuple[List[Symbol], List[Relationship]]]:
        """Parse multiple files in parallel."""
        # Group files by language for better cache usage
        files_by_language = self._group_by_language(file_paths)
        
        # Process each language group
        all_results = {}
        
        for language, files in files_by_language.items():
            # Process files in chunks
            chunk_size = max(1, len(files) // self.max_workers)
            chunks = [
                files[i:i + chunk_size]
                for i in range(0, len(files), chunk_size)
            ]
            
            # Parse chunks in parallel
            tasks = [
                self._parse_chunk(chunk, language)
                for chunk in chunks
            ]
            
            chunk_results = await asyncio.gather(*tasks)
            
            # Merge results
            for chunk_result in chunk_results:
                all_results.update(chunk_result)
        
        return all_results
    
    async def _parse_chunk(
        self,
        files: List[Path],
        language: str
    ) -> Dict[Path, Tuple[List[Symbol], List[Relationship]]]:
        """Parse a chunk of files."""
        loop = asyncio.get_event_loop()
        
        # Run parsing in process pool
        result = await loop.run_in_executor(
            self._executor,
            self._parse_files_sync,
            files,
            language
        )
        
        return result
    
    def _parse_files_sync(
        self,
        files: List[Path],
        language: str
    ) -> Dict[Path, Tuple[List[Symbol], List[Relationship]]]:
        """Synchronous parsing for process pool."""
        # Import here to avoid pickling issues
        import tree_sitter
        from ..parser.extractors import get_extractor
        
        parser = self._get_parser_for_language(language)
        extractor = get_extractor(language)
        
        results = {}
        
        for file_path in files:
            try:
                content = file_path.read_bytes()
                tree = parser.parse(content)
                
                symbols, relationships = extractor.extract_sync(
                    tree,
                    file_path,
                    content
                )
                
                results[file_path] = (symbols, relationships)
            except Exception as e:
                # Log error but continue
                results[file_path] = ([], [])
        
        return results
```

### AST Caching

```python
# code_intelligence_mcp/parser/ast_cache.py
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Tuple
import tree_sitter


class ASTCache:
    """Cache parsed ASTs to avoid re-parsing."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for hot files
        self._memory_cache = {}
        self._memory_cache_size = 0
        self._max_memory_cache_mb = 100
    
    def get_ast(
        self,
        file_path: Path,
        content: bytes,
        parser: tree_sitter.Parser
    ) -> Tuple[tree_sitter.Tree, bool]:
        """Get AST from cache or parse."""
        # Generate cache key
        content_hash = hashlib.sha256(content).hexdigest()
        cache_key = f"{file_path.name}_{content_hash}"
        
        # Check memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key], True
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.ast"
        if cache_file.exists():
            try:
                # Load from disk
                with open(cache_file, 'rb') as f:
                    tree_bytes = f.read()
                
                # Reconstruct tree (pseudo-code, actual implementation varies)
                tree = self._deserialize_tree(tree_bytes, parser)
                
                # Add to memory cache
                self._add_to_memory_cache(cache_key, tree)
                
                return tree, True
            except:
                # Cache corrupted, re-parse
                pass
        
        # Parse file
        tree = parser.parse(content)
        
        # Cache result
        self._cache_ast(cache_key, tree, cache_file)
        
        return tree, False
    
    def _add_to_memory_cache(self, key: str, tree: tree_sitter.Tree):
        """Add to memory cache with size limit."""
        # Estimate tree size (rough approximation)
        tree_size = len(str(tree.root_node)) * 2  # bytes
        
        # Evict if needed
        while (self._memory_cache_size + tree_size > 
               self._max_memory_cache_mb * 1024 * 1024):
            if not self._memory_cache:
                break
            
            # Evict oldest (simple FIFO)
            oldest_key = next(iter(self._memory_cache))
            evicted = self._memory_cache.pop(oldest_key)
            self._memory_cache_size -= len(str(evicted.root_node)) * 2
        
        # Add to cache
        self._memory_cache[key] = tree
        self._memory_cache_size += tree_size
```

## 4. Indexing Optimizations

### Incremental Indexing Strategy

```python
# code_intelligence_mcp/indexing/smart_indexer.py
class SmartIndexer:
    """Smart incremental indexing with optimizations."""
    
    def __init__(self, graph, vector, parser):
        self.graph = graph
        self.vector = vector
        self.parser = parser
        
        # Track indexing state
        self._file_states = {}  # file -> (hash, timestamp)
        self._dependency_graph = {}  # file -> dependent files
    
    async def smart_update(
        self,
        changed_files: List[Path]
    ) -> Dict[str, Any]:
        """Smart incremental update."""
        # Build dependency information
        await self._build_dependency_graph()
        
        # Find all affected files
        affected_files = self._find_affected_files(changed_files)
        
        # Order files by dependency
        ordered_files = self._topological_sort(affected_files)
        
        # Update in optimal order
        stats = {
            'files_processed': 0,
            'symbols_updated': 0,
            'relationships_updated': 0
        }
        
        # Process in batches for efficiency
        batch_size = 10
        for i in range(0, len(ordered_files), batch_size):
            batch = ordered_files[i:i + batch_size]
            
            # Process batch in parallel
            batch_results = await asyncio.gather(*[
                self._update_file(file_path)
                for file_path in batch
            ])
            
            # Aggregate stats
            for result in batch_results:
                stats['files_processed'] += 1
                stats['symbols_updated'] += result['symbols']
                stats['relationships_updated'] += result['relationships']
        
        return stats
    
    async def _build_dependency_graph(self):
        """Build file dependency graph."""
        # Query import relationships
        dependencies = await self.graph.execute_query("""
            SELECT DISTINCT
                s1.file_path as from_file,
                s2.file_path as to_file
            FROM relationships r
            JOIN symbols s1 ON r.from_symbol_id = s1.id
            JOIN symbols s2 ON r.to_symbol_id = s2.id
            WHERE r.relationship_type = 'imports'
            AND s1.file_path != s2.file_path
        """)
        
        # Build graph
        self._dependency_graph.clear()
        for from_file, to_file in dependencies:
            if to_file not in self._dependency_graph:
                self._dependency_graph[to_file] = set()
            self._dependency_graph[to_file].add(from_file)
    
    def _find_affected_files(
        self,
        changed_files: List[Path]
    ) -> Set[Path]:
        """Find all files affected by changes."""
        affected = set(changed_files)
        to_process = list(changed_files)
        
        while to_process:
            current = to_process.pop()
            current_str = str(current)
            
            # Find files that depend on current file
            dependents = self._dependency_graph.get(current_str, set())
            
            for dependent in dependents:
                dep_path = Path(dependent)
                if dep_path not in affected:
                    affected.add(dep_path)
                    to_process.append(dep_path)
        
        return affected
```

### Bulk Operations

```python
# code_intelligence_mcp/indexing/bulk_operations.py
class BulkIndexer:
    """Optimized bulk indexing operations."""
    
    def __init__(self, graph, vector):
        self.graph = graph
        self.vector = vector
    
    async def bulk_insert_symbols(
        self,
        symbols: List[Symbol]
    ) -> Dict[str, int]:
        """Bulk insert symbols efficiently."""
        # Prepare bulk insert data
        symbol_data = []
        for symbol in symbols:
            symbol_data.append((
                symbol.name,
                symbol.qualified_name,
                symbol.symbol_type,
                symbol.file_path,
                symbol.line_start,
                symbol.line_end,
                symbol.language,
                symbol.ast_hash,
                # ... other fields
            ))
        
        # Bulk insert using DuckDB
        with self.graph.connection() as conn:
            # Use COPY for maximum performance
            conn.execute("""
                CREATE TEMPORARY TABLE temp_symbols AS 
                SELECT * FROM symbols WHERE 1=0
            """)
            
            # Insert data
            conn.executemany("""
                INSERT INTO temp_symbols (
                    name, qualified_name, symbol_type, file_path,
                    line_start, line_end, language, ast_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, symbol_data)
            
            # Merge into main table
            result = conn.execute("""
                INSERT INTO symbols 
                SELECT * FROM temp_symbols
                RETURNING id, qualified_name
            """).fetchall()
            
            # Create ID mapping
            id_map = {qname: id for id, qname in result}
            
            # Drop temp table
            conn.execute("DROP TABLE temp_symbols")
        
        return id_map
    
    async def bulk_insert_relationships(
        self,
        relationships: List[Relationship],
        symbol_id_map: Dict[str, int]
    ):
        """Bulk insert relationships."""
        # Prepare relationship data
        rel_data = []
        for rel in relationships:
            from_id = symbol_id_map.get(rel.from_symbol)
            to_id = symbol_id_map.get(rel.to_symbol)
            
            if from_id and to_id:
                rel_data.append((
                    from_id,
                    to_id,
                    rel.relationship_type,
                    rel.file_path,
                    rel.line_number,
                    rel.confidence
                ))
        
        # Bulk insert
        if rel_data:
            with self.graph.connection() as conn:
                conn.executemany("""
                    INSERT INTO relationships (
                        from_symbol_id, to_symbol_id, relationship_type,
                        file_path, line_number, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, rel_data)
```

## 5. Memory Management

### Memory-Efficient Processing

```python
# code_intelligence_mcp/utils/memory_manager.py
import psutil
import gc
from typing import Iterator, List, TypeVar, Callable
import asyncio


T = TypeVar('T')


class MemoryManager:
    """Manage memory usage during processing."""
    
    def __init__(self, threshold_percent: float = 80.0):
        self.threshold_percent = threshold_percent
        self._last_gc_time = 0
    
    def check_memory(self) -> Tuple[float, bool]:
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        should_gc = usage_percent > self.threshold_percent
        
        return usage_percent, should_gc
    
    async def process_with_memory_limit(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        batch_size: int = 100
    ) -> List[Any]:
        """Process items with memory management."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch
            batch_results = await asyncio.gather(*[
                processor(item) for item in batch
            ])
            results.extend(batch_results)
            
            # Check memory
            usage, should_gc = self.check_memory()
            if should_gc:
                # Force garbage collection
                gc.collect()
                
                # If still high, reduce batch size
                if usage > 90:
                    batch_size = max(10, batch_size // 2)
                
                # Wait a bit for memory to settle
                await asyncio.sleep(0.1)
        
        return results
    
    def iterate_large_results(
        self,
        query_func: Callable,
        chunk_size: int = 1000
    ) -> Iterator[List[Any]]:
        """Iterate through large query results efficiently."""
        offset = 0
        
        while True:
            # Fetch chunk
            chunk = query_func(limit=chunk_size, offset=offset)
            
            if not chunk:
                break
            
            yield chunk
            
            # Clear references
            del chunk
            
            # Check memory
            _, should_gc = self.check_memory()
            if should_gc:
                gc.collect()
            
            offset += chunk_size
```

## 6. Caching Strategies

### Multi-Level Cache

```python
# code_intelligence_mcp/cache/multi_level_cache.py
from typing import Any, Optional, Dict
import asyncio
from datetime import datetime, timedelta
import redis
import pickle


class MultiLevelCache:
    """Multi-level caching system."""
    
    def __init__(
        self,
        memory_size_mb: int = 100,
        redis_url: Optional[str] = None,
        disk_cache_dir: Optional[Path] = None
    ):
        # L1: In-memory cache (fastest)
        self._memory_cache = LRUCache(max_size_mb=memory_size_mb)
        
        # L2: Redis cache (fast, shared)
        self._redis = redis.from_url(redis_url) if redis_url else None
        
        # L3: Disk cache (slowest, largest)
        self._disk_cache = DiskCache(disk_cache_dir) if disk_cache_dir else None
        
        # Cache stats
        self._stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0
        }
    
    async def get(
        self,
        key: str,
        loader: Optional[Callable] = None
    ) -> Optional[Any]:
        """Get value from cache hierarchy."""
        # Try L1 (memory)
        value = self._memory_cache.get(key)
        if value is not None:
            self._stats['l1_hits'] += 1
            return value
        
        # Try L2 (Redis)
        if self._redis:
            value = await self._get_from_redis(key)
            if value is not None:
                self._stats['l2_hits'] += 1
                # Promote to L1
                self._memory_cache.set(key, value)
                return value
        
        # Try L3 (disk)
        if self._disk_cache:
            value = await self._disk_cache.get(key)
            if value is not None:
                self._stats['l3_hits'] += 1
                # Promote to L1 and L2
                self._memory_cache.set(key, value)
                if self._redis:
                    await self._set_in_redis(key, value)
                return value
        
        # Miss - load if loader provided
        self._stats['misses'] += 1
        
        if loader:
            value = await loader()
            if value is not None:
                await self.set(key, value)
            return value
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in all cache levels."""
        # Set in all levels
        self._memory_cache.set(key, value, ttl)
        
        if self._redis:
            await self._set_in_redis(key, value, ttl)
        
        if self._disk_cache:
            await self._disk_cache.set(key, value, ttl)
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get from Redis with async."""
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                self._redis.get,
                key
            )
            return pickle.loads(data) if data else None
        except:
            return None
    
    async def _set_in_redis(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set in Redis with async."""
        try:
            data = pickle.dumps(value)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._redis.set(key, data, ex=ttl)
            )
        except:
            pass  # Redis errors shouldn't break the flow
```

## 7. Query Performance Patterns

### Query Result Streaming

```python
# code_intelligence_mcp/graph/query_streaming.py
class QueryStreamer:
    """Stream large query results efficiently."""
    
    def __init__(self, graph: DuckDBGraph):
        self.graph = graph
    
    async def stream_symbols(
        self,
        filters: Dict[str, Any],
        chunk_size: int = 1000
    ) -> AsyncIterator[List[Dict[str, Any]]]:
        """Stream symbol results."""
        # Build query
        query = "SELECT * FROM symbols WHERE 1=1"
        params = []
        
        if 'language' in filters:
            query += " AND language = ?"
            params.append(filters['language'])
        
        if 'symbol_type' in filters:
            query += " AND symbol_type = ?"
            params.append(filters['symbol_type'])
        
        query += " ORDER BY id"
        
        # Stream results
        offset = 0
        while True:
            chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
            
            with self.graph.connection() as conn:
                results = conn.execute(chunk_query, params).fetchall()
                
                if not results:
                    break
                
                # Format results
                columns = [desc[0] for desc in conn.description]
                formatted = [
                    dict(zip(columns, row))
                    for row in results
                ]
                
                yield formatted
                
                offset += chunk_size
    
    async def stream_relationships(
        self,
        symbol_id: Optional[int] = None,
        relationship_type: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream relationships with joins."""
        base_query = """
            SELECT 
                r.*,
                s1.name as from_name,
                s1.qualified_name as from_qualified,
                s2.name as to_name,
                s2.qualified_name as to_qualified
            FROM relationships r
            JOIN symbols s1 ON r.from_symbol_id = s1.id
            JOIN symbols s2 ON r.to_symbol_id = s2.id
            WHERE 1=1
        """
        
        params = []
        
        if symbol_id:
            base_query += " AND (r.from_symbol_id = ? OR r.to_symbol_id = ?)"
            params.extend([symbol_id, symbol_id])
        
        if relationship_type:
            base_query += " AND r.relationship_type = ?"
            params.append(relationship_type)
        
        # Stream with pagination
        async for chunk in self._paginate_query(base_query, params):
            yield chunk
```

## 8. Performance Monitoring

### Performance Metrics Collection

```python
# code_intelligence_mcp/monitoring/performance_monitor.py
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
import prometheus_client as prom


class PerformanceMonitor:
    """Monitor and collect performance metrics."""
    
    def __init__(self):
        # Prometheus metrics
        self.query_duration = prom.Histogram(
            'code_intel_query_duration_seconds',
            'Query execution duration',
            ['query_type']
        )
        
        self.indexing_rate = prom.Counter(
            'code_intel_files_indexed_total',
            'Total files indexed'
        )
        
        self.cache_hits = prom.Counter(
            'code_intel_cache_hits_total',
            'Cache hit count',
            ['cache_level']
        )
        
        self.memory_usage = prom.Gauge(
            'code_intel_memory_usage_bytes',
            'Current memory usage'
        )
        
        # Internal metrics
        self._operation_times = {}
    
    @asynccontextmanager
    async def measure_operation(self, operation_name: str):
        """Measure operation duration."""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            # Record in Prometheus
            self.query_duration.labels(
                query_type=operation_name
            ).observe(duration)
            
            # Store internally
            if operation_name not in self._operation_times:
                self._operation_times[operation_name] = []
            
            self._operation_times[operation_name].append(duration)
            
            # Keep only last 100 measurements
            if len(self._operation_times[operation_name]) > 100:
                self._operation_times[operation_name] = \
                    self._operation_times[operation_name][-100:]
    
    def get_operation_stats(
        self,
        operation_name: str
    ) -> Dict[str, float]:
        """Get statistics for an operation."""
        times = self._operation_times.get(operation_name, [])
        
        if not times:
            return {}
        
        return {
            'count': len(times),
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'p50': sorted(times)[len(times) // 2],
            'p95': sorted(times)[int(len(times) * 0.95)],
            'p99': sorted(times)[int(len(times) * 0.99)]
        }
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        asyncio.create_task(self._monitor_memory())
        asyncio.create_task(self._export_metrics())
    
    async def _monitor_memory(self):
        """Monitor memory usage periodically."""
        while True:
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            
            await asyncio.sleep(10)  # Every 10 seconds
```

## Performance Best Practices

### 1. Database Queries
- Use prepared statements for repeated queries
- Create appropriate indexes for common access patterns
- Use materialized views for complex aggregations
- Batch operations when possible

### 2. Memory Management
- Stream large results instead of loading all at once
- Use generators for processing sequences
- Implement proper cleanup in finally blocks
- Monitor memory usage and trigger GC when needed

### 3. Caching
- Implement multi-level caching (memory → Redis → disk)
- Use appropriate TTLs for different data types
- Cache computed results, not raw data
- Implement cache warming for critical data

### 4. Concurrency
- Use asyncio for I/O-bound operations
- Use process pools for CPU-bound operations
- Implement proper connection pooling
- Avoid blocking the event loop

### 5. Monitoring
- Track key performance metrics
- Set up alerts for performance degradation
- Profile regularly to identify bottlenecks
- Use distributed tracing for complex operations

This comprehensive performance optimization guide ensures the Code Intelligence MCP server can handle large codebases efficiently.
