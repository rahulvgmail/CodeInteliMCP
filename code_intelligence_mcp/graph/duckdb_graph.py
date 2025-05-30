"""DuckDB-based graph database for code intelligence."""

import duckdb
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager
import json
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config import get_config

logger = logging.getLogger(__name__)


class DuckDBGraph:
    """Graph database using DuckDB for code relationships."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize DuckDB graph database."""
        config = get_config()
        self.db_path = db_path or config.graph.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema."""
        with self.connection() as conn:
            # Read and execute schema
            schema_path = Path(__file__).parent / "schema.sql"
            if schema_path.exists():
                conn.execute(schema_path.read_text())
            else:
                logger.warning(f"Schema file not found at {schema_path}")
    
    @contextmanager
    def connection(self):
        """Get a database connection."""
        conn = duckdb.connect(str(self.db_path))
        try:
            # Configure connection
            conn.execute("SET memory_limit='256MB'")
            conn.execute("SET threads=4")
            yield conn
        finally:
            conn.close()
    
    async def add_symbol(
        self,
        name: str,
        qualified_name: str,
        symbol_type: str,
        file_path: str,
        line_start: int,
        line_end: int,
        language: str,
        ast_hash: str,
        repository_id: Optional[int] = None,
        **kwargs
    ) -> int:
        """Add a symbol to the graph."""
        def _add():
            with self.connection() as conn:
                result = conn.execute("""
                    INSERT INTO symbols (
                        repository_id, name, qualified_name, symbol_type, file_path,
                        line_start, line_end, language, ast_hash,
                        column_start, column_end, signature, docstring,
                        visibility, is_async, is_generator, complexity,
                        parent_symbol_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id
                """, [
                    repository_id or 1,  # Default to repository 1 for backward compatibility
                    name, qualified_name, symbol_type, file_path,
                    line_start, line_end, language, ast_hash,
                    kwargs.get('column_start'), kwargs.get('column_end'),
                    kwargs.get('signature'), kwargs.get('docstring'),
                    kwargs.get('visibility', 'public'),
                    kwargs.get('is_async', False),
                    kwargs.get('is_generator', False),
                    kwargs.get('complexity', 1),
                    kwargs.get('parent_symbol_id'),
                    json.dumps(kwargs.get('metadata', {}))
                ]).fetchone()
                return result[0]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _add
        )
    
    async def add_relationship(
        self,
        from_symbol_id: int,
        to_symbol_id: int,
        relationship_type: str,
        file_path: str,
        line_number: int,
        **kwargs
    ) -> int:
        """Add a relationship between symbols."""
        def _add():
            with self.connection() as conn:
                result = conn.execute("""
                    INSERT INTO relationships (
                        from_symbol_id, to_symbol_id, relationship_type,
                        file_path, line_number, column_number,
                        confidence, is_direct, context, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id
                """, [
                    from_symbol_id, to_symbol_id, relationship_type,
                    file_path, line_number,
                    kwargs.get('column_number'),
                    kwargs.get('confidence', 1.0),
                    kwargs.get('is_direct', True),
                    kwargs.get('context'),
                    json.dumps(kwargs.get('metadata', {}))
                ]).fetchone()
                return result[0]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _add
        )
    
    async def find_symbol(
        self, 
        name: Optional[str] = None,
        qualified_name: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find symbols by name or qualified name."""
        def _find():
            with self.connection() as conn:
                query = "SELECT * FROM symbols WHERE 1=1"
                params = []
                
                if name:
                    query += " AND name = ?"
                    params.append(name)
                if qualified_name:
                    query += " AND qualified_name = ?"
                    params.append(qualified_name)
                if file_path:
                    query += " AND file_path = ?"
                    params.append(file_path)
                
                results = conn.execute(query, params).fetchall()
                columns = [desc[0] for desc in conn.description]
                return [dict(zip(columns, row)) for row in results]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _find
        )
    
    async def find_usages(self, symbol_id: int) -> List[Dict[str, Any]]:
        """Find all places where a symbol is used."""
        def _find():
            with self.connection() as conn:
                # Use the pre-defined macro
                results = conn.execute(
                    "SELECT * FROM get_all_callers(?)",
                    [symbol_id]
                ).fetchall()
                
                columns = ['id', 'name', 'qualified_name', 
                          'file_path', 'line_start', 'depth']
                return [dict(zip(columns, row)) for row in results]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _find
        )
    
    async def find_dependencies(
        self, 
        symbol_id: int,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find all dependencies of a symbol."""
        def _find():
            with self.connection() as conn:
                if relationship_types:
                    types_str = ','.join(f"'{t}'" for t in relationship_types)
                    type_filter = f"AND r.relationship_type IN ({types_str})"
                else:
                    type_filter = ""
                
                query = f"""
                    WITH RECURSIVE deps AS (
                        SELECT 
                            s.id, s.name, s.qualified_name, s.file_path,
                            r.relationship_type, 1 as depth
                        FROM symbols s
                        JOIN relationships r ON s.id = r.to_symbol_id
                        WHERE r.from_symbol_id = ? {type_filter}
                        
                        UNION ALL
                        
                        SELECT 
                            s.id, s.name, s.qualified_name, s.file_path,
                            r.relationship_type, d.depth + 1
                        FROM symbols s
                        JOIN relationships r ON s.id = r.to_symbol_id
                        JOIN deps d ON r.from_symbol_id = d.id
                        WHERE d.depth < 5 {type_filter}
                    )
                    SELECT DISTINCT * FROM deps ORDER BY depth, name
                """
                
                results = conn.execute(query, [symbol_id]).fetchall()
                columns = ['id', 'name', 'qualified_name', 
                          'file_path', 'relationship_type', 'depth']
                return [dict(zip(columns, row)) for row in results]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _find
        )
    
    async def find_circular_dependencies(self) -> List[Dict[str, Any]]:
        """Find circular dependencies in the codebase."""
        def _find():
            with self.connection() as conn:
                results = conn.execute("SELECT * FROM find_circular_deps()").fetchall()
                return [{"path": row[0], "cycle": row[1]} for row in results]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _find
        )
    
    async def get_file_symbols(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all symbols in a file."""
        def _get():
            with self.connection() as conn:
                results = conn.execute("""
                    SELECT * FROM symbols 
                    WHERE file_path = ? 
                    ORDER BY line_start
                """, [file_path]).fetchall()
                
                columns = [desc[0] for desc in conn.description]
                return [dict(zip(columns, row)) for row in results]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _get
        )
    
    async def get_call_graph(
        self, 
        root_symbol_id: int,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Get the call graph starting from a symbol."""
        def _get():
            with self.connection() as conn:
                # Get all relationships in the call graph
                results = conn.execute("""
                    WITH RECURSIVE call_tree AS (
                        SELECT 
                            r.from_symbol_id, r.to_symbol_id,
                            s1.name as from_name, s2.name as to_name,
                            1 as depth
                        FROM relationships r
                        JOIN symbols s1 ON r.from_symbol_id = s1.id
                        JOIN symbols s2 ON r.to_symbol_id = s2.id
                        WHERE r.from_symbol_id = ? 
                        AND r.relationship_type = 'calls'
                        
                        UNION ALL
                        
                        SELECT 
                            r.from_symbol_id, r.to_symbol_id,
                            s1.name as from_name, s2.name as to_name,
                            ct.depth + 1
                        FROM relationships r
                        JOIN symbols s1 ON r.from_symbol_id = s1.id
                        JOIN symbols s2 ON r.to_symbol_id = s2.id
                        JOIN call_tree ct ON r.from_symbol_id = ct.to_symbol_id
                        WHERE r.relationship_type = 'calls'
                        AND ct.depth < ?
                    )
                    SELECT * FROM call_tree
                """, [root_symbol_id, max_depth]).fetchall()
                
                # Build graph structure
                nodes = set()
                edges = []
                
                for row in results:
                    from_id, to_id, from_name, to_name, depth = row
                    nodes.add((from_id, from_name))
                    nodes.add((to_id, to_name))
                    edges.append({
                        'from': from_id,
                        'to': to_id,
                        'from_name': from_name,
                        'to_name': to_name,
                        'depth': depth
                    })
                
                return {
                    'nodes': [{'id': n[0], 'name': n[1]} for n in nodes],
                    'edges': edges
                }
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _get
        )
    
    async def update_file(self, file_path: str, new_hash: str) -> None:
        """Update file information."""
        def _update():
            with self.connection() as conn:
                conn.execute("""
                    INSERT INTO files (path, language, content_hash, indexed_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (path) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        updated_at = CURRENT_TIMESTAMP
                """, [file_path, 'unknown', new_hash, datetime.now()])
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _update
        )
    
    async def delete_file_symbols(self, file_path: str) -> None:
        """Delete all symbols from a file."""
        def _delete():
            with self.connection() as conn:
                # Delete symbols (relationships will cascade)
                conn.execute(
                    "DELETE FROM symbols WHERE file_path = ?",
                    [file_path]
                )
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _delete
        )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        def _get_stats():
            with self.connection() as conn:
                stats = {}
                
                # Symbol counts by type
                results = conn.execute("""
                    SELECT symbol_type, COUNT(*) as count
                    FROM symbols
                    GROUP BY symbol_type
                """).fetchall()
                stats['symbols_by_type'] = dict(results)
                
                # Total counts
                stats['total_symbols'] = conn.execute(
                    "SELECT COUNT(*) FROM symbols"
                ).fetchone()[0]
                
                stats['total_relationships'] = conn.execute(
                    "SELECT COUNT(*) FROM relationships"
                ).fetchone()[0]
                
                stats['total_files'] = conn.execute(
                    "SELECT COUNT(*) FROM files"
                ).fetchone()[0]
                
                # Relationship counts by type
                results = conn.execute("""
                    SELECT relationship_type, COUNT(*) as count
                    FROM relationships
                    GROUP BY relationship_type
                """).fetchall()
                stats['relationships_by_type'] = dict(results)
                
                # Database size
                stats['db_size_mb'] = Path(self.db_path).stat().st_size / 1024 / 1024
                
                return stats
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _get_stats
        )
    
    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """Execute a raw SQL query."""
        def _execute():
            with self.connection() as conn:
                if params:
                    result = conn.execute(query, params).fetchall()
                else:
                    result = conn.execute(query).fetchall()
                return result
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _execute
        )
    
    async def add_repository(self, name: str, path: str, description: Optional[str] = None) -> int:
        """Add a new repository."""
        def _add_repo():
            with self.connection() as conn:
                result = conn.execute("""
                    INSERT INTO repositories (name, path, description, created_at, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id
                """, [name, path, description])
                return result.fetchone()[0]
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _add_repo
        )
    
    async def list_repositories(self) -> List[Dict[str, Any]]:
        """List all repositories."""
        def _list_repos():
            with self.connection() as conn:
                result = conn.execute("""
                    SELECT id, name, path, description, language, indexed_at, created_at, updated_at
                    FROM repositories
                    ORDER BY name
                """)
                
                repositories = []
                for row in result.fetchall():
                    repositories.append({
                        'id': row[0],
                        'name': row[1], 
                        'path': row[2],
                        'description': row[3],
                        'language': row[4],
                        'indexed_at': row[5],
                        'created_at': row[6],
                        'updated_at': row[7]
                    })
                return repositories
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _list_repos
        )

    def close(self):
        """Close the graph database."""
        self._executor.shutdown(wait=True)
