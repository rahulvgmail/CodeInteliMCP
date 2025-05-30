"""Extract symbols and relationships from AST."""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import tree_sitter
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    """Represents a code symbol."""
    name: str
    qualified_name: str
    symbol_type: str
    file_path: str
    line_start: int
    line_end: int
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    language: str = ""
    visibility: str = "public"
    is_async: bool = False
    is_generator: bool = False
    complexity: int = 1
    ast_hash: str = ""
    parent_symbol_id: Optional[int] = None
    metadata: Dict[str, Any] = None


@dataclass
class Relationship:
    """Represents a relationship between symbols."""
    from_symbol: str  # qualified name
    to_symbol: str    # qualified name
    relationship_type: str
    file_path: str
    line_number: int
    column_number: Optional[int] = None
    confidence: float = 1.0
    is_direct: bool = True
    context: Optional[str] = None
    metadata: Dict[str, Any] = None


class SymbolExtractor:
    """Extract symbols and relationships from parsed AST."""
    
    def __init__(self, language_registry):
        self.registry = language_registry
    
    def extract_from_file(
        self,
        file_path: str,
        content: bytes,
        language: Optional[str] = None
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Extract symbols and relationships from a file."""
        # Determine language if not provided
        if not language:
            language = self.registry.get_language_from_extension(file_path)
            if not language:
                logger.warning(f"Could not determine language for {file_path}")
                return [], []
        
        # Parse the file
        tree = self.registry.parse(content, language)
        if not tree:
            logger.error(f"Failed to parse {file_path}")
            return [], []
        
        # Extract based on language
        if language == 'python':
            return self._extract_python(tree, file_path, content)
        elif language in ['javascript', 'js']:
            return self._extract_javascript(tree, file_path, content)
        else:
            # Generic extraction for other languages
            return self._extract_generic(tree, file_path, content, language)
    
    def _extract_python(
        self,
        tree: tree_sitter.Tree,
        file_path: str,
        content: bytes
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Extract symbols from Python code."""
        symbols = []
        relationships = []
        
        # Use query if available
        query = self.registry.get_query('python', 'symbols')
        if query:
            return self._extract_with_query(tree, file_path, content, 'python', query)
        
        # Fallback to manual traversal
        def visit_node(node, parent_qualified_name="", class_context=None):
            if node.type == 'module':
                # Process module-level nodes
                for child in node.children:
                    visit_node(child, parent_qualified_name, class_context)
            
            elif node.type == 'class_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = content[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    qualified_name = f"{parent_qualified_name}.{class_name}" if parent_qualified_name else class_name
                    
                    # Extract docstring
                    docstring = self._extract_docstring(node, content)
                    
                    symbol = Symbol(
                        name=class_name,
                        qualified_name=qualified_name,
                        symbol_type='class',
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        docstring=docstring,
                        language='python',
                        ast_hash=self._hash_node(node, content)
                    )
                    symbols.append(symbol)
                    
                    # Look for base classes
                    bases = node.child_by_field_name('superclasses')
                    if bases:
                        for arg in bases.children:
                            if arg.type == 'identifier':
                                base_name = content[arg.start_byte:arg.end_byte].decode('utf-8')
                                rel = Relationship(
                                    from_symbol=qualified_name,
                                    to_symbol=base_name,
                                    relationship_type='inherits',
                                    file_path=file_path,
                                    line_number=arg.start_point[0] + 1,
                                    column_number=arg.start_point[1]
                                )
                                relationships.append(rel)
                    
                    # Process class body
                    body = node.child_by_field_name('body')
                    if body:
                        for child in body.children:
                            visit_node(child, qualified_name, class_name)
            
            elif node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    func_name = content[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    
                    # Determine if it's a method or function
                    is_method = class_context is not None
                    symbol_type = 'method' if is_method else 'function'
                    
                    # Build qualified name
                    if parent_qualified_name:
                        qualified_name = f"{parent_qualified_name}.{func_name}"
                    else:
                        qualified_name = func_name
                    
                    # Extract signature
                    params = node.child_by_field_name('parameters')
                    signature = None
                    if params:
                        signature = content[node.start_byte:params.end_byte].decode('utf-8')
                    
                    # Check if async
                    is_async = any(child.type == 'async' for child in node.children)
                    
                    # Extract docstring
                    docstring = self._extract_docstring(node, content)
                    
                    symbol = Symbol(
                        name=func_name,
                        qualified_name=qualified_name,
                        symbol_type=symbol_type,
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        signature=signature,
                        docstring=docstring,
                        language='python',
                        is_async=is_async,
                        ast_hash=self._hash_node(node, content),
                        visibility='private' if func_name.startswith('_') else 'public'
                    )
                    symbols.append(symbol)
                    
                    # Extract function calls
                    body = node.child_by_field_name('body')
                    if body:
                        calls = self._extract_calls(body, content)
                        for call_name, line, col in calls:
                            rel = Relationship(
                                from_symbol=qualified_name,
                                to_symbol=call_name,
                                relationship_type='calls',
                                file_path=file_path,
                                line_number=line,
                                column_number=col
                            )
                            relationships.append(rel)
            
            # Recursively visit children for other node types
            else:
                for child in node.children:
                    visit_node(child, parent_qualified_name, class_context)
        
        visit_node(tree.root_node)
        return symbols, relationships
    
    def _extract_javascript(
        self,
        tree: tree_sitter.Tree,
        file_path: str,
        content: bytes
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Extract symbols from JavaScript code."""
        symbols = []
        relationships = []
        
        # Use query if available
        query = self.registry.get_query('javascript', 'symbols')
        if query:
            return self._extract_with_query(tree, file_path, content, 'javascript', query)
        
        # Fallback to manual traversal
        # TODO: Implement JavaScript-specific extraction
        return self._extract_generic(tree, file_path, content, 'javascript')
    
    def _extract_generic(
        self,
        tree: tree_sitter.Tree,
        file_path: str,
        content: bytes,
        language: str
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Generic symbol extraction for any language."""
        symbols = []
        relationships = []
        
        # Basic extraction - look for common patterns
        def visit_node(node, depth=0):
            # Try to identify common symbol types
            if 'function' in node.type or 'method' in node.type:
                name = self._get_node_name(node, content)
                if name:
                    symbol = Symbol(
                        name=name,
                        qualified_name=name,
                        symbol_type='function',
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language=language,
                        ast_hash=self._hash_node(node, content)
                    )
                    symbols.append(symbol)
            
            elif 'class' in node.type:
                name = self._get_node_name(node, content)
                if name:
                    symbol = Symbol(
                        name=name,
                        qualified_name=name,
                        symbol_type='class',
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language=language,
                        ast_hash=self._hash_node(node, content)
                    )
                    symbols.append(symbol)
            
            # Recurse
            for child in node.children:
                visit_node(child, depth + 1)
        
        visit_node(tree.root_node)
        return symbols, relationships
    
    def _extract_with_query(
        self,
        tree: tree_sitter.Tree,
        file_path: str,
        content: bytes,
        language: str,
        query: tree_sitter.Query
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Extract symbols using a tree-sitter query."""
        symbols = []
        relationships = []
        
        captures = query.captures(tree.root_node)
        
        for node, capture_name in captures:
            if capture_name == 'function.name':
                # Extract function details
                pass
            elif capture_name == 'class.name':
                # Extract class details
                pass
            # ... handle other captures
        
        return symbols, relationships
    
    def _extract_docstring(self, node: tree_sitter.Node, content: bytes) -> Optional[str]:
        """Extract docstring from a function or class node."""
        body = node.child_by_field_name('body')
        if body and body.child_count > 0:
            first_stmt = body.children[0]
            if first_stmt.type == 'expression_statement':
                expr = first_stmt.children[0] if first_stmt.child_count > 0 else None
                if expr and expr.type == 'string':
                    # Remove quotes and clean up
                    docstring = content[expr.start_byte:expr.end_byte].decode('utf-8')
                    return docstring.strip('"\'')
        return None
    
    def _extract_calls(self, node: tree_sitter.Node, content: bytes) -> List[Tuple[str, int, int]]:
        """Extract function calls from a node."""
        calls = []
        
        def visit(n):
            if n.type == 'call':
                func_node = n.child_by_field_name('function')
                if func_node:
                    if func_node.type == 'identifier':
                        name = content[func_node.start_byte:func_node.end_byte].decode('utf-8')
                        calls.append((name, n.start_point[0] + 1, n.start_point[1]))
                    elif func_node.type == 'attribute':
                        # Handle method calls like obj.method()
                        attr = func_node.child_by_field_name('attribute')
                        if attr:
                            name = content[attr.start_byte:attr.end_byte].decode('utf-8')
                            calls.append((name, n.start_point[0] + 1, n.start_point[1]))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return calls
    
    def _get_node_name(self, node: tree_sitter.Node, content: bytes) -> Optional[str]:
        """Try to extract a name from a node."""
        # Look for common name fields
        for field in ['name', 'identifier']:
            name_node = node.child_by_field_name(field)
            if name_node:
                return content[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Look for identifier children
        for child in node.children:
            if child.type == 'identifier':
                return content[child.start_byte:child.end_byte].decode('utf-8')
        
        return None
    
    def _hash_node(self, node: tree_sitter.Node, content: bytes) -> str:
        """Generate a hash for a node's content."""
        node_content = content[node.start_byte:node.end_byte]
        return hashlib.md5(node_content).hexdigest()