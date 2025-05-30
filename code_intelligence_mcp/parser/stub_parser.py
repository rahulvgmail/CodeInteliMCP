"""Stub parser for testing without tree-sitter complexity."""

from typing import List, Tuple
import re
from .symbol_extractor import Symbol, Relationship


class StubParser:
    """Simple regex-based parser for testing."""
    
    def parse_python(self, content: str, file_path: str) -> Tuple[List[Symbol], List[Relationship]]:
        """Parse Python code with regex."""
        symbols = []
        relationships = []
        lines = content.split('\n')
        
        # Find classes
        class_pattern = re.compile(r'^class\s+(\w+)(?:\((.*?)\))?:')
        # Find functions/methods
        func_pattern = re.compile(r'^(\s*)(?:async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?:')
        # Find docstrings
        docstring_pattern = re.compile(r'^\s*"""(.*?)"""', re.DOTALL)
        
        current_class = None
        
        for i, line in enumerate(lines):
            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                bases = class_match.group(2)
                
                symbol = Symbol(
                    name=class_name,
                    qualified_name=class_name,
                    symbol_type='class',
                    file_path=file_path,
                    line_start=i + 1,
                    line_end=i + 1,
                    language='python',
                    ast_hash=f"stub_{i}"
                )
                symbols.append(symbol)
                current_class = class_name
                
                # Add inheritance relationships
                if bases:
                    for base in bases.split(','):
                        base = base.strip()
                        if base:
                            rel = Relationship(
                                from_symbol=class_name,
                                to_symbol=base,
                                relationship_type='inherits',
                                file_path=file_path,
                                line_number=i + 1
                            )
                            relationships.append(rel)
            
            # Check for function/method
            func_match = func_pattern.match(line)
            if func_match:
                indent = func_match.group(1)
                func_name = func_match.group(2)
                params = func_match.group(3)
                return_type = func_match.group(4)
                
                # Determine if it's a method or function
                is_method = bool(indent) and current_class is not None
                symbol_type = 'method' if is_method else 'function'
                
                qualified_name = f"{current_class}.{func_name}" if is_method else func_name
                
                # Check if async
                is_async = 'async' in line
                
                symbol = Symbol(
                    name=func_name,
                    qualified_name=qualified_name,
                    symbol_type=symbol_type,
                    file_path=file_path,
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"def {func_name}({params})",
                    language='python',
                    is_async=is_async,
                    ast_hash=f"stub_{i}"
                )
                symbols.append(symbol)
                
                # Look for function calls in the next few lines
                for j in range(i + 1, min(i + 20, len(lines))):
                    call_line = lines[j]
                    if not call_line.strip() or (call_line.strip().startswith('"""') and j > i + 1):
                        break
                    
                    # Simple call detection
                    call_matches = re.findall(r'(\w+)\s*\(', call_line)
                    for call_name in call_matches:
                        if call_name not in ['if', 'for', 'while', 'print', 'return']:
                            rel = Relationship(
                                from_symbol=qualified_name,
                                to_symbol=call_name,
                                relationship_type='calls',
                                file_path=file_path,
                                line_number=j + 1
                            )
                            relationships.append(rel)
            
            # Reset current class if we're back at top level
            if line and not line[0].isspace():
                if not class_match:
                    current_class = None
        
        return symbols, relationships
    
    def parse_javascript(self, content: str, file_path: str) -> Tuple[List[Symbol], List[Relationship]]:
        """Parse JavaScript code with regex."""
        symbols = []
        relationships = []
        
        # Simple patterns for JS
        class_pattern = re.compile(r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{')
        func_pattern = re.compile(r'(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)')
        method_pattern = re.compile(r'(\w+)\s*\(([^)]*)\)\s*{')
        arrow_pattern = re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>')
        
        # Basic symbol extraction
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Classes
            class_match = class_pattern.search(line)
            if class_match:
                class_name = class_match.group(1)
                extends = class_match.group(2)
                
                symbol = Symbol(
                    name=class_name,
                    qualified_name=class_name,
                    symbol_type='class',
                    file_path=file_path,
                    line_start=i + 1,
                    line_end=i + 1,
                    language='javascript',
                    ast_hash=f"stub_{i}"
                )
                symbols.append(symbol)
                
                if extends:
                    rel = Relationship(
                        from_symbol=class_name,
                        to_symbol=extends,
                        relationship_type='inherits',
                        file_path=file_path,
                        line_number=i + 1
                    )
                    relationships.append(rel)
            
            # Functions
            func_match = func_pattern.search(line)
            if func_match:
                func_name = func_match.group(1)
                params = func_match.group(2)
                
                symbol = Symbol(
                    name=func_name,
                    qualified_name=func_name,
                    symbol_type='function',
                    file_path=file_path,
                    line_start=i + 1,
                    line_end=i + 1,
                    signature=f"function {func_name}({params})",
                    language='javascript',
                    is_async='async' in line,
                    ast_hash=f"stub_{i}"
                )
                symbols.append(symbol)
        
        return symbols, relationships