# 🔨 Tool Implementation Guide

## Overview

This guide provides implementation details for all MCP tools. Each tool should follow the established patterns and integrate properly with the DI system.

## Base Tool Structure

All tools inherit from `BaseTool` and follow this pattern:

```python
class ToolName(BaseTool):
    def __init__(self, **services):
        super().__init__(**services)
        # Store injected services
    
    async def validate_inputs(self, **kwargs):
        # Input validation
    
    async def _execute(self, **kwargs):
        # Tool logic
```

## 1. Search Tools Implementation

### FindUsagesTool

```python
# code_intelligence_mcp/tools/search_tools.py
from typing import List, Dict, Any, Optional, Set
from .base import BaseTool
from ..graph import GraphService
from ..vector import VectorService
from ..parser import ParserService
from ..cache import CacheService


class SearchTools:
    """Container for search-related tools."""
    
    def __init__(
        self,
        graph: GraphService,
        vector: VectorService,
        parser: ParserService,
        cache: Optional[CacheService] = None
    ):
        self.find_usages = FindUsagesTool(graph, vector, parser, cache)
        self.find_similar_code = FindSimilarCodeTool(graph, vector, parser, cache)
        self.search_by_pattern = SearchByPatternTool(graph, parser, cache)
        self.find_definitions = FindDefinitionsTool(graph, vector, cache)
        self.search_imports = SearchImportsTool(graph, cache)


class FindUsagesTool(BaseTool):
    """Find all usages of a symbol."""
    
    def __init__(
        self,
        graph: GraphService,
        vector: VectorService,
        parser: ParserService,
        cache: Optional[CacheService] = None
    ):
        super().__init__(
            graph=graph,
            vector=vector,
            parser=parser,
            cache=cache
        )
        self.graph = graph
        self.vector = vector
        self.parser = parser
        self.cache = cache
    
    async def validate_inputs(
        self,
        name: str,
        file_path: Optional[str] = None,
        **kwargs
    ):
        """Validate inputs."""
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        
        if file_path and not isinstance(file_path, str):
            raise ValueError("file_path must be a string")
    
    async def _execute(
        self,
        name: str,
        file_path: Optional[str] = None,
        include_indirect: bool = False,
        include_tests: bool = True,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Find all usages of a symbol."""
        # Try cache first
        cache_key = f"usages:{name}:{file_path}:{include_indirect}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached
        
        # Find the symbol
        symbols = await self.graph.find_symbol(
            name=name,
            file_path=file_path
        )
        
        if not symbols:
            # Try fuzzy search
            suggestions = await self._find_similar_symbols(name)
            return {
                "error": f"Symbol '{name}' not found",
                "suggestions": suggestions,
                "hint": "Try one of the suggested symbols or check the spelling"
            }
        
        # Handle multiple matches
        if len(symbols) > 1 and not file_path:
            return {
                "multiple_matches": True,
                "symbols": [
                    {
                        "name": s['name'],
                        "qualified_name": s['qualified_name'],
                        "type": s['symbol_type'],
                        "file": s['file_path'],
                        "line": s['line_start']
                    }
                    for s in symbols
                ],
                "hint": "Multiple symbols found. Specify file_path to narrow down."
            }
        
        # Get the primary symbol
        symbol = symbols[0]
        
        # Find direct usages
        usages = await self.graph.find_usages(symbol['id'])
        
        # Find indirect usages if requested
        indirect_usages = []
        if include_indirect:
            indirect_usages = await self._find_indirect_usages(
                symbol['id'],
                set(u['id'] for u in usages)
            )
        
        # Filter test files if needed
        if not include_tests:
            usages = [u for u in usages if not self._is_test_file(u['file_path'])]
            indirect_usages = [
                u for u in indirect_usages 
                if not self._is_test_file(u['file_path'])
            ]
        
        # Limit results
        all_usages = usages + indirect_usages
        truncated = len(all_usages) > limit
        all_usages = all_usages[:limit]
        
        # Enhance with context
        enhanced_usages = []
        for usage in all_usages:
            context = await self._get_usage_context(usage)
            enhanced_usages.append({
                **usage,
                'context': context,
                'is_indirect': usage in indirect_usages
            })
        
        # Group by file
        grouped = self._group_by_file(enhanced_usages)
        
        result = {
            'symbol': {
                'name': symbol['name'],
                'qualified_name': symbol['qualified_name'],
                'type': symbol['symbol_type'],
                'file': symbol['file_path'],
                'line': symbol['line_start'],
                'signature': symbol.get('signature'),
                'docstring': symbol.get('docstring')
            },
            'usages': enhanced_usages,
            'grouped_by_file': grouped,
            'summary': {
                'total_usages': len(enhanced_usages),
                'direct_usages': len(usages),
                'indirect_usages': len(indirect_usages),
                'files_affected': len(grouped),
                'truncated': truncated
            }
        }
        
        # Cache result
        if self.cache:
            await self.cache.set(cache_key, result, ttl=300)
        
        return result
    
    async def _find_similar_symbols(self, name: str) -> List[Dict[str, Any]]:
        """Find similar symbol names."""
        # Use vector search for similarity
        similar = await self.vector.find_similar_symbols(name, limit=5)
        
        # Also try fuzzy matching on graph
        fuzzy_matches = await self.graph.fuzzy_search_symbols(name, limit=5)
        
        # Combine and deduplicate
        all_suggestions = []
        seen = set()
        
        for s in similar + fuzzy_matches:
            key = f"{s['name']}:{s['file']}"
            if key not in seen:
                seen.add(key)
                all_suggestions.append({
                    'name': s['name'],
                    'type': s['type'],
                    'file': s['file'],
                    'similarity': s.get('similarity', 0)
                })
        
        # Sort by similarity
        all_suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        
        return all_suggestions[:5]
    
    async def _find_indirect_usages(
        self,
        symbol_id: int,
        direct_usage_ids: Set[int]
    ) -> List[Dict[str, Any]]:
        """Find indirect usages through call chains."""
        indirect = []
        
        # Get symbols that call the direct usages
        for usage_id in direct_usage_ids:
            callers = await self.graph.find_usages(usage_id)
            for caller in callers:
                if caller['id'] not in direct_usage_ids:
                    caller['call_path'] = await self._get_call_path(
                        caller['id'],
                        symbol_id
                    )
                    indirect.append(caller)
        
        return indirect
    
    async def _get_usage_context(
        self,
        usage: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get context around a usage."""
        file_path = usage['file_path']
        line = usage['line_start']
        
        # Get surrounding code
        context_lines = await self.parser.get_context_lines(
            file_path,
            line,
            before=3,
            after=3
        )
        
        # Find the specific usage in the line
        usage_line = context_lines[3] if len(context_lines) > 3 else ""
        
        return {
            'code': '\n'.join(context_lines),
            'usage_line': usage_line,
            'function': usage.get('parent_function'),
            'class': usage.get('parent_class'),
            'line_number': line
        }
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file."""
        test_patterns = ['test_', '_test.', '/tests/', '/test/', 'spec.']
        return any(pattern in file_path.lower() for pattern in test_patterns)
    
    def _group_by_file(
        self,
        usages: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group usages by file."""
        grouped = {}
        for usage in usages:
            file_path = usage['file_path']
            if file_path not in grouped:
                grouped[file_path] = []
            grouped[file_path].append(usage)
        
        # Sort usages within each file by line number
        for file_usages in grouped.values():
            file_usages.sort(key=lambda u: u['line_start'])
        
        return grouped
    
    async def _get_call_path(
        self,
        from_symbol_id: int,
        to_symbol_id: int
    ) -> List[str]:
        """Get the call path between two symbols."""
        path = await self.graph.find_path_between_symbols(
            from_symbol_id,
            to_symbol_id,
            relationship_type='calls'
        )
        
        return [
            f"{step['name']} ({step['file_path']}:{step['line_start']})"
            for step in path
        ]


class FindSimilarCodeTool(BaseTool):
    """Find code similar to a given snippet."""
    
    def __init__(
        self,
        graph: GraphService,
        vector: VectorService,
        parser: ParserService,
        cache: Optional[CacheService] = None
    ):
        super().__init__(
            graph=graph,
            vector=vector,
            parser=parser,
            cache=cache
        )
        self.vector = vector
        self.parser = parser
    
    async def validate_inputs(self, code_snippet: str, **kwargs):
        """Validate inputs."""
        if not code_snippet or not isinstance(code_snippet, str):
            raise ValueError("code_snippet must be a non-empty string")
        
        if len(code_snippet) < 10:
            raise ValueError("code_snippet too short for meaningful search")
    
    async def _execute(
        self,
        code_snippet: str,
        limit: int = 10,
        language: Optional[str] = None,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar code snippets."""
        # Search in vector store
        results = await self.vector.search_similar_code(
            code_snippet,
            limit=limit * 2,  # Get more to filter
            filters={'language': language} if language else None
        )
        
        # Filter by minimum similarity
        results = [
            r for r in results
            if r['similarity'] >= min_similarity
        ]
        
        # Limit results
        results = results[:limit]
        
        # Enhance with additional context
        enhanced_results = []
        for result in results:
            # Get symbol information
            symbols = await self.graph.get_symbols_in_range(
                result['metadata']['file'],
                result['metadata']['line_start'],
                result['metadata']['line_end']
            )
            
            enhanced_results.append({
                **result,
                'symbols': symbols,
                'complexity': await self._calculate_complexity(result['code']),
                'patterns': await self._identify_patterns(result['code'])
            })
        
        return {
            'query': code_snippet,
            'results': enhanced_results,
            'total_found': len(enhanced_results),
            'search_metadata': {
                'min_similarity': min_similarity,
                'language_filter': language
            }
        }
    
    async def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        # Simple heuristic - count control flow keywords
        keywords = [
            'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'switch', 'case', 'catch', 'finally'
        ]
        
        complexity = 1  # Base complexity
        for keyword in keywords:
            complexity += code.count(f' {keyword} ')
            complexity += code.count(f'\n{keyword} ')
        
        return complexity
    
    async def _identify_patterns(self, code: str) -> List[str]:
        """Identify common patterns in code."""
        patterns = []
        
        # Check for common patterns
        pattern_checks = {
            'error_handling': ['try', 'except', 'catch', 'finally'],
            'iteration': ['for', 'while', 'map', 'forEach'],
            'conditional': ['if', 'else', 'switch', 'case'],
            'function_definition': ['def', 'function', 'func', '=>'],
            'class_definition': ['class', 'struct', 'interface'],
            'async_code': ['async', 'await', 'promise', 'then'],
            'logging': ['log', 'logger', 'console.log', 'print'],
            'validation': ['validate', 'check', 'verify', 'assert']
        }
        
        for pattern_name, keywords in pattern_checks.items():
            if any(keyword in code.lower() for keyword in keywords):
                patterns.append(pattern_name)
        
        return patterns


class SearchByPatternTool(BaseTool):
    """Search code using tree-sitter patterns."""
    
    async def _execute(
        self,
        pattern: str,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search using tree-sitter query patterns."""
        # Get files to search
        files = await self.graph.get_indexed_files(
            language=language,
            pattern=file_pattern
        )
        
        results = []
        
        for file_info in files:
            if len(results) >= limit:
                break
            
            # Parse file with tree-sitter
            tree = await self.parser.parse_file(file_info['path'])
            if not tree:
                continue
            
            # Execute query
            matches = await self.parser.execute_query(
                tree,
                pattern,
                file_info['language']
            )
            
            for match in matches:
                if len(results) >= limit:
                    break
                
                results.append({
                    'file': file_info['path'],
                    'language': file_info['language'],
                    'match': match,
                    'context': await self._get_match_context(
                        file_info['path'],
                        match
                    )
                })
        
        return {
            'pattern': pattern,
            'results': results,
            'files_searched': len(files),
            'total_matches': len(results)
        }
    
    async def _get_match_context(
        self,
        file_path: str,
        match: Dict[str, Any]
    ) -> str:
        """Get context around a pattern match."""
        lines = await self.parser.get_context_lines(
            file_path,
            match['line_start'],
            before=2,
            after=2
        )
        return '\n'.join(lines)
```

## 2. Analysis Tools Implementation

### Impact Analysis and Dependencies

```python
# code_intelligence_mcp/tools/analysis_tools.py
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from .base import BaseTool
from ..graph import GraphService
from ..indexing import IndexingService


class AnalysisTools:
    """Container for analysis tools."""
    
    def __init__(
        self,
        graph: GraphService,
        indexing: IndexingService,
        cache: Optional[CacheService] = None
    ):
        self.analyze_impact = AnalyzeImpactTool(graph, cache)
        self.find_dependencies = FindDependenciesTool(graph, cache)
        self.detect_patterns = DetectPatternsTool(graph, indexing, cache)
        self.analyze_complexity = AnalyzeComplexityTool(graph, cache)
        self.find_dead_code = FindDeadCodeTool(graph, cache)


class AnalyzeImpactTool(BaseTool):
    """Analyze impact of changes to a symbol."""
    
    def __init__(self, graph: GraphService, cache: Optional[CacheService] = None):
        super().__init__(graph=graph, cache=cache)
        self.graph = graph
        self.cache = cache
    
    async def _execute(
        self,
        name: str,
        change_type: str = "modify",
        include_tests: bool = True,
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """Analyze impact of changing a symbol."""
        # Validate change type
        valid_types = ["modify", "delete", "rename", "signature_change"]
        if change_type not in valid_types:
            raise ValueError(f"change_type must be one of {valid_types}")
        
        # Find the symbol
        symbols = await self.graph.find_symbol(name=name)
        if not symbols:
            return {"error": f"Symbol '{name}' not found"}
        
        symbol = symbols[0]
        
        # Analyze based on change type
        if change_type == "delete":
            impact = await self._analyze_deletion_impact(symbol, max_depth)
        elif change_type == "rename":
            impact = await self._analyze_rename_impact(symbol, max_depth)
        elif change_type == "signature_change":
            impact = await self._analyze_signature_change_impact(symbol, max_depth)
        else:  # modify
            impact = await self._analyze_modification_impact(symbol, max_depth)
        
        # Filter test files if needed
        if not include_tests:
            impact = self._filter_test_files(impact)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(impact, change_type)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            impact,
            change_type,
            risk_score
        )
        
        return {
            'symbol': {
                'name': symbol['name'],
                'type': symbol['symbol_type'],
                'file': symbol['file_path']
            },
            'change_type': change_type,
            'impact': impact,
            'risk_score': risk_score,
            'recommendations': recommendations,
            'summary': self._generate_summary(impact)
        }
    
    async def _analyze_deletion_impact(
        self,
        symbol: Dict[str, Any],
        max_depth: int
    ) -> Dict[str, Any]:
        """Analyze impact of deleting a symbol."""
        # Find all direct usages
        direct_usages = await self.graph.find_usages(symbol['id'])
        
        # Find transitive impact
        all_affected = set()
        to_process = [(u['id'], 1) for u in direct_usages]
        
        while to_process:
            current_id, depth = to_process.pop(0)
            if depth > max_depth or current_id in all_affected:
                continue
            
            all_affected.add(current_id)
            
            # Find what depends on this symbol
            dependents = await self.graph.find_usages(current_id)
            to_process.extend([(d['id'], depth + 1) for d in dependents])
        
        # Get details for all affected symbols
        affected_details = []
        for symbol_id in all_affected:
            details = await self.graph.get_symbol_details(symbol_id)
            affected_details.append(details)
        
        return {
            'directly_affected': direct_usages,
            'transitively_affected': affected_details,
            'total_affected': len(all_affected),
            'files_affected': len(set(s['file_path'] for s in affected_details)),
            'would_break': True  # Deletion always breaks dependents
        }
    
    async def _analyze_signature_change_impact(
        self,
        symbol: Dict[str, Any],
        max_depth: int
    ) -> Dict[str, Any]:
        """Analyze impact of changing function signature."""
        # Get all call sites
        call_sites = await self.graph.find_relationships(
            to_symbol_id=symbol['id'],
            relationship_type='calls'
        )
        
        # Analyze each call site
        affected_calls = []
        for call in call_sites:
            context = await self._get_call_context(call)
            affected_calls.append({
                **call,
                'context': context,
                'needs_update': True  # All calls need update for signature change
            })
        
        return {
            'call_sites': affected_calls,
            'total_updates_needed': len(affected_calls),
            'files_to_update': len(set(c['file_path'] for c in affected_calls)),
            'estimated_effort': self._estimate_update_effort(affected_calls)
        }
    
    def _calculate_risk_score(
        self,
        impact: Dict[str, Any],
        change_type: str
    ) -> Dict[str, Any]:
        """Calculate risk score for the change."""
        # Base scores for change types
        base_scores = {
            'delete': 0.8,
            'rename': 0.5,
            'signature_change': 0.7,
            'modify': 0.3
        }
        
        score = base_scores[change_type]
        
        # Adjust based on impact
        if 'total_affected' in impact:
            if impact['total_affected'] > 50:
                score += 0.2
            elif impact['total_affected'] > 20:
                score += 0.1
        
        if 'files_affected' in impact:
            if impact['files_affected'] > 20:
                score += 0.1
        
        # Cap at 1.0
        score = min(score, 1.0)
        
        # Determine level
        if score >= 0.7:
            level = 'high'
        elif score >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'score': score,
            'level': level,
            'factors': {
                'change_type': change_type,
                'symbols_affected': impact.get('total_affected', 0),
                'files_affected': impact.get('files_affected', 0)
            }
        }
    
    def _generate_recommendations(
        self,
        impact: Dict[str, Any],
        change_type: str,
        risk_score: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []
        
        if risk_score['level'] == 'high':
            recommendations.append(
                "⚠️ High-risk change. Consider breaking into smaller changes."
            )
            recommendations.append(
                "📝 Document the change and notify team members."
            )
        
        if change_type == 'delete':
            recommendations.append(
                "🔍 Review all usages before deletion."
            )
            recommendations.append(
                "💡 Consider deprecation instead of immediate deletion."
            )
        
        if change_type == 'signature_change':
            recommendations.append(
                "🔧 Update all call sites to match new signature."
            )
            recommendations.append(
                "📚 Update documentation and type hints."
            )
        
        if impact.get('files_affected', 0) > 10:
            recommendations.append(
                "🧪 Extensive testing recommended due to wide impact."
            )
        
        return recommendations


class FindDependenciesTool(BaseTool):
    """Find and analyze dependencies."""
    
    async def _execute(
        self,
        name: str,
        direction: str = "both",
        max_depth: int = 3,
        include_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Find dependencies of a symbol."""
        # Find the symbol
        symbols = await self.graph.find_symbol(name=name)
        if not symbols:
            return {"error": f"Symbol '{name}' not found"}
        
        symbol = symbols[0]
        
        # Get dependencies based on direction
        depends_on = []
        depended_by = []
        
        if direction in ["both", "depends_on"]:
            depends_on = await self._get_dependencies(
                symbol['id'],
                max_depth,
                include_types
            )
        
        if direction in ["both", "depended_by"]:
            depended_by = await self._get_dependents(
                symbol['id'],
                max_depth,
                include_types
            )
        
        # Build dependency graph
        graph = self._build_dependency_graph(
            symbol,
            depends_on,
            depended_by
        )
        
        # Detect circular dependencies
        cycles = await self._detect_cycles_involving_symbol(symbol['id'])
        
        return {
            'symbol': {
                'name': symbol['name'],
                'type': symbol['symbol_type'],
                'file': symbol['file_path']
            },
            'depends_on': depends_on,
            'depended_by': depended_by,
            'graph': graph,
            'circular_dependencies': cycles,
            'metrics': {
                'fan_in': len(depended_by),
                'fan_out': len(depends_on),
                'instability': self._calculate_instability(
                    len(depends_on),
                    len(depended_by)
                )
            }
        }
    
    async def _get_dependencies(
        self,
        symbol_id: int,
        max_depth: int,
        include_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Get what this symbol depends on."""
        dependencies = await self.graph.find_dependencies(
            symbol_id,
            relationship_types=include_types,
            max_depth=max_depth
        )
        
        # Group by depth
        by_depth = defaultdict(list)
        for dep in dependencies:
            by_depth[dep['depth']].append(dep)
        
        return {
            'all': dependencies,
            'by_depth': dict(by_depth),
            'direct': by_depth[1]
        }
    
    def _calculate_instability(self, fan_out: int, fan_in: int) -> float:
        """Calculate instability metric (I = Ce / (Ce + Ca))."""
        if fan_out + fan_in == 0:
            return 0.0
        return fan_out / (fan_out + fan_in)
    
    def _build_dependency_graph(
        self,
        root: Dict[str, Any],
        depends_on: List[Dict[str, Any]],
        depended_by: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build a graph representation of dependencies."""
        nodes = {root['id']: root}
        edges = []
        
        # Add depends_on relationships
        for dep_group in depends_on.get('all', []):
            nodes[dep_group['id']] = dep_group
            edges.append({
                'from': root['id'],
                'to': dep_group['id'],
                'type': dep_group.get('relationship_type', 'depends_on')
            })
        
        # Add depended_by relationships  
        for dep_group in depended_by.get('all', []):
            nodes[dep_group['id']] = dep_group
            edges.append({
                'from': dep_group['id'],
                'to': root['id'],
                'type': dep_group.get('relationship_type', 'depends_on')
            })
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges
        }


class DetectPatternsTool(BaseTool):
    """Detect design patterns and code smells."""
    
    async def _execute(
        self,
        scope: str = "project",
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect patterns in code."""
        pattern_types = pattern_types or [
            'singleton', 'factory', 'observer', 'strategy',
            'repository', 'adapter', 'decorator'
        ]
        
        detected_patterns = {}
        
        for pattern_type in pattern_types:
            detector = self._get_pattern_detector(pattern_type)
            if detector:
                patterns = await detector.detect(scope)
                if patterns:
                    detected_patterns[pattern_type] = patterns
        
        # Also detect anti-patterns
        anti_patterns = await self._detect_anti_patterns(scope)
        
        return {
            'design_patterns': detected_patterns,
            'anti_patterns': anti_patterns,
            'summary': self._summarize_patterns(
                detected_patterns,
                anti_patterns
            )
        }
    
    def _get_pattern_detector(self, pattern_type: str):
        """Get pattern detector for type."""
        detectors = {
            'singleton': SingletonDetector(self.graph),
            'factory': FactoryDetector(self.graph),
            'observer': ObserverDetector(self.graph),
            'repository': RepositoryDetector(self.graph)
        }
        return detectors.get(pattern_type)
    
    async def _detect_anti_patterns(self, scope: str) -> Dict[str, Any]:
        """Detect common anti-patterns."""
        anti_patterns = {}
        
        # God classes (too many methods/responsibilities)
        god_classes = await self._detect_god_classes()
        if god_classes:
            anti_patterns['god_classes'] = god_classes
        
        # Long methods
        long_methods = await self._detect_long_methods()
        if long_methods:
            anti_patterns['long_methods'] = long_methods
        
        # Circular dependencies
        circular_deps = await self.graph.find_circular_dependencies()
        if circular_deps:
            anti_patterns['circular_dependencies'] = circular_deps
        
        # Duplicate code
        duplicates = await self._detect_duplicate_code()
        if duplicates:
            anti_patterns['duplicate_code'] = duplicates
        
        return anti_patterns
```

## 3. Index Tools Implementation

### Index Management Tools

```python
# code_intelligence_mcp/tools/index_tools.py
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from .base import BaseTool
from ..graph import GraphService
from ..vector import VectorService
from ..indexing import IndexingService, FileWatcher
from ..config import get_config


class IndexTools:
    """Container for indexing tools."""
    
    def __init__(
        self,
        graph: GraphService,
        vector: VectorService,
        indexing: IndexingService,
        watcher: FileWatcher
    ):
        self.update_index = UpdateIndexTool(indexing, watcher)
        self.index_stats = IndexStatsTool(graph, vector)
        self.validate_index = ValidateIndexTool(graph, vector, indexing)
        self.manage_index = ManageIndexTool(graph, vector, indexing)


class UpdateIndexTool(BaseTool):
    """Tool for updating the code index."""
    
    def __init__(
        self,
        indexing: IndexingService,
        watcher: FileWatcher
    ):
        super().__init__(indexing=indexing, watcher=watcher)
        self.indexing = indexing
        self.watcher = watcher
    
    async def _execute(
        self,
        paths: Optional[List[str]] = None,
        full_rebuild: bool = False,
        watch: bool = True
    ) -> Dict[str, Any]:
        """Update the code index."""
        config = get_config()
        start_time = datetime.now()
        
        if full_rebuild:
            # Full project rebuild
            stats = await self.indexing.index_project(
                config.project_root,
                force_reindex=True
            )
            
            # Start watching if requested
            if watch:
                await self.watcher.watch_directory(config.project_root)
            
        elif paths:
            # Update specific paths
            path_objects = [Path(p) for p in paths]
            stats = await self.indexing.update_files(path_objects)
            
        else:
            # Smart incremental update
            stats = await self.indexing.smart_update()
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        stats['duration_seconds'] = duration
        
        # Add performance metrics
        if stats.get('files_processed', 0) > 0:
            stats['files_per_second'] = (
                stats['files_processed'] / duration
            )
            stats['symbols_per_second'] = (
                stats.get('symbols_extracted', 0) / duration
            )
        
        return {
            'operation': 'full_rebuild' if full_rebuild else 'incremental',
            'stats': stats,
            'watching': watch and not paths,
            'timestamp': datetime.now().isoformat()
        }


class IndexStatsTool(BaseTool):
    """Tool for getting index statistics."""
    
    async def _execute(self, detailed: bool = False) -> Dict[str, Any]:
        """Get current index statistics."""
        # Get basic stats from graph
        graph_stats = await self.graph.get_statistics()
        
        # Get vector store stats
        vector_stats = await self.vector.get_statistics()
        
        # Combine stats
        stats = {
            'symbols': {
                'total': graph_stats['total_symbols'],
                'by_type': graph_stats['symbols_by_type'],
                'unique_names': await self._count_unique_names()
            },
            'relationships': {
                'total': graph_stats['total_relationships'],
                'by_type': graph_stats['relationships_by_type']
            },
            'files': {
                'total': graph_stats['total_files'],
                'by_language': await self._count_files_by_language()
            },
            'vectors': {
                'total_embeddings': vector_stats['total_embeddings'],
                'collections': vector_stats['collections']
            },
            'storage': {
                'graph_size_mb': graph_stats['db_size_mb'],
                'vector_size_mb': vector_stats['storage_size_mb'],
                'total_size_mb': (
                    graph_stats['db_size_mb'] +
                    vector_stats['storage_size_mb']
                )
            }
        }
        
        if detailed:
            # Add detailed statistics
            stats['quality_metrics'] = await self._calculate_quality_metrics()
            stats['coverage'] = await self._calculate_coverage()
            stats['performance'] = await self._get_performance_metrics()
        
        return stats
    
    async def _count_unique_names(self) -> int:
        """Count unique symbol names."""
        result = await self.graph.execute_query(
            "SELECT COUNT(DISTINCT name) FROM symbols"
        )
        return result[0][0] if result else 0
    
    async def _count_files_by_language(self) -> Dict[str, int]:
        """Count files by programming language."""
        result = await self.graph.execute_query("""
            SELECT language, COUNT(*) as count
            FROM files
            GROUP BY language
        """)
        return dict(result)
    
    async def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate code quality metrics."""
        return {
            'avg_file_complexity': await self._get_avg_complexity(),
            'symbols_without_docs': await self._count_undocumented(),
            'test_coverage_estimate': await self._estimate_test_coverage(),
            'circular_dependencies': len(
                await self.graph.find_circular_dependencies()
            )
        }


class ValidateIndexTool(BaseTool):
    """Tool for validating index integrity."""
    
    async def _execute(
        self,
        fix_issues: bool = False
    ) -> Dict[str, Any]:
        """Validate the index integrity."""
        issues = []
        
        # Check for orphaned symbols
        orphaned = await self._find_orphaned_symbols()
        if orphaned:
            issues.append({
                'type': 'orphaned_symbols',
                'count': len(orphaned),
                'severity': 'medium',
                'description': 'Symbols without valid file references'
            })
            
            if fix_issues:
                await self._fix_orphaned_symbols(orphaned)
        
        # Check for missing relationships
        missing_rels = await self._find_missing_relationships()
        if missing_rels:
            issues.append({
                'type': 'missing_relationships',
                'count': len(missing_rels),
                'severity': 'low',
                'description': 'Potential relationships not indexed'
            })
        
        # Check for stale data
        stale_files = await self._find_stale_files()
        if stale_files:
            issues.append({
                'type': 'stale_files',
                'count': len(stale_files),
                'severity': 'high',
                'description': 'Files changed since last index'
            })
            
            if fix_issues:
                await self._reindex_stale_files(stale_files)
        
        # Check vector store consistency
        vector_issues = await self._validate_vector_store()
        issues.extend(vector_issues)
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'fixed': fix_issues,
            'recommendations': self._generate_recommendations(issues)
        }
    
    async def _find_orphaned_symbols(self) -> List[Dict[str, Any]]:
        """Find symbols pointing to non-existent files."""
        return await self.graph.execute_query("""
            SELECT s.*
            FROM symbols s
            LEFT JOIN files f ON s.file_path = f.path
            WHERE f.id IS NULL
        """)
    
    def _generate_recommendations(
        self,
        issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        if any(i['type'] == 'stale_files' for i in issues):
            recommendations.append(
                "Run incremental index update to refresh stale files"
            )
        
        if any(i['type'] == 'orphaned_symbols' for i in issues):
            recommendations.append(
                "Run validation with fix_issues=True to clean orphaned data"
            )
        
        return recommendations
```

## 4. Visualization Tools

### Graph Visualization Tools

```python
# code_intelligence_mcp/tools/viz_tools.py
from typing import Dict, Any, List, Optional
from .base import BaseTool
from ..graph import GraphService


class VizTools:
    """Container for visualization tools."""
    
    def __init__(self, graph: GraphService):
        self.get_call_graph = GetCallGraphTool(graph)
        self.get_dependency_graph = GetDependencyGraphTool(graph)
        self.get_file_structure = GetFileStructureTool(graph)
        self.get_inheritance_tree = GetInheritanceTreeTool(graph)


class GetCallGraphTool(BaseTool):
    """Get call graph visualization data."""
    
    async def _execute(
        self,
        name: str,
        max_depth: int = 3,
        include_external: bool = False,
        layout: str = "hierarchical"
    ) -> Dict[str, Any]:
        """Get call graph for visualization."""
        # Find the root symbol
        symbols = await self.graph.find_symbol(name=name)
        if not symbols:
            return {"error": f"Symbol '{name}' not found"}
        
        symbol = symbols[0]
        
        # Get call graph data
        graph_data = await self.graph.get_call_graph(
            symbol['id'],
            max_depth
        )
        
        # Enhance nodes with additional data
        enhanced_nodes = []
        for node in graph_data['nodes']:
            symbol_details = await self.graph.get_symbol_details(node['id'])
            enhanced_nodes.append({
                **node,
                'type': symbol_details['symbol_type'],
                'file': symbol_details['file_path'],
                'line': symbol_details['line_start'],
                'complexity': symbol_details.get('complexity', 1),
                'is_root': node['id'] == symbol['id']
            })
        
        # Calculate layout positions
        positions = self._calculate_layout(
            enhanced_nodes,
            graph_data['edges'],
            layout
        )
        
        # Add positions to nodes
        for node, pos in zip(enhanced_nodes, positions):
            node['x'] = pos[0]
            node['y'] = pos[1]
        
        return {
            'root': symbol,
            'nodes': enhanced_nodes,
            'edges': graph_data['edges'],
            'layout': layout,
            'stats': {
                'total_nodes': len(enhanced_nodes),
                'total_edges': len(graph_data['edges']),
                'max_depth_reached': max(
                    e.get('depth', 0) for e in graph_data['edges']
                ) if graph_data['edges'] else 0
            }
        }
    
    def _calculate_layout(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout_type: str
    ) -> List[Tuple[float, float]]:
        """Calculate node positions for layout."""
        if layout_type == "hierarchical":
            return self._hierarchical_layout(nodes, edges)
        elif layout_type == "force":
            return self._force_directed_layout(nodes, edges)
        else:
            return self._circular_layout(nodes)
    
    def _hierarchical_layout(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[Tuple[float, float]]:
        """Calculate hierarchical layout positions."""
        # Group nodes by depth
        node_depths = {}
        root_node = next(n for n in nodes if n.get('is_root'))
        node_depths[root_node['id']] = 0
        
        # Calculate depths
        changed = True
        while changed:
            changed = False
            for edge in edges:
                from_id, to_id = edge['from'], edge['to']
                if from_id in node_depths and to_id not in node_depths:
                    node_depths[to_id] = node_depths[from_id] + 1
                    changed = True
        
        # Group by depth
        depth_groups = {}
        for node in nodes:
            depth = node_depths.get(node['id'], 0)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node)
        
        # Calculate positions
        positions = []
        y_spacing = 100
        x_spacing = 150
        
        for depth, group in sorted(depth_groups.items()):
            y = depth * y_spacing
            x_offset = -(len(group) - 1) * x_spacing / 2
            
            for i, node in enumerate(group):
                x = x_offset + i * x_spacing
                positions.append((x, y))
        
        return positions


class GetDependencyGraphTool(BaseTool):
    """Get dependency graph visualization data."""
    
    async def _execute(
        self,
        scope: str = "file",
        file_path: Optional[str] = None,
        module_path: Optional[str] = None,
        max_nodes: int = 50
    ) -> Dict[str, Any]:
        """Get dependency graph for visualization."""
        if scope == "file" and file_path:
            return await self._get_file_dependencies(file_path)
        elif scope == "module" and module_path:
            return await self._get_module_dependencies(module_path)
        else:
            return await self._get_project_dependencies(max_nodes)
    
    async def _get_file_dependencies(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """Get dependencies for a single file."""
        # Get all symbols in file
        symbols = await self.graph.get_file_symbols(file_path)
        
        # Get imports
        imports = await self.graph.get_file_imports(file_path)
        
        # Build dependency graph
        nodes = []
        edges = []
        
        # Add current file as root
        nodes.append({
            'id': file_path,
            'name': Path(file_path).name,
            'type': 'file',
            'is_root': True
        })
        
        # Add imported files
        for imp in imports:
            nodes.append({
                'id': imp['imported_file'],
                'name': Path(imp['imported_file']).name,
                'type': 'file',
                'is_external': imp.get('is_external', False)
            })
            
            edges.append({
                'from': file_path,
                'to': imp['imported_file'],
                'type': 'imports'
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'scope': 'file',
            'root': file_path
        }
```

## 5. Error Handling and Validation

### Common Error Patterns

```python
# code_intelligence_mcp/tools/errors.py
from typing import Optional, Dict, Any


class ToolError(Exception):
    """Base exception for tool errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class SymbolNotFoundError(ToolError):
    """Symbol not found error."""
    
    def __init__(self, symbol_name: str, suggestions: List[str] = None):
        super().__init__(
            f"Symbol '{symbol_name}' not found",
            "SYMBOL_NOT_FOUND",
            {
                'symbol_name': symbol_name,
                'suggestions': suggestions or []
            }
        )


class InvalidPatternError(ToolError):
    """Invalid tree-sitter pattern error."""
    
    def __init__(self, pattern: str, error_message: str):
        super().__init__(
            f"Invalid pattern: {error_message}",
            "INVALID_PATTERN",
            {
                'pattern': pattern,
                'parse_error': error_message
            }
        )


# Error handling decorator
def handle_tool_errors(func):
    """Decorator to handle tool errors consistently."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ToolError as e:
            return {
                'error': str(e),
                'error_code': e.error_code,
                'details': e.details
            }
        except Exception as e:
            return {
                'error': f"Unexpected error: {str(e)}",
                'error_code': 'INTERNAL_ERROR',
                'details': {'exception_type': type(e).__name__}
            }
    
    return wrapper
```

## Tool Registration

### Registering Tools with MCP Server

```python
# code_intelligence_mcp/server.py (updated section)
def _register_all_tools(self):
    """Register all tools with the MCP server."""
    # Initialize tool containers
    self.search_tools = SearchTools(
        self.container.get(GraphService),
        self.container.get(VectorService),
        self.container.get(ParserService),
        self.container.get(CacheService)
    )
    
    # Register search tools
    self._register_tool(
        "find_usages",
        self.search_tools.find_usages,
        "Find all usages of a function, class, or variable"
    )
    
    self._register_tool(
        "find_similar_code",
        self.search_tools.find_similar_code,
        "Find code similar to the given snippet"
    )
    
    # Continue registering all tools...

def _register_tool(self, name: str, tool: BaseTool, description: str):
    """Register a single tool."""
    @self.server.tool(name=name, description=description)
    @handle_tool_errors
    async def tool_wrapper(**kwargs):
        return await tool.execute(**kwargs)
```

## Testing Tools

### Example Tool Tests

```python
# tests/test_tools/test_find_usages.py
import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_find_usages_basic(mock_container):
    """Test basic usage finding."""
    # Setup mocks
    graph_mock = mock_container.get(GraphService)
    graph_mock.find_symbol.return_value = [{
        'id': 1,
        'name': 'test_func',
        'file_path': 'test.py'
    }]
    graph_mock.find_usages.return_value = [
        {
            'id': 2,
            'name': 'caller_func',
            'file_path': 'caller.py',
            'line_start': 10
        }
    ]
    
    # Create tool
    tool = FindUsagesTool(
        graph_mock,
        AsyncMock(),  # vector
        AsyncMock(),  # parser
        None  # cache
    )
    
    # Execute
    result = await tool.execute(name="test_func")
    
    # Verify
    assert result['symbol']['name'] == 'test_func'
    assert len(result['usages']) == 1
    assert result['usages'][0]['name'] == 'caller_func'


@pytest.mark.asyncio
async def test_find_usages_not_found(mock_container):
    """Test when symbol is not found."""
    graph_mock = mock_container.get(GraphService)
    graph_mock.find_symbol.return_value = []
    
    vector_mock = AsyncMock()
    vector_mock.find_similar_symbols.return_value = [
        {'name': 'test_function', 'similarity': 0.8}
    ]
    
    tool = FindUsagesTool(graph_mock, vector_mock, AsyncMock(), None)
    
    result = await tool.execute(name="test_func")
    
    assert 'error' in result
    assert 'suggestions' in result
    assert result['suggestions'][0]['name'] == 'test_function'
```

## Performance Considerations

### Optimization Strategies

1. **Caching**: Cache frequently accessed data
2. **Batch Operations**: Process multiple items together
3. **Async Execution**: Use async/await properly
4. **Query Optimization**: Optimize database queries
5. **Lazy Loading**: Load data only when needed

### Example Optimized Tool

```python
class OptimizedFindUsagesTool(FindUsagesTool):
    """Optimized version with better performance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._symbol_cache = {}
        self._usage_cache = {}
    
    async def _execute(self, name: str, **kwargs) -> Dict[str, Any]:
        # Check cache first
        cache_key = f"{name}:{kwargs}"
        if cache_key in self._usage_cache:
            return self._usage_cache[cache_key]
        
        # Use batch operations
        if name in self._symbol_cache:
            symbol = self._symbol_cache[name]
        else:
            # Batch fetch related symbols
            symbols = await self.graph.batch_find_symbols([name])
            for s in symbols:
                self._symbol_cache[s['name']] = s
            symbol = self._symbol_cache.get(name)
        
        # Continue with optimized logic...
```

This completes the tool implementation guide. Each tool should follow these patterns for consistency and maintainability.
