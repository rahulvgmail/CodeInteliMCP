"""Parser service for code analysis."""

import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import aiofiles

from ..core import BaseService, CacheableService
from ..config import ParserConfig
from .language_registry import LanguageRegistry
from .symbol_extractor import SymbolExtractor, Symbol, Relationship

logger = logging.getLogger(__name__)


class ParserService(CacheableService):
    """Service for parsing source code and extracting symbols."""
    
    def __init__(self, config: ParserConfig, cache_service=None):
        super().__init__(cache_service=cache_service, config=config)
        self.config = config
        self._registry: Optional[LanguageRegistry] = None
        self._extractor: Optional[SymbolExtractor] = None
        self._stub_parser = None
    
    async def _initialize(self) -> None:
        """Initialize parser service."""
        try:
            self._registry = LanguageRegistry()
            # Check if any languages loaded
            if not self._registry.languages:
                raise RuntimeError("No languages loaded")
            self._extractor = SymbolExtractor(self._registry)
        except Exception as e:
            # Fall back to stub parser
            self._logger.warning(f"Failed to initialize tree-sitter, using stub parser: {e}")
            from .stub_parser import StubParser
            self._stub_parser = StubParser()
            self._registry = None
            self._extractor = None
        self._logger.info("ParserService initialized")
    
    async def analyze_file(
        self,
        file_path: Path,
        content: Optional[bytes] = None,
        language: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Analyze a file and extract symbols and relationships."""
        file_str = str(file_path)
        
        # Check cache if enabled
        if use_cache and self._cache:
            cache_key = f"parse:{file_str}:{language or 'auto'}"
            cached = await self._get_cached(cache_key)
            if cached:
                return cached
        
        # Read file if content not provided
        if content is None:
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    content = await f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return [], []
        
        # Determine language if not provided
        if not language:
            language = self.get_language_from_file(file_str)
            if not language:
                logger.warning(f"Could not determine language for {file_str}")
                return [], []
        
        # Extract symbols
        if self._extractor:
            symbols, relationships = self._extractor.extract_from_file(
                file_str,
                content,
                language
            )
        else:
            # Use stub parser
            content_str = content.decode('utf-8')
            if language == 'python':
                symbols, relationships = self._stub_parser.parse_python(content_str, file_str)
            elif language in ['javascript', 'js']:
                symbols, relationships = self._stub_parser.parse_javascript(content_str, file_str)
            else:
                logger.warning(f"Unsupported language for stub parser: {language}")
                symbols, relationships = [], []
        
        # Cache results if enabled
        if use_cache and self._cache:
            await self._set_cached(
                cache_key,
                (symbols, relationships),
                ttl=self.config.cache_ttl
            )
        
        return symbols, relationships
    
    async def parse_content(
        self,
        content: str,
        language: str,
        file_path: str = "<string>"
    ) -> Tuple[List[Symbol], List[Relationship]]:
        """Parse content directly without reading from file."""
        if self._extractor:
            content_bytes = content.encode('utf-8')
            return self._extractor.extract_from_file(
                file_path,
                content_bytes,
                language
            )
        else:
            # Use stub parser
            if language == 'python':
                return self._stub_parser.parse_python(content, file_path)
            elif language in ['javascript', 'js']:
                return self._stub_parser.parse_javascript(content, file_path)
            else:
                return [], []
    
    async def get_context_lines(
        self,
        file_path: str,
        line: int,
        before: int = 3,
        after: int = 3
    ) -> List[str]:
        """Get context lines around a specific line number."""
        try:
            async with aiofiles.open(file_path, 'r') as f:
                lines = await f.readlines()
            
            start = max(0, line - before - 1)
            end = min(len(lines), line + after)
            
            return [
                f"{i+1}: {lines[i].rstrip()}"
                for i in range(start, end)
            ]
        except Exception as e:
            logger.error(f"Failed to get context lines from {file_path}: {e}")
            return []
    
    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        if not self._registry:
            return False
        return self._registry.is_supported(language)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        if not self._registry:
            return []
        return list(self._registry.languages.keys())
    
    def get_language_from_file(self, file_path: str) -> Optional[str]:
        """Determine language from file path."""
        if self._registry:
            return self._registry.get_language_from_extension(file_path)
        else:
            # Simple extension mapping for stub parser
            ext_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.jsx': 'javascript',
                '.ts': 'typescript',
                '.tsx': 'tsx',
                '.go': 'go',
                '.rs': 'rust',
            }
            from pathlib import Path
            return ext_map.get(Path(file_path).suffix.lower())
    
    async def extract_imports(
        self,
        file_path: Path,
        content: Optional[bytes] = None
    ) -> List[Dict[str, Any]]:
        """Extract import statements from a file."""
        # This would be implemented with language-specific logic
        # For now, return empty list
        return []
    
    async def calculate_complexity(
        self,
        file_path: Path,
        content: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Calculate cyclomatic complexity and other metrics."""
        # This would be implemented with language-specific logic
        # For now, return basic metrics
        return {
            "cyclomatic_complexity": 1,
            "cognitive_complexity": 1,
            "lines_of_code": 0
        }