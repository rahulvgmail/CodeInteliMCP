"""Language registry for tree-sitter parsers."""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_go
import tree_sitter_rust

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """Manages tree-sitter language parsers and queries."""
    
    def __init__(self):
        self.languages: Dict[str, tree_sitter.Language] = {}
        self.parsers: Dict[str, tree_sitter.Parser] = {}
        self.queries: Dict[str, Dict[str, tree_sitter.Query]] = {}
        self._load_languages()
    
    def _load_languages(self):
        """Load all supported language parsers."""
        # For tree-sitter 0.20.x, languages are used directly without Language wrapper
        languages_to_load = [
            ('python', tree_sitter_python.language),
            ('javascript', tree_sitter_javascript.language),
            ('typescript', tree_sitter_typescript.language_typescript),
            ('tsx', tree_sitter_typescript.language_tsx),
            ('go', tree_sitter_go.language),
            ('rust', tree_sitter_rust.language),
        ]
        
        for lang_name, lang_factory in languages_to_load:
            try:
                # Get the language capsule
                lang_capsule = lang_factory()
                
                # Wrap it in a Language object
                language = tree_sitter.Language(lang_capsule, lang_name)
                self.languages[lang_name] = language
                
                # Create parser with tree-sitter 0.20.x API
                parser = tree_sitter.Parser()
                parser.set_language(language)
                self.parsers[lang_name] = parser
                
                logger.debug(f"Loaded {lang_name} language")
            except Exception as e:
                logger.warning(f"Failed to load {lang_name} language: {e}")
        
        # Add aliases
        if 'javascript' in self.languages:
            self.languages['js'] = self.languages['javascript']
            self.parsers['js'] = self.parsers['javascript']
        
        if 'typescript' in self.languages:
            self.languages['ts'] = self.languages['typescript']
            self.parsers['ts'] = self.parsers['typescript']
        
        # Load queries
        self._load_queries()
        
        logger.info(f"Loaded {len(self.languages)} languages")
    
    def _load_queries(self):
        """Load tree-sitter queries for each language."""
        queries_dir = Path(__file__).parent / "queries"
        
        for lang_name in ['python', 'javascript']:
            query_file = queries_dir / f"{lang_name}.scm"
            if query_file.exists():
                try:
                    query_text = query_file.read_text()
                    language = self.languages.get(lang_name)
                    if language:
                        # tree-sitter 0.20.x uses language.query() method
                        query = language.query(query_text)
                        self.queries[lang_name] = {
                            'symbols': query
                        }
                        logger.debug(f"Loaded queries for {lang_name}")
                except Exception as e:
                    logger.error(f"Failed to load queries for {lang_name}: {e}")
    
    def get_parser(self, language: str) -> Optional[tree_sitter.Parser]:
        """Get parser for a language."""
        return self.parsers.get(language.lower())
    
    def get_language(self, language: str) -> Optional[tree_sitter.Language]:
        """Get language object."""
        return self.languages.get(language.lower())
    
    def get_query(self, language: str, query_type: str = 'symbols') -> Optional[tree_sitter.Query]:
        """Get a query for a language."""
        lang_queries = self.queries.get(language.lower(), {})
        return lang_queries.get(query_type)
    
    def parse(self, code: bytes, language: str) -> Optional[tree_sitter.Tree]:
        """Parse code in the specified language."""
        parser = self.get_parser(language)
        if parser:
            return parser.parse(code)
        return None
    
    def get_language_from_extension(self, file_path: str) -> Optional[str]:
        """Determine language from file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.go': 'go',
            '.rs': 'rust',
        }
        
        path = Path(file_path)
        return ext_map.get(path.suffix.lower())
    
    def is_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language.lower() in self.languages