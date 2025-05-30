"""Configuration management for Code Intelligence MCP."""

from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import os


class IndexingConfig(BaseModel):
    """Configuration for indexing behavior."""
    
    auto_index_threshold: int = Field(
        default=5,
        description="Number of file changes before auto-indexing"
    )
    incremental_batch_size: int = Field(
        default=10,
        description="Files to process in each incremental batch"
    )
    full_index_file_limit: int = Field(
        default=10000,
        description="Maximum files to index in full rebuild"
    )
    ignore_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.pyc", "__pycache__", "*.egg-info",
            "node_modules", ".git", ".venv", "venv",
            "dist", "build", "*.min.js", "*.map"
        ],
        description="Patterns to ignore during indexing"
    )
    file_size_limit_mb: float = Field(
        default=10.0,
        description="Skip files larger than this (in MB)"
    )


class GraphConfig(BaseModel):
    """Configuration for DuckDB graph database."""
    
    db_path: Path = Field(
        default=Path("./data/code_graph.duckdb"),
        description="Path to DuckDB database file"
    )
    connection_timeout: int = Field(
        default=5000,
        description="Connection timeout in milliseconds"
    )
    cache_size_mb: int = Field(
        default=256,
        description="DuckDB cache size in MB"
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress during operations"
    )


class VectorConfig(BaseModel):
    """Configuration for ChromaDB vector store."""
    
    chroma_path: Path = Field(
        default=Path("./data/chroma"),
        description="ChromaDB persistence directory"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model for code embeddings"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    collection_name: str = Field(
        default="code_intelligence",
        description="Main collection name"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for results"
    )


class ParserConfig(BaseModel):
    """Configuration for Tree-sitter parsing."""
    
    supported_languages: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "python": [".py", ".pyw"],
            "javascript": [".js", ".jsx", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "go": [".go"],
            "rust": [".rs"],
            "java": [".java"],
            "c": [".c", ".h"],
            "cpp": [".cpp", ".cc", ".hpp", ".hh", ".cxx"],
        },
        description="Language to file extension mapping"
    )
    parse_timeout_ms: int = Field(
        default=5000,
        description="Timeout for parsing a single file"
    )
    max_file_size_mb: float = Field(
        default=5.0,
        description="Maximum file size to parse"
    )


class CacheConfig(BaseModel):
    """Configuration for caching system."""
    
    cache_directory: Path = Field(
        default=Path("./data/cache"),
        description="Directory for cache files"
    )
    ast_cache_ttl_hours: int = Field(
        default=24,
        description="AST cache time-to-live in hours"
    )
    query_cache_size: int = Field(
        default=1000,
        description="Number of query results to cache"
    )
    enable_cache: bool = Field(
        default=True,
        description="Enable caching system"
    )


class RepositoryConfig(BaseModel):
    """Configuration for a single repository."""
    
    name: str = Field(description="Human-readable repository name")
    path: Path = Field(description="Path to repository root")
    description: Optional[str] = Field(default=None, description="Repository description")
    active: bool = Field(default=False, description="Whether this is the active repository")


class Config(BaseModel):
    """Main configuration object."""
    
    repositories: List[RepositoryConfig] = Field(
        default_factory=list,
        description="List of repositories to manage"
    )
    active_repository: Optional[str] = Field(
        default=None,
        description="Name of the currently active repository"
    )
    data_directory: Path = Field(
        default=Path("./data"),
        description="Base directory for all data storage"
    )
    
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Performance settings
    max_workers: int = Field(
        default_factory=lambda: min(4, os.cpu_count() or 1),
        description="Maximum parallel workers"
    )
    
    # MCP specific
    mcp_server_name: str = Field(
        default="code-intelligence",
        description="Name of the MCP server"
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Ensure all directories exist."""
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.graph.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector.chroma_path.mkdir(parents=True, exist_ok=True)
        self.cache.cache_directory.mkdir(parents=True, exist_ok=True)
    
    def get_active_repository(self) -> Optional[RepositoryConfig]:
        """Get the currently active repository configuration."""
        if not self.active_repository:
            return self.repositories[0] if self.repositories else None
        
        for repo in self.repositories:
            if repo.name == self.active_repository:
                return repo
        return None
    
    def add_repository(self, name: str, path: Path, description: Optional[str] = None, make_active: bool = False) -> None:
        """Add a new repository to the configuration."""
        # Check if repository already exists
        for repo in self.repositories:
            if repo.name == name:
                raise ValueError(f"Repository '{name}' already exists")
        
        # Create new repository config
        repo_config = RepositoryConfig(
            name=name,
            path=path,
            description=description,
            active=make_active
        )
        
        self.repositories.append(repo_config)
        
        if make_active or not self.repositories:
            self.active_repository = name
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config_dict = {}
        
        # Override from environment
        if data_dir := os.getenv("CODE_INTEL_DATA_DIR"):
            config_dict["data_directory"] = Path(data_dir)
        
        # Legacy support: if project root is specified, add it as a repository
        if project_root := os.getenv("CODE_INTEL_PROJECT_ROOT"):
            repo_path = Path(project_root)
            repo_name = repo_path.name or "default"
            config_dict["repositories"] = [
                RepositoryConfig(
                    name=repo_name,
                    path=repo_path,
                    description=f"Repository at {repo_path}",
                    active=True
                )
            ]
            config_dict["active_repository"] = repo_name
            
        return cls(**config_dict)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
