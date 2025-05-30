"""Integration test for vector service with semantic search."""

import asyncio
import logging
from pathlib import Path
import tempfile

from code_intelligence_mcp.config import Config, GraphConfig, ParserConfig, VectorConfig
from code_intelligence_mcp.server_minimal import MinimalCodeIntelligenceMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample Python files with different patterns
AUTHENTICATION_CODE = '''
"""Authentication module."""

import hashlib
import secrets
from typing import Optional, Dict, Any

class AuthService:
    """Service for user authentication."""
    
    def __init__(self):
        self.active_sessions = {}
    
    async def login(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        if self.verify_password(username, password):
            token = self.generate_session_token()
            self.active_sessions[token] = username
            return token
        return None
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password against stored hash."""
        # Simulate password verification
        stored_hash = self.get_password_hash(username)
        return self.hash_password(password) == stored_hash
    
    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm."""
        salt = secrets.token_hex(16)
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    
    def generate_session_token(self) -> str:
        """Generate secure session token."""
        return secrets.token_urlsafe(32)
    
    def get_password_hash(self, username: str) -> str:
        """Get stored password hash for user."""
        # Simulate database lookup
        return "dummy_hash"
'''

DATABASE_CODE = '''
"""Database operations module."""

import sqlite3
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

class DatabaseService:
    """Service for database operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    async def create_user(self, username: str, email: str) -> int:
        """Create new user in database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                (username, email)
            )
            conn.commit()
            return cursor.lastrowid
    
    async def find_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Find user by username."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            if row:
                return {"id": row[0], "username": row[1], "email": row[2]}
        return None
    
    async def update_user_profile(self, user_id: int, profile_data: Dict[str, Any]) -> bool:
        """Update user profile information."""
        fields = []
        values = []
        for key, value in profile_data.items():
            fields.append(f"{key} = ?")
            values.append(value)
        
        values.append(user_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
'''

API_CODE = '''
"""API endpoint handlers."""

from typing import Dict, Any
import json
from http.server import BaseHTTPRequestHandler

class ApiHandler:
    """Handle HTTP API requests."""
    
    def __init__(self, auth_service, db_service):
        self.auth = auth_service
        self.db = db_service
    
    async def handle_login(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user login request."""
        username = request_data.get('username')
        password = request_data.get('password')
        
        if not username or not password:
            return {"error": "Username and password required", "status": 400}
        
        token = await self.auth.login(username, password)
        if token:
            return {"token": token, "status": 200}
        else:
            return {"error": "Invalid credentials", "status": 401}
    
    async def handle_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get user profile information."""
        user = await self.db.find_user_by_id(user_id)
        if user:
            return {"user": user, "status": 200}
        else:
            return {"error": "User not found", "status": 404}
    
    async def handle_registration(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user registration."""
        username = request_data.get('username')
        email = request_data.get('email')
        password = request_data.get('password')
        
        if not all([username, email, password]):
            return {"error": "All fields required", "status": 400}
        
        # Check if user exists
        existing = await self.db.find_user(username)
        if existing:
            return {"error": "Username already exists", "status": 409}
        
        # Create user
        user_id = await self.db.create_user(username, email)
        return {"user_id": user_id, "status": 201}
    
    def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate incoming request format."""
        required_headers = ['Content-Type', 'Authorization']
        return all(header in request.get('headers', {}) for header in required_headers)
'''


async def test_vector_integration():
    """Test the vector service integration with semantic search."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"=== Vector Integration Test ===")
        logger.info(f"Temporary directory: {tmpdir}")
        
        # Create test configuration
        config = Config(
            project_root=Path(tmpdir),
            data_directory=Path(tmpdir) / "data",
            graph=GraphConfig(
                db_path=Path(tmpdir) / "data" / "test.duckdb"
            ),
            vector=VectorConfig(
                chroma_path=Path(tmpdir) / "data" / "chroma"
            ),
            parser=ParserConfig()
        )
        
        # Create server
        server = MinimalCodeIntelligenceMCP(config)
        await server.initialize()
        
        # Create test files
        auth_file = Path(tmpdir) / "auth.py"
        auth_file.write_text(AUTHENTICATION_CODE)
        
        db_file = Path(tmpdir) / "database.py"
        db_file.write_text(DATABASE_CODE)
        
        api_file = Path(tmpdir) / "api.py"
        api_file.write_text(API_CODE)
        
        logger.info("Created test files with different patterns")
        
        # Index all files
        logger.info("\n=== Indexing files ===")
        
        for file_path in [auth_file, db_file, api_file]:
            result = await server.server.call_tool("index_file", {"file_path": str(file_path)})
            logger.info(f"Index result for {file_path.name}: {result}")
            if isinstance(result, dict) and 'symbols_indexed' in result:
                logger.info(f"Indexed {file_path.name}: {result['symbols_indexed']} symbols")
            else:
                logger.error(f"Unexpected result format: {type(result)} - {result}")
        
        # Get statistics
        logger.info("\n=== Statistics ===")
        graph_stats = await server.server.call_tool("get_index_stats", {})
        logger.info(f"Graph stats: {graph_stats['stats']['total_symbols']} symbols")
        
        vector_stats = await server.server.call_tool("get_vector_stats", {})
        logger.info(f"Vector stats: {vector_stats['vector_stats']['total_vectors']} vectors")
        
        # Test semantic searches
        logger.info("\n=== Semantic Search Tests ===")
        
        # Search for authentication-related code
        auth_search = await server.server.call_tool("semantic_search", {
            "query": "user authentication login password verification",
            "limit": 5
        })
        logger.info(f"Auth search found {auth_search['total_found']} results:")
        for result in auth_search['results'][:3]:
            logger.info(f"  - {result['name']} ({result['symbol_type']}) in {Path(result['file_path']).name}")
        
        # Search for database operations
        db_search = await server.server.call_tool("semantic_search", {
            "query": "database operations SQL query insert update",
            "limit": 5
        })
        logger.info(f"\nDB search found {db_search['total_found']} results:")
        for result in db_search['results'][:3]:
            logger.info(f"  - {result['name']} ({result['symbol_type']}) in {Path(result['file_path']).name}")
        
        # Search for API handlers
        api_search = await server.server.call_tool("semantic_search", {
            "query": "HTTP API endpoint request handler",
            "limit": 5
        })
        logger.info(f"\nAPI search found {api_search['total_found']} results:")
        for result in api_search['results'][:3]:
            logger.info(f"  - {result['name']} ({result['symbol_type']}) in {Path(result['file_path']).name}")
        
        # Test finding similar code
        logger.info("\n=== Similar Code Tests ===")
        
        # Find code similar to login method
        similar_login = await server.server.call_tool("find_similar_code", {
            "symbol_name": "login"
        })
        if similar_login['status'] == 'success':
            logger.info(f"Found {similar_login['total_found']} symbols similar to login:")
            for result in similar_login['similar_symbols'][:3]:
                logger.info(f"  - {result['name']} ({result['symbol_type']}) in {Path(result['file_path']).name}")
        
        # Find code similar to create_user
        similar_create = await server.server.call_tool("find_similar_code", {
            "symbol_name": "create_user"
        })
        if similar_create['status'] == 'success':
            logger.info(f"\nFound {similar_create['total_found']} symbols similar to create_user:")
            for result in similar_create['similar_symbols'][:3]:
                logger.info(f"  - {result['name']} ({result['symbol_type']}) in {Path(result['file_path']).name}")
        
        # Test type-specific searches
        logger.info("\n=== Type-Specific Search Tests ===")
        
        # Search only for classes
        class_search = await server.server.call_tool("semantic_search", {
            "query": "service class handler",
            "symbol_types": "class",
            "limit": 10
        })
        logger.info(f"Class search found {class_search['total_found']} classes:")
        for result in class_search['results']:
            logger.info(f"  - {result['name']} in {Path(result['file_path']).name}")
        
        # Search only for methods
        method_search = await server.server.call_tool("semantic_search", {
            "query": "handle process create",
            "symbol_types": "method",
            "limit": 10
        })
        logger.info(f"\nMethod search found {method_search['total_found']} methods:")
        for result in method_search['results'][:5]:
            logger.info(f"  - {result['name']} in {Path(result['file_path']).name}")
        
        # Shutdown
        await server.shutdown()
        logger.info("\nVector integration test completed!")


async def main():
    """Run all tests."""
    try:
        await test_vector_integration()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())