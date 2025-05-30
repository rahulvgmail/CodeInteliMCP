#!/usr/bin/env python3
"""
Test script to verify the Code Intelligence MCP installation.
"""

import asyncio
import tempfile
import logging
from pathlib import Path

# Test imports
try:
    from code_intelligence_mcp.config import Config, GraphConfig, ParserConfig, VectorConfig
    from code_intelligence_mcp.server_minimal import MinimalCodeIntelligenceMCP
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test dependencies
required_modules = [
    'mcp', 'duckdb', 'chromadb', 'sentence_transformers', 
    'aiofiles', 'tree_sitter'
]

missing_modules = []
for module in required_modules:
    try:
        __import__(module)
        print(f"✅ {module} available")
    except ImportError:
        print(f"❌ {module} missing")
        missing_modules.append(module)

if missing_modules:
    print(f"\n❌ Missing dependencies: {missing_modules}")
    print("Run: pip install " + " ".join(missing_modules))
    exit(1)

# Test basic functionality
async def test_basic_functionality():
    """Test basic server functionality."""
    print("\n🧪 Testing basic functionality...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
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
        
        # Create and initialize server
        server = MinimalCodeIntelligenceMCP(config)
        
        try:
            print("🔄 Initializing server...")
            await server.initialize()
            print("✅ Server initialized successfully")
            
            # Test connection
            print("🔄 Testing connection...")
            result = await server.server.call_tool("test_connection", {})
            print(f"✅ Connection test: {result}")
            
            # Test stats
            print("🔄 Testing stats...")
            stats = await server.server.call_tool("get_index_stats", {})
            print(f"✅ Stats retrieved: {stats}")
            
            # Create a simple test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("""
def hello_world():
    print("Hello, World!")

class TestClass:
    def __init__(self):
        self.value = 42
""")
            
            print("🔄 Testing file indexing...")
            index_result = await server.server.call_tool("index_file", {"file_path": str(test_file)})
            print(f"✅ Indexing successful: {index_result}")
            
            print("🔄 Testing symbol search...")
            search_result = await server.server.call_tool("find_symbol", {"name": "TestClass"})
            print(f"✅ Symbol search successful: {search_result}")
            
            await server.shutdown()
            print("✅ All tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            await server.shutdown()
            return False
    
    return True

# Main test
async def main():
    print("🧠 Code Intelligence MCP Installation Test")
    print("=" * 50)
    
    success = await test_basic_functionality()
    
    if success:
        print("\n🎉 Installation test completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run 'python setup_for_claude.py' to configure Claude")
        print("2. Restart Claude Desktop/Code")
        print("3. Use 'test_connection' in Claude to verify")
        print("4. See USER_GUIDE.md for detailed usage")
    else:
        print("\n❌ Installation test failed!")
        print("Check the error messages above for troubleshooting")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))