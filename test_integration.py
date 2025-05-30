"""Integration test for the code intelligence system."""

import asyncio
import logging
from pathlib import Path
import tempfile

from code_intelligence_mcp.config import Config, GraphConfig, ParserConfig
from code_intelligence_mcp.server_minimal import MinimalCodeIntelligenceMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample Python file content
SAMPLE_PYTHON = '''
"""Sample module for testing."""

class BaseClass:
    """Base class for testing."""
    
    def base_method(self):
        """Base method."""
        pass

class DerivedClass(BaseClass):
    """Derived class for testing."""
    
    def __init__(self):
        super().__init__()
        self.value = 42
    
    def derived_method(self):
        """Call the base method."""
        self.base_method()
        return self.value
    
    def process_data(self, data):
        """Process some data."""
        result = []
        for item in data:
            processed = self.transform(item)
            result.append(processed)
        return result
    
    def transform(self, item):
        """Transform a single item."""
        return item * 2

def main():
    """Main function."""
    obj = DerivedClass()
    result = obj.process_data([1, 2, 3])
    print(f"Result: {result}")
    
    # Test derived method
    value = obj.derived_method()
    print(f"Value: {value}")

if __name__ == "__main__":
    main()
'''


async def test_full_integration():
    """Test the full integration of parsing and indexing."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"=== Full Integration Test ===")
        logger.info(f"Temporary directory: {tmpdir}")
        
        # Create test configuration
        config = Config(
            project_root=Path(tmpdir),
            data_directory=Path(tmpdir) / "data",
            graph=GraphConfig(
                db_path=Path(tmpdir) / "data" / "test.duckdb"
            ),
            parser=ParserConfig()
        )
        
        # Create server
        server = MinimalCodeIntelligenceMCP(config)
        await server.initialize()
        
        # Create a test file
        test_file = Path(tmpdir) / "test_module.py"
        test_file.write_text(SAMPLE_PYTHON)
        logger.info(f"Created test file: {test_file}")
        
        # Index the file
        logger.info("\n=== Indexing file ===")
        result = await server.server.call_tool("index_file", {"file_path": str(test_file)})
        logger.info(f"Index result: {result}")
        
        # Get statistics
        logger.info("\n=== Database statistics ===")
        stats = await server.server.call_tool("get_index_stats", {})
        logger.info(f"Stats: {stats}")
        
        # Test finding symbols
        logger.info("\n=== Finding symbols ===")
        
        # Find BaseClass
        result = await server.server.call_tool("find_symbol", {"name": "BaseClass"})
        logger.info(f"BaseClass: {result}")
        
        # Find DerivedClass
        result = await server.server.call_tool("find_symbol", {"name": "DerivedClass"})
        logger.info(f"DerivedClass: {result}")
        
        # Find process_data
        result = await server.server.call_tool("find_symbol", {"name": "process_data"})
        logger.info(f"process_data: {result}")
        
        # Test finding usages
        logger.info("\n=== Finding usages ===")
        
        # Find usages of base_method
        result = await server.server.call_tool("find_usages", {"name": "base_method"})
        logger.info(f"Usages of base_method: {result}")
        
        # Find usages of transform
        result = await server.server.call_tool("find_usages", {"name": "transform"})
        logger.info(f"Usages of transform: {result}")
        
        # Find usages of DerivedClass
        result = await server.server.call_tool("find_usages", {"name": "DerivedClass"})
        logger.info(f"Usages of DerivedClass: {result}")
        
        # Test direct graph queries
        logger.info("\n=== Direct graph queries ===")
        
        # Get all symbols
        all_symbols = await server.graph.execute_query("SELECT * FROM symbols ORDER BY line_start")
        logger.info(f"Total symbols in database: {len(all_symbols)}")
        for symbol in all_symbols:
            logger.info(f"  - {symbol[2]} {symbol[1]} at line {symbol[5]}")  # type, name, line
        
        # Get all relationships
        all_rels = await server.graph.execute_query("SELECT * FROM relationships")
        logger.info(f"\nTotal relationships: {len(all_rels)}")
        
        # Get inheritance relationships
        inheritance = await server.graph.execute_query(
            "SELECT s1.name, s2.name FROM relationships r "
            "JOIN symbols s1 ON r.from_symbol_id = s1.id "
            "JOIN symbols s2 ON r.to_symbol_id = s2.id "
            "WHERE r.relationship_type = 'inherits'"
        )
        logger.info(f"\nInheritance relationships:")
        for rel in inheritance:
            logger.info(f"  - {rel[0]} inherits from {rel[1]}")
        
        # Get call relationships
        calls = await server.graph.execute_query(
            "SELECT s1.qualified_name, s2.name, r.line_number FROM relationships r "
            "JOIN symbols s1 ON r.from_symbol_id = s1.id "
            "JOIN symbols s2 ON r.to_symbol_id = s2.id "
            "WHERE r.relationship_type = 'calls' "
            "ORDER BY r.line_number"
        )
        logger.info(f"\nCall relationships:")
        for rel in calls:
            logger.info(f"  - {rel[0]} calls {rel[1]} at line {rel[2]}")
        
        # Shutdown
        await server.shutdown()
        logger.info("\nIntegration test completed!")


async def main():
    """Run all tests."""
    try:
        await test_full_integration()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())