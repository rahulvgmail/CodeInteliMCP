"""Test script for minimal server."""

import asyncio
from pathlib import Path
import tempfile
import logging

from code_intelligence_mcp.config import Config, GraphConfig
from code_intelligence_mcp.server_minimal import MinimalCodeIntelligenceMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_server():
    """Test the minimal server."""
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Created temporary directory: {tmpdir}")
        
        # Create test configuration
        config = Config(
            project_root=Path(__file__).parent,
            data_directory=Path(tmpdir) / "data",
            graph=GraphConfig(
                db_path=Path(tmpdir) / "data" / "test.duckdb"
            )
        )
        
        # Create server
        server = MinimalCodeIntelligenceMCP(config)
        
        # Initialize
        logger.info("Initializing server...")
        await server.initialize()
        
        # Test tools by calling them directly through the server
        logger.info("\n=== Testing connection ===")
        try:
            result = await server.server.call_tool("test_connection", {})
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error calling test_connection: {e}")
        
        logger.info("\n=== Getting index stats ===")
        try:
            result = await server.server.call_tool("get_index_stats", {})
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error calling get_index_stats: {e}")
        
        logger.info("\n=== Adding test symbol ===")
        try:
            result = await server.server.call_tool("add_test_symbol", {})
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error calling add_test_symbol: {e}")
        
        logger.info("\n=== Searching for symbol ===")
        try:
            result = await server.server.call_tool("find_symbol", {"name": "test_function"})
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error calling find_symbol: {e}")
        
        logger.info("\n=== Finding usages ===")
        try:
            result = await server.server.call_tool("find_usages", {"name": "test_function"})
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error calling find_usages: {e}")
        
        # Test direct service access
        logger.info("\n=== Direct service test ===")
        stats = await server.graph.get_statistics()
        logger.info(f"Database statistics: {stats}")
        
        # Shutdown
        await server.shutdown()
        logger.info("\nTest completed successfully!")


async def test_integration():
    """Test with more realistic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"\n=== Integration Test ===")
        logger.info(f"Temporary directory: {tmpdir}")
        
        # Create test configuration
        config = Config(
            project_root=Path(__file__).parent,
            data_directory=Path(tmpdir) / "data",
            graph=GraphConfig(
                db_path=Path(tmpdir) / "data" / "integration.duckdb"
            )
        )
        
        # Create server
        server = MinimalCodeIntelligenceMCP(config)
        await server.initialize()
        
        # Add some test data
        logger.info("\nAdding test symbols...")
        
        # Add a module
        module_id = await server.graph.add_symbol(
            name="test_module",
            qualified_name="test_module",
            symbol_type="module",
            file_path="/test/test_module.py",
            line_start=1,
            line_end=100,
            language="python",
            ast_hash="module_hash"
        )
        
        # Add a class
        class_id = await server.graph.add_symbol(
            name="TestClass",
            qualified_name="test_module.TestClass",
            symbol_type="class",
            file_path="/test/test_module.py",
            line_start=10,
            line_end=50,
            language="python",
            ast_hash="class_hash",
            parent_symbol_id=module_id
        )
        
        # Add a method
        method_id = await server.graph.add_symbol(
            name="test_method",
            qualified_name="test_module.TestClass.test_method",
            symbol_type="method",
            file_path="/test/test_module.py",
            line_start=20,
            line_end=30,
            language="python",
            ast_hash="method_hash",
            parent_symbol_id=class_id,
            signature="def test_method(self, arg1: str) -> None"
        )
        
        # Add another function that calls the method
        caller_id = await server.graph.add_symbol(
            name="caller_function",
            qualified_name="test_module.caller_function",
            symbol_type="function",
            file_path="/test/test_module.py",
            line_start=60,
            line_end=70,
            language="python",
            ast_hash="caller_hash"
        )
        
        # Add relationship
        rel_id = await server.graph.add_relationship(
            from_symbol_id=caller_id,
            to_symbol_id=method_id,
            relationship_type="calls",
            file_path="/test/test_module.py",
            line_number=65
        )
        
        logger.info(f"Created symbols: module={module_id}, class={class_id}, method={method_id}, caller={caller_id}")
        logger.info(f"Created relationship: {rel_id}")
        
        # Test queries
        logger.info("\n=== Testing queries ===")
        
        # Find symbols
        symbols = await server.graph.find_symbol(name="TestClass")
        logger.info(f"Found symbols named 'TestClass': {len(symbols)}")
        
        # Find usages
        if symbols:
            usages = await server.graph.find_usages(method_id)
            logger.info(f"Found usages of test_method: {usages}")
        
        # Get statistics
        stats = await server.graph.get_statistics()
        logger.info(f"Final statistics: {stats}")
        
        await server.shutdown()
        logger.info("\nIntegration test completed!")


async def main():
    """Run all tests."""
    try:
        await test_server()
        await test_integration()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())