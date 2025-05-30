"""Simple integration test to isolate issues."""

import asyncio
import logging
from pathlib import Path
import tempfile

from code_intelligence_mcp.config import Config, GraphConfig, ParserConfig, VectorConfig
from code_intelligence_mcp.di import create_container
from code_intelligence_mcp.graph import GraphService
from code_intelligence_mcp.parser import ParserService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Python test content
SIMPLE_PYTHON = '''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")

class TestClass:
    """A simple test class."""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        """Get the stored value."""
        return self.value

if __name__ == "__main__":
    hello_world()
    test = TestClass()
    print(test.get_value())
'''


async def test_simple_integration():
    """Test basic functionality without vector service."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"=== Simple Integration Test ===")
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
        
        # Create DI container with just graph and parser
        container = await create_container(config)
        
        # Get services
        graph_service = await container.get(GraphService)
        parser_service = await container.get(ParserService)
        
        logger.info("Services initialized successfully")
        
        # Create a test file
        test_file = Path(tmpdir) / "simple_test.py"
        test_file.write_text(SIMPLE_PYTHON)
        logger.info(f"Created test file: {test_file}")
        
        # Parse the file
        logger.info("\n=== Parsing file ===")
        symbols, relationships = await parser_service.analyze_file(test_file)
        logger.info(f"Parsed {len(symbols)} symbols and {len(relationships)} relationships")
        
        for symbol in symbols:
            logger.info(f"  Symbol: {symbol.name} ({symbol.symbol_type}) at line {symbol.line_start}")
        
        # Store in graph
        logger.info("\n=== Storing in graph ===")
        symbol_ids = []
        for symbol in symbols:
            symbol_id = await graph_service.add_symbol(
                name=symbol.name,
                qualified_name=symbol.qualified_name,
                symbol_type=symbol.symbol_type,
                file_path=str(test_file),
                line_start=symbol.line_start,
                line_end=symbol.line_end,
                language=symbol.language or "unknown",
                ast_hash=symbol.ast_hash,
                signature=symbol.signature,
                docstring=symbol.docstring
            )
            symbol_ids.append(symbol_id)
            logger.info(f"  Stored symbol {symbol.name} with ID {symbol_id}")
        
        # Store relationships
        for rel in relationships:
            # Find symbol IDs for the relationship
            from_symbols = await graph_service.find_symbol(qualified_name=rel.from_symbol)
            to_symbols = await graph_service.find_symbol(qualified_name=rel.to_symbol)
            
            if from_symbols and to_symbols:
                rel_id = await graph_service.add_relationship(
                    from_symbol_id=from_symbols[0]['id'],
                    to_symbol_id=to_symbols[0]['id'],
                    relationship_type=rel.relationship_type,
                    file_path=str(test_file),
                    line_number=rel.line_number
                )
                logger.info(f"  Stored relationship: {rel.from_symbol} -> {rel.to_symbol}")
        
        # Get statistics
        logger.info("\n=== Graph Statistics ===")
        stats = await graph_service.get_statistics()
        logger.info(f"Total symbols: {stats['total_symbols']}")
        logger.info(f"Total relationships: {stats['total_relationships']}")
        logger.info(f"Symbols by type: {stats['symbols_by_type']}")
        
        # Test search
        logger.info("\n=== Testing Search ===")
        hello_symbols = await graph_service.find_symbol(name="hello_world")
        logger.info(f"Found {len(hello_symbols)} symbols named 'hello_world'")
        
        class_symbols = await graph_service.find_symbol(name="TestClass")
        logger.info(f"Found {len(class_symbols)} symbols named 'TestClass'")
        
        # Shutdown
        await container.shutdown()
        logger.info("\nSimple integration test completed successfully!")


async def main():
    """Run the test."""
    try:
        await test_simple_integration()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())