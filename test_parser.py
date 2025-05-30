"""Test the parser service."""

import asyncio
import logging
from pathlib import Path

from code_intelligence_mcp.config import Config, ParserConfig
from code_intelligence_mcp.parser import ParserService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test Python code
PYTHON_CODE = '''
class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    
    async def divide(self, a: float, b: float) -> float:
        """Divide two numbers asynchronously."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

def main():
    """Main function."""
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"Result: {result}")
    
    # Test multiplication
    product = calc.multiply(4, 7)
    print(f"Product: {product}")

if __name__ == "__main__":
    main()
'''

# Test JavaScript code
JAVASCRIPT_CODE = '''
// Calculator class
class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(a, b) {
        return a + b;
    }
    
    multiply(a, b) {
        return a * b;
    }
    
    async divide(a, b) {
        if (b === 0) {
            throw new Error("Cannot divide by zero");
        }
        return a / b;
    }
}

// Arrow function
const createCalculator = () => new Calculator();

// Function expression
const calculate = function(operation, a, b) {
    const calc = createCalculator();
    switch(operation) {
        case 'add':
            return calc.add(a, b);
        case 'multiply':
            return calc.multiply(a, b);
        default:
            throw new Error(`Unknown operation: ${operation}`);
    }
};

// Async function
async function performDivision(a, b) {
    const calc = new Calculator();
    return await calc.divide(a, b);
}

export { Calculator, createCalculator, calculate };
'''


async def test_parser():
    """Test the parser service."""
    # Create parser config
    config = ParserConfig()
    
    # Create parser service
    parser = ParserService(config)
    await parser.initialize()
    
    logger.info("=== Testing Python parsing ===")
    
    # Parse Python code
    symbols, relationships = await parser.parse_content(
        PYTHON_CODE,
        'python',
        'test_calculator.py'
    )
    
    logger.info(f"Found {len(symbols)} symbols:")
    for symbol in symbols:
        logger.info(f"  - {symbol.symbol_type} {symbol.qualified_name} at line {symbol.line_start}")
        if symbol.docstring:
            logger.info(f"    Docstring: {symbol.docstring}")
        if symbol.signature:
            logger.info(f"    Signature: {symbol.signature}")
    
    logger.info(f"\nFound {len(relationships)} relationships:")
    for rel in relationships:
        logger.info(f"  - {rel.from_symbol} {rel.relationship_type} {rel.to_symbol} at line {rel.line_number}")
    
    logger.info("\n=== Testing JavaScript parsing ===")
    
    # Parse JavaScript code
    symbols, relationships = await parser.parse_content(
        JAVASCRIPT_CODE,
        'javascript',
        'test_calculator.js'
    )
    
    logger.info(f"Found {len(symbols)} symbols:")
    for symbol in symbols:
        logger.info(f"  - {symbol.symbol_type} {symbol.qualified_name} at line {symbol.line_start}")
    
    logger.info(f"\nFound {len(relationships)} relationships:")
    for rel in relationships:
        logger.info(f"  - {rel.from_symbol} {rel.relationship_type} {rel.to_symbol} at line {rel.line_number}")
    
    # Test language detection
    logger.info("\n=== Testing language detection ===")
    test_files = [
        "test.py",
        "test.js",
        "test.ts",
        "test.go",
        "test.rs",
        "test.unknown"
    ]
    
    for file_name in test_files:
        lang = parser.get_language_from_file(file_name)
        logger.info(f"  {file_name} -> {lang or 'Unknown'}")
    
    # Test supported languages
    logger.info("\n=== Supported languages ===")
    supported = parser.get_supported_languages()
    logger.info(f"  {', '.join(supported)}")
    
    await parser.shutdown()
    logger.info("\nParser test completed!")


async def test_file_parsing():
    """Test parsing actual files."""
    config = ParserConfig()
    parser = ParserService(config)
    await parser.initialize()
    
    # Try to parse this test file itself
    logger.info("\n=== Parsing this test file ===")
    
    test_file = Path(__file__)
    symbols, relationships = await parser.analyze_file(test_file)
    
    logger.info(f"Found {len(symbols)} symbols in {test_file.name}")
    for symbol in symbols[:5]:  # Show first 5
        logger.info(f"  - {symbol.symbol_type} {symbol.name} at line {symbol.line_start}")
    
    if len(symbols) > 5:
        logger.info(f"  ... and {len(symbols) - 5} more")
    
    await parser.shutdown()


async def main():
    """Run all tests."""
    try:
        await test_parser()
        await test_file_parsing()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())