#!/usr/bin/env python3
"""
Quick setup script for Code Intelligence MCP with Claude Desktop/Code.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def get_claude_config_path():
    """Get the Claude Desktop configuration file path."""
    if platform.system() == "Darwin":  # macOS
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    elif platform.system() == "Windows":
        return Path(os.environ["APPDATA"]) / "Claude/claude_desktop_config.json"
    else:  # Linux
        return Path.home() / ".config/claude/claude_desktop_config.json"


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "mcp>=1.0.0",
        "duckdb>=1.0.0", 
        "chromadb>=1.0.0",
        "sentence-transformers>=2.2.0",
        "aiofiles>=23.0.0",
        "tree-sitter>=0.20.0",
        "tree-sitter-python>=0.20.0",
        "tree-sitter-javascript>=0.20.0",
        "tree-sitter-typescript>=0.20.0"
    ]
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + dependencies)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_claude_desktop_config(project_root: str, data_dir: str):
    """Setup Claude Desktop configuration."""
    config_path = get_claude_config_path()
    server_script = Path(__file__).parent / "code_intelligence_mcp" / "server_minimal.py"
    
    # Create config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config or create new one
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {"mcpServers": {}}
    
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    # Add our server configuration
    config["mcpServers"]["code-intelligence"] = {
        "command": sys.executable,
        "args": [str(server_script)],
        "env": {
            "CODE_INTEL_PROJECT_ROOT": project_root,
            "CODE_INTEL_DATA_DIR": data_dir
        }
    }
    
    # Write config back
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Claude Desktop config updated: {config_path}")
    return True


def setup_claude_code_config(project_root: str):
    """Setup Claude Code configuration in project directory."""
    project_path = Path(project_root)
    clauderc_path = project_path / ".clauderc"
    server_script = Path(__file__).parent / "code_intelligence_mcp" / "server_minimal.py"
    
    config = {
        "mcp": {
            "servers": {
                "code-intelligence": {
                    "command": sys.executable,
                    "args": [str(server_script)],
                    "env": {
                        "CODE_INTEL_PROJECT_ROOT": ".",
                        "CODE_INTEL_DATA_DIR": "./.code-intelligence"
                    }
                }
            }
        }
    }
    
    with open(clauderc_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Claude Code config created: {clauderc_path}")
    return True


def create_data_directory(data_dir: str):
    """Create data directory for storing indices."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(f"✅ Data directory created: {data_dir}")


def main():
    """Main setup function."""
    print("🧠 Code Intelligence MCP Setup")
    print("=" * 40)
    
    # Get project root
    default_project = str(Path.cwd())
    project_root = input(f"📁 Project root directory [{default_project}]: ").strip()
    if not project_root:
        project_root = default_project
    
    # Get data directory
    default_data = str(Path(project_root) / ".code-intelligence")
    data_dir = input(f"💾 Data storage directory [{default_data}]: ").strip()
    if not data_dir:
        data_dir = default_data
    
    # Choose setup type
    print("\n🔧 Setup Options:")
    print("1. Claude Desktop (global)")
    print("2. Claude Code (project-specific)")
    print("3. Both")
    
    choice = input("Choose setup type [1-3]: ").strip()
    
    print(f"\n🚀 Setting up for project: {project_root}")
    print(f"📊 Data directory: {data_dir}")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return 1
    
    # Create data directory
    create_data_directory(data_dir)
    
    # Setup configurations
    success = True
    
    if choice in ["1", "3"]:
        print("\n⚙️  Setting up Claude Desktop configuration...")
        success &= setup_claude_desktop_config(project_root, data_dir)
    
    if choice in ["2", "3"]:
        print("\n⚙️  Setting up Claude Code configuration...")
        success &= setup_claude_code_config(project_root)
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Restart Claude Desktop/Code")
        print("2. Use 'test_connection' to verify setup")
        print("3. Use 'index_file: /path/to/file.py' to start indexing")
        print("4. Try 'semantic_search: \"your search query\"'")
        print("\n📖 See USER_GUIDE.md for detailed usage instructions")
    else:
        print("❌ Setup completed with errors")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())