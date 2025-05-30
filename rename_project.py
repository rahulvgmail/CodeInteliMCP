#!/usr/bin/env python3
"""
Script to update file references after renaming project directory.
Run this AFTER renaming the directory to CodeInteliMCP.
"""

import os
from pathlib import Path

def update_file_references():
    """Update all file references from treeSitterMCP to CodeInteliMCP."""
    
    # Files that need updating (relative to project root)
    files_to_update = [
        "example_claude_config.json",
        "USER_GUIDE.md", 
        "README.md",
        "ARCHITECTURE.md"
    ]
    
    old_name = "treeSitterMCP"
    new_name = "CodeInteliMCP"
    
    project_root = Path(__file__).parent
    
    for file_path in files_to_update:
        full_path = project_root / file_path
        
        if not full_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
            
        print(f"📝 Updating {file_path}...")
        
        # Read file content
        content = full_path.read_text()
        
        # Replace all occurrences
        updated_content = content.replace(old_name, new_name)
        
        # Write back if changed
        if content != updated_content:
            full_path.write_text(updated_content)
            print(f"✅ Updated {file_path}")
        else:
            print(f"ℹ️  No changes needed in {file_path}")
    
    print(f"\n🎉 Project references updated from '{old_name}' to '{new_name}'!")
    print("\n📋 Next steps:")
    print("1. Restart Claude Desktop/Code to pick up new paths")
    print("2. Update any existing Claude configurations with new paths")
    print("3. Re-run setup_for_claude.py if needed")

if __name__ == "__main__":
    update_file_references()