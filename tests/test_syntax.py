"""Syntax validation test.

ast.parse every file under scripts/.
"""
import ast
import os
import pytest


def get_all_python_files(directory, exclude_dirs=None):
    """Recursively get all Python files in directory."""
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', '.pytest_cache', '.git'}
    
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files


def test_scripts_syntax():
    """Verify all Python files in scripts/ have valid syntax."""
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
    
    python_files = get_all_python_files(scripts_dir)
    
    assert len(python_files) > 0, "No Python files found in scripts/"
    
    errors = []
    for filepath in python_files:
        rel_path = os.path.relpath(filepath, os.path.dirname(scripts_dir))
        try:
            with open(filepath, 'r') as f:
                source = f.read()
            ast.parse(source)
        except SyntaxError as e:
            errors.append(f"{rel_path}: {e}")
    
    assert len(errors) == 0, f"Syntax errors found:\n" + "\n".join(errors)


def test_lerna_syntax():
    """Verify all Python files in lerna/ have valid syntax."""
    lerna_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lerna')
    
    python_files = get_all_python_files(lerna_dir)
    
    assert len(python_files) > 0, "No Python files found in lerna/"
    
    errors = []
    for filepath in python_files:
        rel_path = os.path.relpath(filepath, os.path.dirname(lerna_dir))
        try:
            with open(filepath, 'r') as f:
                source = f.read()
            ast.parse(source)
        except SyntaxError as e:
            errors.append(f"{rel_path}: {e}")
    
    assert len(errors) == 0, f"Syntax errors found:\n" + "\n".join(errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])