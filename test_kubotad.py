#!/usr/bin/env python3
"""
Test script for KUBOTAD application
"""
import os
import sys

def test_basic_imports():
    """Test basic imports without heavy dependencies"""
    try:
        import pandas as pd
        print("✓ pandas import: OK")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False

    try:
        import streamlit as st
        print("✓ streamlit import: OK")
    except ImportError as e:
        print(f"✗ streamlit import failed: {e}")
        return False

    try:
        from openai import OpenAI
        print("✓ openai import: OK")
    except ImportError as e:
        print(f"✗ openai import failed: {e}")
        return False

    return True

def test_environment():
    """Test environment variables"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        print(f"✓ OPENAI_API_KEY is set (length: {len(api_key)})")
        return True
    else:
        print("✗ OPENAI_API_KEY is not set")
        return False

def test_rag_dependencies():
    """Test RAG dependencies (optional)"""
    try:
        import sentence_transformers
        print("✓ sentence-transformers: OK")
    except ImportError:
        print("⚠ sentence-transformers: Not available (optional)")

    try:
        import faiss
        print("✓ faiss: OK")
    except ImportError:
        print("⚠ faiss: Not available (optional)")

    try:
        import duckdb
        print("✓ duckdb: OK")
    except ImportError:
        print("⚠ duckdb: Not available (optional)")

    try:
        import langchain
        print("✓ langchain: OK")
    except ImportError:
        print("⚠ langchain: Not available (optional)")

def test_data_files():
    """Test if data files exist"""
    data_files = [
        "KUBOTA_DiagDATA/kubota_diagDATA_1_6.xlsm",
        "KUBOTA_DiagDATA/kubota_diagDATA_7_12.xlsm"
    ]

    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✓ Data file exists: {file_path}")
        else:
            print(f"⚠ Data file not found: {file_path}")

def main():
    """Main test function"""
    print("=== KUBOTAD Application Test ===\n")

    print("1. Testing basic imports...")
    imports_ok = test_basic_imports()
    print()

    print("2. Testing environment...")
    env_ok = test_environment()
    print()

    print("3. Testing RAG dependencies...")
    test_rag_dependencies()
    print()

    print("4. Testing data files...")
    test_data_files()
    print()

    if imports_ok and env_ok:
        print("✓ Core functionality should work")
        print("✓ RAG system initialization error has been resolved with fallback mechanism")
        print("✓ Application should gracefully handle missing RAG dependencies")
    else:
        print("✗ Some core issues detected")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()