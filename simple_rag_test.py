# simple_rag_test.py
# Simplified test for RAG system functionality

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_test_data():
    """Create simple test maintenance data"""
    data = {
        '整備案件NO': ['M202400001', 'M202400002', 'M202400003'],
        '機種名': ['U-30', 'KX-080', 'U-15'],
        '作業内容': ['オイル漏れ修理', 'ポンプ交換', 'エンジンオーバーホール'],
        '不具合内容': ['油圧系統からのオイル漏れ', 'ポンプの異音と振動', 'エンジン始動不良'],
        '部品品名': ['オイルシール', '油圧ポンプ', 'スターターモーター']
    }
    return pd.DataFrame(data)

def test_basic_functionality():
    """Test basic system functionality without heavy dependencies"""
    print("🧪 RAG-Enhanced Search System - Basic Test")
    print("=" * 50)

    try:
        # Test data creation
        test_data = create_test_data()
        print(f"✅ Test data created: {len(test_data)} records")

        # Test DuckDB functionality
        try:
            import duckdb
            conn = duckdb.connect(':memory:')

            # Test basic query
            conn.register('test_df', test_data)
            result = conn.execute("SELECT COUNT(*) FROM test_df").fetchone()
            print(f"✅ DuckDB basic test: {result[0]} records")

            # Test text search
            search_result = conn.execute("""
                SELECT * FROM test_df
                WHERE 作業内容 LIKE '%オイル%' OR 不具合内容 LIKE '%オイル%'
            """).fetchall()
            print(f"✅ DuckDB text search: Found {len(search_result)} oil-related records")

            conn.close()

        except ImportError:
            print("❌ DuckDB not available")
            return False

        # Test pandas operations
        filtered_data = test_data[test_data['作業内容'].str.contains('オイル', na=False)]
        print(f"✅ Pandas text filtering: {len(filtered_data)} records found")

        # Test search functionality
        query_terms = ['オイル', 'ポンプ', 'エンジン']
        for term in query_terms:
            matches = test_data[test_data.apply(
                lambda row: any(term in str(cell) for cell in row), axis=1
            )]
            print(f"✅ Search for '{term}': {len(matches)} matches")

        print("\n🎉 Basic functionality test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_integration_readiness():
    """Test if the system is ready for RAG integration"""
    print("\n🔧 Integration Readiness Check")
    print("-" * 30)

    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key available")
    else:
        print("⚠️  OpenAI API key not set")

    # Check file structure
    required_files = ['rag_enhanced_search.py', 'KUBOTAD.py']
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")

    # Check dependencies
    available_deps = []
    optional_deps = [
        ('duckdb', 'DuckDB database'),
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical operations'),
        ('streamlit', 'Web interface')
    ]

    for dep_name, description in optional_deps:
        try:
            __import__(dep_name)
            print(f"✅ {dep_name} - {description}")
            available_deps.append(dep_name)
        except ImportError:
            print(f"❌ {dep_name} - {description} (not installed)")

    print(f"\n📊 Available dependencies: {len(available_deps)}/{len(optional_deps)}")

    return len(available_deps) >= 3  # Need at least core dependencies

if __name__ == "__main__":
    # Run basic tests
    basic_success = test_basic_functionality()

    # Check integration readiness
    integration_ready = test_integration_readiness()

    print(f"\n📋 Test Summary")
    print(f"   Basic functionality: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   Integration readiness: {'✅ READY' if integration_ready else '⚠️  PARTIAL'}")

    if basic_success and integration_ready:
        print(f"\n🚀 System is ready for RAG enhancement!")
        print(f"   Next steps:")
        print(f"   1. Install remaining dependencies: uv add sentence-transformers faiss-cpu")
        print(f"   2. Set OPENAI_API_KEY environment variable")
        print(f"   3. Run: streamlit run KUBOTAD.py")
    else:
        print(f"\n⚠️  Please resolve the issues above before proceeding.")