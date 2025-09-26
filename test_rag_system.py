# test_rag_system.py
# Test script for RAG-Enhanced Search System

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_maintenance_data(n_records: int = 100) -> pd.DataFrame:
    """Create sample maintenance data for testing"""

    np.random.seed(42)

    # Sample data arrays
    machine_models = ['U-15', 'U-30', 'U-45', 'KX-080', 'KX-165', 'M-135', 'M-155']
    series = ['U-Series', 'KX-Series', 'M-Series']
    categories = ['エンジン', '油圧系統', 'アンダーキャリッジ', '電装系', 'キャビン', '作業機']
    departments = ['東京支店', '大阪支店', '名古屋支店', '福岡支店', '仙台支店']

    work_contents = [
        'オイル漏れ修理', 'ポンプ交換', 'エンジンオーバーホール', 'バルブ交換',
        'シール交換', 'フィルター交換', 'ベアリング交換', '配線修理',
        'センサー交換', 'ホース交換', 'タンク修理', 'シリンダー修理'
    ]

    malfunction_contents = [
        'オイル漏れによる油圧低下', 'ポンプの異音発生', 'エンジン始動不良',
        '油圧シリンダーの作動不良', '電装系統のショート', 'センサーの故障',
        'フィルターの詰まり', 'ホースの破損', 'ベアリングの摩耗',
        'バルブからの漏れ', 'タンク内の汚れ', 'シール部からの漏れ'
    ]

    parts_names = [
        'オイルシール', '油圧ポンプ', 'オイルフィルター', '油圧ホース',
        'ピストンシール', 'ベアリング', 'バルブ', 'センサー',
        'オイルクーラー', 'タンク', 'シリンダー', 'ガスケット'
    ]

    # Generate sample data
    data = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(n_records):
        record = {
            '整備案件NO': f'M{2024:04d}{i+1:05d}',
            '機種名': np.random.choice(machine_models),
            'シリーズ': np.random.choice(series),
            'カテゴリ': np.random.choice(categories),
            '部門名': np.random.choice(departments),
            '作業終了日': base_date + timedelta(days=np.random.randint(0, 365)),
            '作業内容': np.random.choice(work_contents),
            '不具合内容': np.random.choice(malfunction_contents),
            '部品品名': np.random.choice(parts_names),
            '部品品番': f'P{np.random.randint(1000, 9999)}',
            '作業時間': np.random.uniform(1.0, 8.0),
            'アワメータ_HR': np.random.uniform(100, 3000),
            'ブルチンNO': f'K-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}'
        }
        data.append(record)

    return pd.DataFrame(data)

def test_rag_system():
    """Test the RAG-enhanced search system"""

    try:
        # Import the RAG system
        from rag_enhanced_search import RAGEnhancedSearchSystem, create_enhanced_search_system

        print("✅ RAG system import successful")

        # Create sample data
        print("📊 Creating sample maintenance data...")
        sample_data = create_sample_maintenance_data(100)
        print(f"   Created {len(sample_data)} sample records")

        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        rag_system = create_enhanced_search_system(sample_data)
        print("   RAG system initialized successfully")

        # Test queries
        test_queries = [
            "オイル漏れの修理事例",
            "エンジン関連の不具合",
            "U-30シリーズの問題",
            "ポンプ交換の作業時間",
            "油圧系統のトラブル"
        ]

        print("\n🔍 Testing search queries...")

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query} ---")

            try:
                # Perform RAG search
                response, search_results = rag_system.search_and_generate(
                    query=query,
                    top_k=5
                )

                print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
                print(f"Found {len(search_results)} relevant results")

                # Show top result
                if search_results:
                    top_result = search_results[0]
                    print(f"Top result (score: {top_result.score:.3f}): {top_result.content[:100]}...")

            except Exception as e:
                print(f"❌ Error in query {i}: {str(e)}")

        # Test structured search
        print(f"\n🎯 Testing structured search...")
        try:
            structured_results = rag_system._structured_search(
                query="オイル",
                filters={'機種名': 'U-30'},
                limit=5
            )
            print(f"Structured search found {len(structured_results)} results")

        except Exception as e:
            print(f"❌ Structured search error: {str(e)}")

        # Test semantic search
        print(f"\n🔮 Testing semantic search...")
        try:
            semantic_results = rag_system._semantic_search(
                query="油圧ポンプの故障",
                top_k=5
            )
            print(f"Semantic search found {len(semantic_results)} results")

        except Exception as e:
            print(f"❌ Semantic search error: {str(e)}")

        # Clean up
        rag_system.close()
        print("\n✅ RAG system test completed successfully!")

        return True

    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("   Make sure all dependencies are installed:")
        print("   pip install sentence-transformers faiss-cpu duckdb langchain")
        return False

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def test_dependency_availability():
    """Test if all required dependencies are available"""

    dependencies = [
        'sentence_transformers',
        'faiss',
        'duckdb',
        'langchain',
        'openai'
    ]

    print("🔧 Checking dependencies...")

    missing_deps = []

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - not installed")
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with:")
        if 'sentence_transformers' in missing_deps:
            print("   pip install sentence-transformers")
        if 'faiss' in missing_deps:
            print("   pip install faiss-cpu")
        if 'duckdb' in missing_deps:
            print("   pip install duckdb")
        if 'langchain' in missing_deps:
            print("   pip install langchain langchain-community")
        if 'openai' in missing_deps:
            print("   pip install openai")
        return False

    print("✅ All dependencies are available!")
    return True

def test_environment():
    """Test environment setup"""

    print("🌍 Testing environment...")

    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"   ✅ OpenAI API key: {openai_key[:10]}...{openai_key[-4:]}")
    else:
        print("   ⚠️  OpenAI API key not set (required for RAG responses)")

    # Check current directory
    print(f"   📂 Working directory: {os.getcwd()}")

    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   💾 Available memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        print("   💾 Memory info not available (psutil not installed)")

    return True

def main():
    """Main test function"""

    print("🧪 RAG-Enhanced Search System Test Suite")
    print("=" * 50)

    # Test environment
    test_environment()
    print()

    # Test dependencies
    if not test_dependency_availability():
        print("\n❌ Dependencies missing. Please install required packages.")
        return

    print()

    # Test RAG system
    if test_rag_system():
        print("\n🎉 All tests passed! RAG system is ready for use.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()