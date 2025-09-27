#!/usr/bin/env python3
"""
RAG統合テストスクリプト
KENKIAPP.pyのRAG機能統合をテストします
"""

import pandas as pd
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_import():
    """RAGシステムのインポートテスト"""
    try:
        from lightweight_rag import LightweightRAGSystem, create_lightweight_rag_system
        logger.info("✅ RAGシステムのインポート成功")
        return True
    except ImportError as e:
        logger.error(f"❌ RAGシステムのインポート失敗: {e}")
        return False

def test_sample_data_creation():
    """サンプルデータ作成テスト"""
    try:
        # Create sample maintenance data
        sample_data = pd.DataFrame({
            '整備案件NO': ['CASE001', 'CASE002', 'CASE003', 'CASE004', 'CASE005'],
            '機種名': ['RX505', 'KX080', 'U35', 'RX505', 'KX080'],
            '作業内容': [
                'エンジンオイル漏れ修理',
                '油圧ポンプ交換',
                'エアコン故障修理',
                'エンジン始動不良',
                'バケット動作不良'
            ],
            '不具合内容': [
                'エンジンからオイル漏れ',
                '油圧ポンプ異音',
                'エアコンが冷えない',
                'エンジンがかからない',
                'バケットが上がらない'
            ],
            '部品品名': [
                'エンジンオイルシール',
                '油圧ポンプ',
                'エアコンコンプレッサー',
                'バッテリー',
                '油圧シリンダー'
            ],
            'カテゴリ': ['エンジン', '油圧', 'エアコン', '電気', '油圧'],
            '基準日': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05', '2024-05-12'])
        })

        logger.info(f"✅ サンプルデータ作成成功: {len(sample_data)}件")
        return sample_data
    except Exception as e:
        logger.error(f"❌ サンプルデータ作成失敗: {e}")
        return None

def test_rag_system_creation(df):
    """RAGシステム作成テスト"""
    try:
        from lightweight_rag import create_lightweight_rag_system

        rag_system = create_lightweight_rag_system(df)
        if rag_system:
            logger.info("✅ RAGシステム作成成功")
            return rag_system
        else:
            logger.error("❌ RAGシステム作成失敗: Noneが返された")
            return None
    except Exception as e:
        logger.error(f"❌ RAGシステム作成失敗: {e}")
        return None

def test_rag_search(rag_system):
    """RAG検索テスト"""
    test_queries = [
        "エンジンの問題",
        "油圧系統の故障",
        "オイル漏れ",
        "ポンプ交換"
    ]

    for query in test_queries:
        try:
            response, search_results = rag_system.search_and_generate(query, top_k=3)
            logger.info(f"✅ 検索テスト成功 - 質問: '{query}'")
            logger.info(f"   回答: {response[:100]}...")
            logger.info(f"   検索結果数: {len(search_results)}件")

            # Show search results
            for i, result in enumerate(search_results[:2]):
                logger.info(f"   結果{i+1}: スコア={result.score:.3f}, 内容={result.content[:50]}...")

        except Exception as e:
            logger.error(f"❌ 検索テスト失敗 - 質問: '{query}', エラー: {e}")

def test_kenkiapp_integration():
    """KENKIAPP統合テスト"""
    try:
        # Try importing updated KENKIAPP
        import importlib.util
        spec = importlib.util.spec_from_file_location("kenkiapp", "/workspaces/AI_APP_Repository/KENKIAPP.py")
        kenkiapp = importlib.util.module_from_spec(spec)

        # Check if RAG imports are working
        if hasattr(kenkiapp, 'RAG_AVAILABLE'):
            logger.info("✅ KENKIAPP RAG統合確認済み")
        else:
            logger.warning("⚠️ KENKIAPP RAG統合が見つからない")

        return True
    except Exception as e:
        logger.error(f"❌ KENKIAPP統合テスト失敗: {e}")
        return False

def main():
    """メインテスト実行"""
    logger.info("🚀 RAG統合テスト開始")

    # Test 1: RAG import
    if not test_rag_import():
        logger.error("RAGインポートテストに失敗しました")
        return False

    # Test 2: Sample data creation
    sample_df = test_sample_data_creation()
    if sample_df is None:
        logger.error("サンプルデータ作成テストに失敗しました")
        return False

    # Test 3: RAG system creation
    rag_system = test_rag_system_creation(sample_df)
    if rag_system is None:
        logger.error("RAGシステム作成テストに失敗しました")
        return False

    # Test 4: RAG search
    test_rag_search(rag_system)

    # Test 5: KENKIAPP integration
    test_kenkiapp_integration()

    # Cleanup
    try:
        rag_system.close()
        logger.info("✅ RAGシステムクリーンアップ完了")
    except:
        pass

    logger.info("🎉 RAG統合テスト完了")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)