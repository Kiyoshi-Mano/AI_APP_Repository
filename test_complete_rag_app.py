#!/usr/bin/env python3
"""
完全版RAG統合アプリの動作テスト
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_rag_app():
    """完全版RAG統合アプリのテスト"""
    try:
        # Import test
        from complete_rag_app import RAG_AVAILABLE, create_sample_data, main
        logger.info("✅ 完全版RAG統合アプリのインポート成功")

        # Test sample data creation
        df = create_sample_data()
        logger.info(f"✅ サンプルデータ作成成功: {len(df)}行")

        # Test RAG availability
        if RAG_AVAILABLE:
            logger.info("✅ RAGシステムが利用可能")

            # Test RAG system creation
            from lightweight_rag import create_lightweight_rag_system
            rag_system = create_lightweight_rag_system(df)

            if rag_system:
                logger.info("✅ RAGシステム初期化成功")

                # Test search
                response, results = rag_system.search_and_generate("オイル漏れの修理", top_k=3)
                logger.info(f"✅ RAG検索テスト成功: {len(results)}件の結果")

                # Cleanup
                rag_system.close()
                logger.info("✅ RAGシステムクリーンアップ完了")
            else:
                logger.error("❌ RAGシステム初期化失敗")
                return False
        else:
            logger.warning("⚠️ RAGシステムが利用できません")

        # Test function availability
        from complete_rag_app import (
            llm_to_json_spec, execute_spec, generate_natural_response,
            qa_chat_with_rag, normalize_text
        )
        logger.info("✅ 必要な関数がすべて利用可能")

        logger.info("🎉 完全版RAG統合アプリのテスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        return False

def main():
    """メインテスト実行"""
    logger.info("🚀 完全版RAG統合アプリテスト開始")

    success = test_complete_rag_app()

    if success:
        logger.info("✅ すべてのテストが成功しました")
        print("\n🎉 RAG統合アプリが正常に動作します！")
        print("\n📝 使用方法:")
        print("  streamlit run complete_rag_app.py")
        print("\n🔧 機能:")
        print("  - LLM（データ分析）: 構造化データ分析")
        print("  - RAG（事例検索）: 類似修理事例検索")
        print("  - ハイブリッド: 両方を組み合わせ")
    else:
        logger.error("❌ テストに失敗しました")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)