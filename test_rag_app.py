#!/usr/bin/env python3
"""
Simple RAG test application
"""

import streamlit as st
import pandas as pd
import os

# Simple test to verify RAG functionality
st.title("RAG機能テスト")

# Test RAG dependencies
try:
    from rag_enhanced_search import RAGEnhancedSearchSystem, create_enhanced_search_system
    st.success("✅ RAG依存関係が正常にインポートされました")

    # Create sample data for testing
    sample_data = pd.DataFrame({
        '整備案件NO': ['T001', 'T002', 'T003'],
        '機種名': ['U-30', 'U-40', 'U-30'],
        '作業内容': ['エンジンオイル交換', 'オイル漏れ修理', 'ポンプ交換'],
        '不具合内容': ['定期点検', 'オイルシール破損', 'ポンプ故障'],
        '部品品名': ['エンジンオイル', 'オイルシール', '油圧ポンプ']
    })

    st.subheader("サンプルデータ")
    st.dataframe(sample_data)

    if st.button("RAGシステムテスト"):
        with st.spinner("RAGシステムを初期化中..."):
            try:
                rag_system = create_enhanced_search_system(sample_data)
                if rag_system:
                    st.success("✅ RAGシステムが正常に初期化されました")

                    # Test search
                    test_query = "オイル漏れ"
                    response, results = rag_system.search_and_generate(test_query)

                    st.subheader(f"テスト検索: '{test_query}'")
                    st.write("**回答:**")
                    st.write(response)

                    if results:
                        st.write("**検索結果:**")
                        for i, result in enumerate(results):
                            st.write(f"{i+1}. スコア: {result.score:.3f}")
                            st.write(f"   内容: {result.content}")

                else:
                    st.error("❌ RAGシステムの初期化に失敗しました")

            except Exception as e:
                st.error(f"❌ エラー: {str(e)}")

except ImportError as e:
    st.error(f"❌ RAG依存関係のインポートに失敗: {str(e)}")
    st.info("必要なパッケージをインストールしてください:")
    st.code("uv add sentence-transformers faiss-cpu langchain torch")

# Environment check
st.subheader("環境チェック")
api_key = os.environ.get('OPENAI_API_KEY')
if api_key:
    st.success("✅ OPENAI_API_KEYが設定されています")
else:
    st.warning("⚠️ OPENAI_API_KEYが設定されていません")

# Show current directory and files
st.subheader("ファイル確認")
if os.path.exists("KUBOTA_DiagDATA"):
    files = os.listdir("KUBOTA_DiagDATA")
    st.success(f"✅ データディレクトリが存在します: {len(files)}ファイル")
    for file in files:
        st.write(f"- {file}")
else:
    st.error("❌ KUBOTA_DiagDATAディレクトリが見つかりません")