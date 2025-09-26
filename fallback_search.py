# fallback_search.py
# Fallback search functionality when RAG system is not available

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import re

def simple_text_search(df: pd.DataFrame, query: str, search_columns: List[str] = None) -> pd.DataFrame:
    """
    Simple text search across specified columns

    Args:
        df: DataFrame to search
        query: Search query
        search_columns: Columns to search in (default: all text columns)

    Returns:
        Filtered DataFrame with matching rows
    """
    if df.empty or not query.strip():
        return df

    # Default to all text columns if not specified
    if search_columns is None:
        search_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Create search mask
    search_mask = pd.Series([False] * len(df), index=df.index)

    # Split query into keywords
    keywords = [kw.strip() for kw in query.split() if kw.strip()]

    for keyword in keywords:
        keyword_mask = pd.Series([False] * len(df), index=df.index)

        for col in search_columns:
            if col in df.columns:
                col_mask = df[col].astype(str).str.contains(
                    keyword, case=False, na=False, regex=False
                )
                keyword_mask = keyword_mask | col_mask

        search_mask = search_mask | keyword_mask

    return df[search_mask]

def enhanced_keyword_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Enhanced keyword search with better matching

    Args:
        df: DataFrame to search
        query: Search query

    Returns:
        DataFrame with results and relevance scores
    """
    if df.empty or not query.strip():
        return df

    # Define important columns for maintenance data
    important_columns = [
        '作業内容', '不具合内容', '部品品名', '機種名', 'カテゴリ',
        'work_content', 'malfunction_content', 'parts_name', 'machine_model', 'category'
    ]

    # Get available important columns
    search_columns = [col for col in important_columns if col in df.columns]

    # If no specific columns found, use all text columns
    if not search_columns:
        search_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Perform search
    result_df = simple_text_search(df, query, search_columns)

    # Add simple relevance scoring
    if not result_df.empty:
        relevance_scores = []
        query_keywords = [kw.lower().strip() for kw in query.split() if kw.strip()]

        for idx, row in result_df.iterrows():
            score = 0
            for col in search_columns:
                if col in row and pd.notna(row[col]):
                    text = str(row[col]).lower()
                    for keyword in query_keywords:
                        if keyword in text:
                            score += 1 if col in important_columns[:3] else 0.5

            relevance_scores.append(score)

        result_df = result_df.copy()
        result_df['relevance_score'] = relevance_scores
        result_df = result_df.sort_values('relevance_score', ascending=False)

        # Remove score column for display
        if 'relevance_score' in result_df.columns:
            result_df = result_df.drop('relevance_score', axis=1)

    return result_df

def generate_simple_response(query: str, result_df: pd.DataFrame) -> str:
    """
    Generate simple response text based on search results

    Args:
        query: Original search query
        result_df: Search results

    Returns:
        Response text
    """
    if result_df.empty:
        return f"「{query}」に関連するデータが見つかりませんでした。\n\n💡 検索のヒント:\n• より一般的なキーワードを使用してください\n• 複数のキーワードで検索してください\n• 期間フィルタを調整してください"

    count = len(result_df)

    # Analyze results
    response_parts = [f"「{query}」に関連するデータが{count}件見つかりました。"]

    # Check for common patterns
    if '機種名' in result_df.columns or 'machine_model' in result_df.columns:
        machine_col = '機種名' if '機種名' in result_df.columns else 'machine_model'
        if machine_col in result_df.columns:
            top_machines = result_df[machine_col].value_counts().head(3)
            if not top_machines.empty:
                machine_list = "、".join([f"{machine}({count}件)" for machine, count in top_machines.items()])
                response_parts.append(f"主な機種: {machine_list}")

    # Check for common work types
    work_columns = ['作業内容', 'work_content']
    work_col = None
    for col in work_columns:
        if col in result_df.columns:
            work_col = col
            break

    if work_col:
        top_work = result_df[work_col].value_counts().head(3)
        if not top_work.empty:
            work_list = "、".join([f"{work}({count}件)" for work, count in top_work.items()])
            response_parts.append(f"主な作業内容: {work_list}")

    return "\n".join(response_parts)

def fallback_search_and_generate(query: str, df_cases: pd.DataFrame, df_parts: pd.DataFrame) -> tuple:
    """
    Fallback search function when RAG system is not available

    Args:
        query: Search query
        df_cases: Cases DataFrame
        df_parts: Parts DataFrame

    Returns:
        Tuple of (response_text, search_results_df)
    """
    try:
        # Combine dataframes for comprehensive search
        combined_df = pd.concat([df_cases, df_parts], ignore_index=True)

        # Perform enhanced search
        search_results = enhanced_keyword_search(combined_df, query)

        # Generate response
        response = generate_simple_response(query, search_results)

        # Limit results for display performance
        display_results = search_results.head(50) if len(search_results) > 50 else search_results

        return response, display_results

    except Exception as e:
        error_msg = f"検索中にエラーが発生しました: {str(e)}"
        return error_msg, pd.DataFrame()

# Quick test function
def test_fallback_search():
    """Test the fallback search functionality"""
    # Create sample data
    sample_data = {
        '機種名': ['U-30', 'KX-080', 'U-15'],
        '作業内容': ['オイル漏れ修理', 'ポンプ交換', 'エンジンオーバーホール'],
        '不具合内容': ['油圧系統からのオイル漏れ', 'ポンプの異音と振動', 'エンジン始動不良']
    }

    df = pd.DataFrame(sample_data)

    # Test searches
    test_queries = ["オイル", "ポンプ", "エンジン", "U-30"]

    print("🧪 Fallback Search Test")
    print("=" * 30)

    for query in test_queries:
        results = enhanced_keyword_search(df, query)
        response = generate_simple_response(query, results)
        print(f"\n検索: '{query}'")
        print(f"結果: {len(results)}件")
        print(f"応答: {response[:100]}...")

if __name__ == "__main__":
    test_fallback_search()