# KUBOTADiag.py

import os
import re
import json
import unicodedata
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from openai import OpenAI
from rapidfuzz import fuzz
# Lazy import for RAG functionality to avoid heavy dependency loading at startup
RAGEnhancedSearchSystem = None
create_enhanced_search_system = None
LightweightRAGSystem = None
create_lightweight_rag_system = None

def _lazy_import_rag():
    """Lazy import of RAG functionality with fallback to lightweight version"""
    global RAGEnhancedSearchSystem, create_enhanced_search_system, LightweightRAGSystem, create_lightweight_rag_system

    if RAGEnhancedSearchSystem is None:
        try:
            from rag_enhanced_search import RAGEnhancedSearchSystem, create_enhanced_search_system
        except ImportError:
            pass
        except Exception:
            # Handle other import errors (like Bus errors from heavy dependencies)
            pass

    if LightweightRAGSystem is None:
        try:
            from lightweight_rag import LightweightRAGSystem, create_lightweight_rag_system
        except ImportError:
            pass
        except Exception:
            pass

    return RAGEnhancedSearchSystem, create_enhanced_search_system, LightweightRAGSystem, create_lightweight_rag_system

# Import fallback search functionality
try:
    from fallback_search import fallback_search_and_generate
except ImportError:
    fallback_search_and_generate = None

# ---------- Config ----------
APP_TITLE = "kubota建機　修理情報アプリ"
DATA_DIR = "KUBOTA_DiagDATA"
DEFAULT_FILES = [
    f"{DATA_DIR}/kubota_diagDATA_1_6.xlsm",
    f"{DATA_DIR}/kubota_diagDATA_7_12.xlsm",
]

# ---------- Utility Functions ----------

@st.cache_data(show_spinner=False)
def load_excel(paths: List[str], sheets: Optional[List[str]] = None) -> pd.DataFrame:
    """Load Excel files and concatenate sheets"""
    all_dataframes = []

    for path in paths:
        if not os.path.exists(path):
            st.warning(f"ファイルが見つかりません: {path}")
            continue

        try:
            excel_file = pd.ExcelFile(path, engine='openpyxl')
            available_sheets = excel_file.sheet_names

            if sheets is None:
                sheets_to_read = available_sheets
            else:
                sheets_to_read = [s for s in sheets if s in available_sheets]

            for sheet in sheets_to_read:
                # Try different header positions to find the correct one
                df = None
                for header_row in [0, 1]:
                    try:
                        temp_df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl', header=header_row)
                        # Check if we have meaningful column names (not all "Unnamed")
                        unnamed_count = sum(1 for col in temp_df.columns if str(col).startswith('Unnamed'))
                        if unnamed_count < len(temp_df.columns) * 0.8:  # Less than 80% unnamed columns
                            df = temp_df
                            break
                    except:
                        continue

                if df is None:
                    # Fallback: try header=0
                    df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl', header=0)

                # Add metadata columns
                df['ファイル名'] = os.path.basename(path)
                df['シート名'] = sheet
                all_dataframes.append(df)

        except Exception as e:
            st.error(f"ファイル読み込みエラー {path}: {str(e)}")
            continue

    if not all_dataframes:
        return pd.DataFrame()

    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

    # Clean column names
    combined_df.columns = combined_df.columns.astype(str).str.strip()

    # Handle duplicate column names
    cols = list(combined_df.columns)
    seen = {}
    for i, col in enumerate(cols):
        if col in seen:
            seen[col] += 1
            cols[i] = f"{col}_{seen[col]}"
        else:
            seen[col] = 0
    combined_df.columns = cols

    # Replace "« NULL »" with NaN
    combined_df = combined_df.replace(["« NULL »", "", " ", "  "], np.nan)

    return combined_df

def normalize_text(text):
    """Normalize text: NFKC, full-width to half-width, trim spaces"""
    if pd.isna(text):
        return text
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def parse_duration_to_days(duration_str):
    """Parse duration string like '27カ月' or '15日' to days"""
    if pd.isna(duration_str):
        return np.nan

    duration_str = str(duration_str).strip()

    # Extract months
    month_match = re.search(r'(\d+)カ月', duration_str)
    if month_match:
        return int(month_match.group(1)) * 30

    # Extract days
    day_match = re.search(r'(\d+)日', duration_str)
    if day_match:
        return int(day_match.group(1))

    # Try to extract just numbers
    number_match = re.search(r'(\d+)', duration_str)
    if number_match:
        return int(number_match.group(1))

    return np.nan

def create_hour_bins(hours):
    """Create hour meter bins"""
    if pd.isna(hours):
        return "不明"

    hours = float(hours)
    if hours < 250:
        return "0-250"
    elif hours < 500:
        return "250-500"
    elif hours < 1000:
        return "500-1000"
    elif hours < 2000:
        return "1000-2000"
    else:
        return "2000+"

def extract_bulletin_number(text):
    """Extract bulletin number from text"""
    if pd.isna(text):
        return np.nan

    text = str(text)
    # Look for patterns like K-XX-XXX
    bulletin_match = re.search(r'K-\d+-\d+', text)
    if bulletin_match:
        return bulletin_match.group(0)

    return np.nan

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean the dataframe"""
    if df.empty:
        return df

    df = df.copy()

    # Normalize text columns
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].apply(normalize_text)

    # Parse date columns
    date_columns = ['作業受付日', '作業開始日', '作業終了日', '売上日', '保証開始日', '部品交換日']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Parse numeric columns
    if 'アワメータ_HR' in df.columns:
        df['アワメータ_HR'] = pd.to_numeric(df['アワメータ_HR'], errors='coerce')

    if '作業時間' in df.columns:
        df['作業時間'] = pd.to_numeric(df['作業時間'], errors='coerce')

    if '部品数量' in df.columns:
        df['部品数量'] = pd.to_numeric(df['部品数量'], errors='coerce')

    # Parse duration
    if '経過日数' in df.columns:
        df['経過日数_日'] = df['経過日数'].apply(parse_duration_to_days)

    # Create base date for time analysis (売上日 > 作業終了日 > 作業開始日)
    base_date_columns = ['売上日', '作業終了日', '作業開始日']
    df['基準日'] = pd.NaT

    for col in base_date_columns:
        if col in df.columns:
            if df['基準日'].isna().all():
                df['基準日'] = df[col]
            else:
                df['基準日'] = df['基準日'].fillna(df[col])

    # Create time-based columns
    if '基準日' in df.columns:
        df['年'] = df['基準日'].dt.year
        df['月'] = df['基準日'].dt.month
        df['週'] = df['基準日'].dt.isocalendar().week

    # Create hour bins
    if 'アワメータ_HR' in df.columns:
        df['アワ帯'] = df['アワメータ_HR'].apply(create_hour_bins)

    # Create malfunction flag
    malfunction_columns = ['カテゴリ', '不具合内容', '故障状況']
    df['不具合フラグ'] = 0
    for col in malfunction_columns:
        if col in df.columns:
            mask = df[col].notna() & (df[col] != '') & (~df[col].str.contains('点検|検査', na=False))
            df.loc[mask, '不具合フラグ'] = 1

    # Extract bulletin numbers
    bulletin_columns = ['作業内容', '作業明細備考']
    df['ブルチンNO'] = np.nan
    for col in bulletin_columns:
        if col in df.columns:
            bulletin_nums = df[col].apply(extract_bulletin_number)
            df['ブルチンNO'] = df['ブルチンNO'].fillna(bulletin_nums)

    return df

def split_fact_tables(df: pd.DataFrame) -> tuple:
    """Split into case-level and parts-level fact tables"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Case-level facts (unique by 整備案件NO)
    case_columns = [col for col in df.columns if col not in ['部品品番', '部品品名', '部品数量']]
    if '整備案件NO' in df.columns:
        df_cases = df[case_columns].drop_duplicates(subset=['整備案件NO'], keep='last')
    else:
        df_cases = df[case_columns].drop_duplicates()

    # Parts-level facts (all rows)
    df_parts = df.copy()

    return df_cases, df_parts

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters based on sidebar state"""
    if df.empty:
        return df

    filtered_df = df.copy()

    # Date range filter
    if filters.get('date_range') and '基準日' in filtered_df.columns:
        start_date, end_date = filters['date_range']
        if start_date:
            filtered_df = filtered_df[filtered_df['基準日'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['基準日'] <= pd.to_datetime(end_date)]

    # Categorical filters
    categorical_filters = ['機種名', 'シリーズ', '種別', '部門名', '整備区分']
    for filter_name in categorical_filters:
        if filters.get(filter_name) and filter_name in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[filter_name].isin(filters[filter_name])]

    # Hour meter filter
    if filters.get('hour_bins') and 'アワ帯' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['アワ帯'].isin(filters['hour_bins'])]

    # Exclude inspections toggle
    if filters.get('exclude_inspections', False):
        inspection_keywords = ['点検', '検査', '特自検', '定期点検']
        if '整備区分' in filtered_df.columns:
            mask = ~filtered_df['整備区分'].str.contains('|'.join(inspection_keywords), na=False)
            filtered_df = filtered_df[mask]

    # Apply sampling for performance
    if filters.get('use_sampling', False) and filters.get('sample_size'):
        sample_size = filters['sample_size']
        if len(filtered_df) > sample_size:
            filtered_df = filtered_df.sample(n=sample_size, random_state=42)

    return filtered_df

# ---------- Visualization Functions ----------

def render_dashboard(df_cases: pd.DataFrame, df_parts: pd.DataFrame, filters: Dict[str, Any]):
    """Render the main dashboard with required visualizations"""

    st.header("📊 ダッシュボード")

    # Choose data based on aggregation unit
    df_active = df_parts if filters.get('aggregation_unit') == 'parts' else df_cases

    if df_active.empty:
        st.warning("表示するデータがありません")
        return

    # 1. Machine type malfunction trends
    st.subheader("1. 機種名ごとの不具合傾向")

    if '機種名' in df_active.columns and '不具合フラグ' in df_active.columns:
        malfunction_data = df_active[df_active['不具合フラグ'] == 1]
        machine_counts = malfunction_data['機種名'].value_counts().head(20)

        if not machine_counts.empty:
            fig = px.bar(
                x=machine_counts.values,
                y=machine_counts.index,
                orientation='h',
                title="機種名別不具合件数 (上位20)",
                labels={'x': '件数', 'y': '機種名'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Show detailed table
            with st.expander("詳細データ"):
                total_counts = df_active['機種名'].value_counts()
                detail_df = pd.DataFrame({
                    '機種名': machine_counts.index,
                    '不具合件数': machine_counts.values,
                    '総件数': [total_counts.get(machine, 0) for machine in machine_counts.index],
                })
                detail_df['不具合率(%)'] = (detail_df['不具合件数'] / detail_df['総件数'] * 100).round(2)
                st.dataframe(detail_df)

    # 2. Annual malfunction trends
    st.subheader("2. 年間の不具合傾向")

    if '基準日' in df_active.columns and '不具合フラグ' in df_active.columns:
        malfunction_data = df_active[df_active['不具合フラグ'] == 1]
        if not malfunction_data.empty:
            monthly_data = malfunction_data.set_index('基準日').resample('M').size()

            fig = px.line(
                x=monthly_data.index,
                y=monthly_data.values,
                title="月別不具合件数推移",
                labels={'x': '月', 'y': '不具合件数'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # 3. Hour meter vs malfunction trends
    st.subheader("3. アワーメータに対する不具合傾向")

    if 'アワ帯' in df_active.columns and '不具合フラグ' in df_active.columns:
        hour_malfunction = df_active[df_active['不具合フラグ'] == 1]['アワ帯'].value_counts()

        if not hour_malfunction.empty:
            fig = px.bar(
                x=hour_malfunction.index,
                y=hour_malfunction.values,
                title="アワ帯別不具合件数",
                labels={'x': 'アワ帯', 'y': '不具合件数'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # 4. Parts replacement ratio for malfunctions
    st.subheader("4. 不具合に対する交換部品比率")

    if '部品品名' in df_parts.columns and '不具合フラグ' in df_parts.columns:
        parts_malfunction = df_parts[df_parts['不具合フラグ'] == 1]
        parts_counts = parts_malfunction['部品品名'].value_counts().head(10)

        if not parts_counts.empty:
            fig = px.pie(
                values=parts_counts.values,
                names=parts_counts.index,
                title="不具合時の交換部品比率 (上位10)"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_additional_kpis(df_cases: pd.DataFrame, _: pd.DataFrame):
    """Render additional KPI visualizations"""

    st.header("📈 追加KPI")

    col1, col2 = st.columns(2)

    with col1:
        # Bulletin rankings
        if 'ブルチンNO' in df_cases.columns:
            bulletin_counts = df_cases['ブルチンNO'].value_counts().head(10)
            if not bulletin_counts.empty:
                fig = px.bar(
                    x=bulletin_counts.values,
                    y=bulletin_counts.index,
                    orientation='h',
                    title="ブルチン対応ランキング"
                )
                st.plotly_chart(fig, use_container_width=True)


    with col2:
        # Work location analysis
        if '整備場所' in df_cases.columns:
            location_counts = df_cases['整備場所'].value_counts()
            if not location_counts.empty:
                fig = px.bar(
                    x=location_counts.index,
                    y=location_counts.values,
                    title="整備場所別件数"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Department MTTR analysis
        if '部門名' in df_cases.columns and '作業時間' in df_cases.columns:
            dept_mttr = df_cases.groupby('部門名')['作業時間'].median().sort_values(ascending=False)
            if not dept_mttr.empty:
                fig = px.bar(
                    x=dept_mttr.index,
                    y=dept_mttr.values,
                    title="部門別MTTR (中央値・時間)"
                )
                st.plotly_chart(fig, use_container_width=True)

def render_table(df: pd.DataFrame):
    """Render detailed table view"""
    st.header("📋 詳細テーブル")

    if df.empty:
        st.warning("表示するデータがありません")
        return

    # Search functionality
    search_term = st.text_input("🔍 検索", help="テーブル内容を検索")

    if search_term:
        # Search across all text columns
        text_columns = df.select_dtypes(include=['object']).columns
        mask = df[text_columns].astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        df_filtered = df[mask]
    else:
        df_filtered = df

    st.write(f"表示件数: {len(df_filtered):,} / 全体: {len(df):,}")

    # Display dataframe
    st.dataframe(df_filtered, use_container_width=True, height=600)

def render_quality_report(df: pd.DataFrame):
    """Render data quality report"""
    st.header("🔍 データ品質レポート")

    if df.empty:
        st.warning("分析するデータがありません")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("欠損率分析")
        missing_rates = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_df = pd.DataFrame({
            '列名': missing_rates.index,
            '欠損率(%)': missing_rates.values
        })
        missing_df = missing_df[missing_df['欠損率(%)'] > 0]

        if not missing_df.empty:
            fig = px.bar(
                missing_df.head(20),
                x='欠損率(%)',
                y='列名',
                orientation='h',
                title="欠損率上位20列"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("欠損データはありません")

    with col2:
        st.subheader("重複データ分析")
        if '整備案件NO' in df.columns:
            duplicate_counts = df['整備案件NO'].value_counts()
            duplicates = duplicate_counts[duplicate_counts > 1]

            if not duplicates.empty:
                st.warning(f"重複する整備案件NO: {len(duplicates)}件")
                st.dataframe(duplicates.head(10).to_frame('件数'))
            else:
                st.success("重複する整備案件NOはありません")

        # Value counts for key categorical fields
        st.subheader("主要項目の値分布")
        key_fields = ['機種名', 'シリーズ', '整備区分', '保証区分']
        for field in key_fields:
            if field in df.columns:
                value_counts = df[field].value_counts().head(5)
                if not value_counts.empty:
                    st.write(f"**{field}** (上位5)")
                    st.dataframe(value_counts.to_frame('件数'))

# ---------- RAG-Enhanced Search Integration ----------

def initialize_rag_system(df_cases: pd.DataFrame, df_parts: pd.DataFrame) -> Optional[object]:
    """Initialize RAG-enhanced search system with maintenance data"""
    try:
        # Lazy import RAG functionality
        RAGEnhancedSearchSystem, create_func, LightweightRAGSystem, create_lightweight_func = _lazy_import_rag()

        # Try enhanced RAG first, fallback to lightweight
        if create_func is not None:
            try:
                # Check if OpenAI API key is available
                if not os.environ.get('OPENAI_API_KEY'):
                    st.warning("OPENAI_API_KEYが設定されていません。軽量RAGシステムを使用します。")
                    return _initialize_lightweight_rag(df_cases, df_parts, create_lightweight_func)

                # Combine cases and parts data for comprehensive search
                combined_df = pd.concat([df_cases, df_parts], ignore_index=True)

                if combined_df.empty:
                    st.warning("データがありません。RAGシステムを初期化できません。")
                    return None

                # Initialize enhanced RAG system
                with st.spinner("高機能RAGシステムを初期化中..."):
                    rag_system = create_func(combined_df)

                if rag_system is None:
                    st.info("高機能RAGシステムの初期化に失敗しました。軽量RAGシステムを使用します。")
                    return _initialize_lightweight_rag(df_cases, df_parts, create_lightweight_func)
                else:
                    st.success("高機能RAGシステムが正常に初期化されました。")
                    return rag_system

            except Exception as e:
                st.warning(f"高機能RAGシステムでエラーが発生: {str(e)}")
                st.info("軽量RAGシステムにフォールバックします。")
                return _initialize_lightweight_rag(df_cases, df_parts, create_lightweight_func)

        else:
            return _initialize_lightweight_rag(df_cases, df_parts, create_lightweight_func)

    except Exception as e:
        st.error(f"RAGシステムの初期化でエラーが発生しました: {str(e)}")
        return None

def _initialize_lightweight_rag(df_cases: pd.DataFrame, df_parts: pd.DataFrame, create_lightweight_func) -> Optional[object]:
    """Initialize lightweight RAG system"""
    if create_lightweight_func is None:
        st.info("軽量RAG機能が利用できません。従来の検索機能を使用します。")
        return None

    try:
        # Combine cases and parts data for comprehensive search
        combined_df = pd.concat([df_cases, df_parts], ignore_index=True)

        # Check if we have enough data
        if combined_df.empty:
            st.warning("データがありません。RAGシステムを初期化できません。")
            return None

        # Initialize lightweight RAG system
        with st.spinner("軽量RAGシステムを初期化中..."):
            rag_system = create_lightweight_func(combined_df)

        if rag_system is None:
            st.info("軽量RAGシステムの初期化に失敗しました。従来の検索機能を使用します。")
            return None
        else:
            st.success("軽量RAGシステムが正常に初期化されました。")
            return rag_system

    except Exception as e:
        st.warning(f"軽量RAGシステムの初期化でエラーが発生しました: {str(e)}")
        st.info("従来の検索機能を使用します。")
        return None

def rag_enhanced_query(question: str, df_cases: pd.DataFrame, df_parts: pd.DataFrame, **kwargs):
    """Perform RAG-enhanced query for improved search accuracy with better error handling"""

    # Initialize RAG system if not already done
    if 'rag_system' not in st.session_state:
        with st.spinner("RAGシステムを初期化中..."):
            st.session_state.rag_system = initialize_rag_system(df_cases, df_parts)

    rag_system = st.session_state.rag_system

    if rag_system is None:
        st.info("RAGシステムが利用できません。従来の検索方式を使用します。")
        # Fallback to original LLM query if RAG system fails
        return llm_enhanced_query(question, df_cases, df_parts)

    try:
        # Get parameters from kwargs or use defaults
        top_k = kwargs.get('top_k', 15)
        semantic_weight = kwargs.get('semantic_weight', 0.6)

        # Show RAG search progress
        with st.spinner("RAG検索を実行中..."):
            # Perform RAG-enhanced search with user-specified parameters
            response, search_results = rag_system.search_and_generate(
                query=question,
                filters=None,
                top_k=top_k,
                semantic_weight=semantic_weight
            )

        # Convert search results to DataFrame for display
        if search_results:
            result_data = []
            for i, result in enumerate(search_results[:10]):  # Show top 10 results
                # Create more informative display
                content_preview = result.content[:300] + "..." if len(result.content) > 300 else result.content

                # Extract key metadata for display
                metadata = result.metadata
                metadata_display = []

                if isinstance(metadata, dict):
                    if metadata.get('machine_model'):
                        metadata_display.append(f"機種: {metadata['machine_model']}")
                    if metadata.get('category'):
                        metadata_display.append(f"カテゴリ: {metadata['category']}")
                    if metadata.get('case_no'):
                        metadata_display.append(f"案件: {metadata['case_no']}")
                    if metadata.get('work_date'):
                        metadata_display.append(f"日付: {metadata['work_date']}")

                metadata_str = " | ".join(metadata_display) if metadata_display else str(metadata)[:100]

                result_data.append({
                    '順位': i + 1,
                    'スコア': f"{result.score:.3f}",
                    '関連内容': content_preview,
                    '詳細情報': metadata_str
                })

            results_df = pd.DataFrame(result_data)

            # Add summary information
            avg_score = sum(r.score for r in search_results) / len(search_results)
            st.info(f"🔍 RAG検索結果: {len(search_results)}件の関連情報を発見（平均関連度: {avg_score:.3f}）")

        else:
            results_df = pd.DataFrame()
            st.warning("RAG検索で関連情報が見つかりませんでした。")

        return response, results_df

    except Exception as e:
        error_msg = f"RAG検索でエラーが発生しました: {str(e)}"
        st.error(error_msg)

        # Provide more detailed error information if available
        if "OpenAI" in str(e):
            st.error("OpenAI APIの問題が発生しました。APIキーまたはネットワーク接続を確認してください。")
        elif "FAISS" in str(e) or "vector" in str(e).lower():
            st.error("ベクトル検索の問題が発生しました。システムを再初期化してください。")
        elif "DuckDB" in str(e):
            st.error("データベース検索の問題が発生しました。")

        # Reset RAG system on critical errors
        if "FAISS" in str(e) or "vector" in str(e).lower():
            st.session_state.pop('rag_system', None)
            st.info("RAGシステムをリセットしました。次回の検索時に再初期化されます。")

        # Fallback to original method
        st.info("従来の検索方式にフォールバックします。")
        return llm_enhanced_query(question, df_cases, df_parts)

def llm_enhanced_query(question: str, df_cases: pd.DataFrame, df_parts: pd.DataFrame):
    """Enhanced LLM query (original method with improvements)"""
    try:
        # Debug output
        st.write(f"🔍 分析開始: {question}")

        # Check if dataframes are available
        if df_cases.empty and df_parts.empty:
            return "データが読み込まれていません。データファイルを確認してください。", pd.DataFrame()

        # Get available columns
        available_columns = list(set(df_cases.columns.tolist() + df_parts.columns.tolist()))
        st.write(f"📊 利用可能な列数: {len(available_columns)}")

        # Generate query specification
        st.write("🤖 OpenAI APIに問い合わせ中...")
        spec = llm_to_json_spec(question, available_columns)

        if spec:
            # Execute query
            result_df = execute_spec(df_cases, df_parts, spec)

            if not result_df.empty:
                # Generate natural language response
                response_text = generate_natural_response(question, result_df, spec)
                return response_text, result_df
            else:
                # Try fallback search if no results
                if fallback_search_and_generate:
                    fallback_response, fallback_results = fallback_search_and_generate(question, df_cases, df_parts)
                    if not fallback_results.empty:
                        return f"構造化検索で結果が見つからなかったため、キーワード検索を実行しました:\n\n{fallback_response}", fallback_results

                return "該当するデータが見つかりませんでした。", pd.DataFrame()
        else:
            # Try fallback search if query cannot be understood
            if fallback_search_and_generate:
                return fallback_search_and_generate(question, df_cases, df_parts)
            else:
                return "質問を理解できませんでした。", pd.DataFrame()

    except Exception as e:
        # Try fallback search if LLM query fails
        if fallback_search_and_generate:
            try:
                return fallback_search_and_generate(question, df_cases, df_parts)
            except:
                pass

        return f"検索エラー: {str(e)}", pd.DataFrame()

# ---------- Q&A Chat Functions ----------

def llm_to_json_spec(question: str, columns: List[str]) -> Dict[str, Any]:
    """Convert user question to structured query specification using OpenAI"""

    # Get OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    system_prompt = f"""あなたは建機の修理データアナリストです。
与えられた日本語の質問を、pandasで実行可能な集計仕様のJSONに変換してください。
データ列名は日本語です。利用可能な列名: {', '.join(columns)}

出力JSONスキーマ：
{{
  "time_range": {{"from": "YYYY-MM-DD|null", "to": "YYYY-MM-DD|null"}},
  "filters": [{{"column": "列名", "op": "==|!=|in|not in|contains|regex|fuzzy_search", "value": "any"}}, ...],
  "groupby": ["列名", ...],
  "metrics": [{{"name":"件数","expr":"count"}}, {{"name":"数量合計","expr":"sum:部品数量"}}, ...],
  "topn": 10,
  "sort": {{"by":"件数","asc":false}},
  "free_text_search": {{"enabled": true|false, "keywords": ["keyword1", "keyword2"], "target_columns": ["故障状況", "不具合内容", "作業内容", "作業明細備考", "部品品名"]}}
}}

**重要**: metrics[].name は最終テーブルの列名として使用されます。
- expr:"count" の場合: groupbyのサイズ（全レコード数）を集計し、列名は必ずname（例："name": "件数"）
- expr:"sum:<列名>" の場合: 合計結果の列名は必ずname（例："name": "数量合計"）
- 人間が理解しやすい列名を使用してください（「件数」「合計」「平均値」など）

特別な検索機能:
- fuzzy_search: あいまい検索（類似した文字列も検索）
- free_text_search: フリーワード検索（複数の列を横断して検索）

故障現象や作業内容に関する質問の場合は、free_text_search を活用してください。
質問に曖昧さがある場合は、常識的に補完してください（例：「1年」は直近365日）。
JSONのみを出力してください。説明は不要です。"""

    try:
        st.write("🌐 OpenAI APIに接続中...")
        st.write(f"📝 使用モデル: gpt-4o-mini")
        st.write(f"🔑 APIキー確認: {'✅' if api_key else '❌'}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=1000,
            timeout=30  # Add timeout to prevent hanging
        )

        st.write("✅ OpenAI APIレスポンス受信完了")
        json_text = response.choices[0].message.content.strip()
        # Remove code block markers if present
        json_text = re.sub(r'^```json\s*|\s*```$', '', json_text, flags=re.MULTILINE)

        parsed_json = json.loads(json_text)
        st.write("✅ JSONパース成功")
        return parsed_json

    except json.JSONDecodeError as e:
        st.error(f"JSON解析エラー: {str(e)}")
        st.write("受信したテキスト:", json_text if 'json_text' in locals() else "N/A")
        return {}
    except Exception as e:
        error_msg = f"OpenAI API エラー: {str(e)}"
        error_type = type(e).__name__

        # More detailed error information
        st.error(f"[{error_type}] {error_msg}")
        st.write("🔍 詳細エラー情報:")
        st.write(f"   - エラータイプ: {error_type}")
        st.write(f"   - エラーメッセージ: {str(e)}")

        # Add helpful information for common errors
        if "API key" in str(e) or "authentication" in str(e).lower():
            st.write("💡 API キーを確認してください")
        elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
            st.write("💡 API使用量制限に達しました。しばらく待ってから再試行してください")
        elif "timeout" in str(e).lower():
            st.write("💡 接続がタイムアウトしました。再試行してください")
        elif "404" in str(e):
            st.write("💡 **404エラー発生** - モデル名またはエンドポイントに問題があります")
            st.write(f"   - 使用モデル: gpt-4o-mini")
            import openai as openai_lib
            st.write(f"   - OpenAIバージョン: {openai_lib.__version__}")

        # Import traceback for more details
        import traceback
        st.write("📊 完全なトレースバック:")
        st.code(traceback.format_exc())

        return {}

def execute_spec(df_cases: pd.DataFrame, df_parts: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """Execute query specification safely on dataframes"""

    try:
        # Choose appropriate dataframe
        if any('部品' in str(metric.get('expr', '')) for metric in spec.get('metrics', [])):
            df = df_parts.copy()
        else:
            df = df_cases.copy()

        if df.empty:
            return pd.DataFrame()

        # Apply time range filter
        time_range = spec.get('time_range', {})
        if time_range.get('from') and '基準日' in df.columns:
            df = df[df['基準日'] >= pd.to_datetime(time_range['from'])]
        if time_range.get('to') and '基準日' in df.columns:
            df = df[df['基準日'] <= pd.to_datetime(time_range['to'])]

        # Apply filters
        for filter_item in spec.get('filters', []):
            column = filter_item.get('column')
            op = filter_item.get('op')
            value = filter_item.get('value')

            if column not in df.columns:
                continue

            if op == '==':
                df = df[df[column] == value]
            elif op == '!=':
                df = df[df[column] != value]
            elif op == 'in':
                df = df[df[column].isin(value if isinstance(value, list) else [value])]
            elif op == 'not in':
                df = df[~df[column].isin(value if isinstance(value, list) else [value])]
            elif op == 'contains':
                df = df[df[column].astype(str).str.contains(str(value), na=False, case=False)]
            elif op == 'regex':
                df = df[df[column].astype(str).str.contains(str(value), na=False, case=False, regex=True)]
            elif op == 'fuzzy_search':
                # Fuzzy search using rapidfuzz
                mask = df[column].astype(str).apply(
                    lambda x: fuzz.partial_ratio(str(value).lower(), str(x).lower()) > 70
                )
                df = df[mask]

        # Apply free text search if specified
        free_text_search = spec.get('free_text_search', {})
        if free_text_search.get('enabled', False):
            keywords = free_text_search.get('keywords', [])
            target_columns = free_text_search.get('target_columns', [])

            if keywords and target_columns:
                # Create search mask across multiple columns
                search_mask = pd.Series([False] * len(df), index=df.index)

                for keyword in keywords:
                    for col in target_columns:
                        if col in df.columns:
                            col_mask = df[col].astype(str).str.contains(
                                keyword, na=False, case=False, regex=False
                            )
                            search_mask = search_mask | col_mask

                df = df[search_mask]

        # Get groupby columns
        groupby_cols = [col for col in spec.get('groupby', []) if col in df.columns]

        # Process metrics safely - create DataFrame for each metric and merge
        metric_frames = []
        for metric in spec.get('metrics', []):
            metric_name = metric.get('name')
            expr = metric.get('expr', 'count')

            if not metric_name:
                continue

            if groupby_cols:
                # With groupby
                grouped = df.groupby(groupby_cols, dropna=False)

                if expr == 'count':
                    # Use size() to get total record count
                    metric_result = grouped.size().rename(metric_name).reset_index()
                elif expr.startswith('sum:'):
                    col_name = expr.split(':', 1)[1]
                    if col_name not in df.columns:
                        df[col_name] = 0  # Create missing column with zeros
                    # Convert to numeric and handle errors
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                    metric_result = grouped[col_name].sum().rename(metric_name).reset_index()
                elif expr.startswith('mean:'):
                    col_name = expr.split(':', 1)[1]
                    if col_name not in df.columns:
                        df[col_name] = 0
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                    metric_result = grouped[col_name].mean().rename(metric_name).reset_index()
                else:
                    raise ValueError(f"Unsupported expression: {expr}")

            else:
                # No groupby - global aggregation
                if expr == 'count':
                    metric_result = pd.DataFrame({metric_name: [len(df)]})
                elif expr.startswith('sum:'):
                    col_name = expr.split(':', 1)[1]
                    if col_name not in df.columns:
                        df[col_name] = 0
                    col_sum = pd.to_numeric(df[col_name], errors='coerce').fillna(0).sum()
                    metric_result = pd.DataFrame({metric_name: [col_sum]})
                elif expr.startswith('mean:'):
                    col_name = expr.split(':', 1)[1]
                    if col_name not in df.columns:
                        df[col_name] = 0
                    col_mean = pd.to_numeric(df[col_name], errors='coerce').fillna(0).mean()
                    metric_result = pd.DataFrame({metric_name: [col_mean]})
                else:
                    raise ValueError(f"Unsupported expression: {expr}")

            metric_frames.append(metric_result)

        # Combine all metrics
        if not metric_frames:
            # Default to count if no metrics specified
            if groupby_cols:
                result_df = df.groupby(groupby_cols, dropna=False).size().rename('件数').reset_index()
            else:
                result_df = pd.DataFrame({'件数': [len(df)]})
        else:
            result_df = metric_frames[0]
            for metric_frame in metric_frames[1:]:
                if groupby_cols:
                    # Merge on groupby columns
                    result_df = result_df.merge(metric_frame, on=groupby_cols, how='outer')
                else:
                    # Concatenate columns for global metrics
                    result_df = pd.concat([result_df, metric_frame], axis=1)

        # Safe sorting
        sort_spec = spec.get('sort', {})
        sort_by = sort_spec.get('by', '件数')
        sort_asc = sort_spec.get('asc', False)

        # Check if sort column exists, fallback to first metric column
        if sort_by not in result_df.columns:
            metric_names = [m.get('name') for m in spec.get('metrics', []) if m.get('name') in result_df.columns]
            if metric_names:
                sort_by = metric_names[0]
            elif '件数' in result_df.columns:
                sort_by = '件数'
            else:
                sort_by = None

        if sort_by and sort_by in result_df.columns:
            # Convert to numeric for proper sorting
            result_df[sort_by] = pd.to_numeric(result_df[sort_by], errors='ignore')
            result_df = result_df.sort_values(by=sort_by, ascending=sort_asc, kind='mergesort')

        # Apply topn limit
        topn = spec.get('topn')
        if isinstance(topn, int) and topn > 0:
            result_df = result_df.head(topn)

        # Clean up and return
        return result_df.reset_index(drop=True)

    except Exception as e:
        error_msg = f"クエリ実行エラー: {str(e)}"

        # Add helpful debugging information
        if "Column(s)" in str(e) and "do not exist" in str(e):
            error_msg += "\n\n💡 ヒント:\n"
            error_msg += "• より広い条件で検索する\n"
            error_msg += "• 期間を変更する\n"
            error_msg += "• 異なる項目で集計する"

        st.error(error_msg)
        return pd.DataFrame()

def generate_natural_response(question: str, result_df: pd.DataFrame, _: Dict[str, Any]) -> str:
    """Generate natural language response for query results"""

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return "分析結果を表示します。"

    client = OpenAI(api_key=api_key)

    # Prepare context about the results
    result_summary = f"質問: {question}\n\n"

    if result_df.empty:
        result_summary += "該当するデータが見つかりませんでした。"
    else:
        result_summary += f"検索結果: {len(result_df)}件\n"
        result_summary += result_df.to_string(index=False, max_rows=10)

        if len(result_df) > 10:
            result_summary += "\n...（他の結果は省略）"

    system_prompt = """あなたは建機の修理データ分析専門のアシスタントです。
分析結果を基に、自然で分かりやすい日本語で回答してください。

回答のガイドライン:
- 数値は具体的に示す
- 傾向や特徴があれば指摘する
- 実用的な洞察を提供する
- 簡潔で明確な表現を使う
- 必要に応じて改善提案を含める

200文字以内で回答してください。"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": result_summary}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"分析結果を表示します。（自然言語生成エラー: {str(e)}）"

def qa_chat(df_cases: pd.DataFrame, df_parts: pd.DataFrame):
    """Render Q&A chat interface"""

    st.header("💬 QAチャット")
    st.write("データベースの内容について質問してください。")

    # Search mode selection
    col1, col2 = st.columns([2, 1])

    with col1:
        search_mode = st.radio(
            "🔍 検索モード選択",
            options=["RAG検索", "LLM検索", "自動選択"],
            index=2,  # Default to auto-select
            horizontal=True,
            help="""
            - RAG検索: 高精度なベクトル検索 + LLM回答生成
            - LLM検索: 従来のOpenAI API + 構造化検索
            - 自動選択: システムが最適な検索方法を自動選択
            """
        )

    with col2:
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("🔄 RAG再初期化", help="RAGシステムを再初期化します"):
                if 'rag_system' in st.session_state:
                    del st.session_state.rag_system
                st.rerun()

        with col2b:
            compare_mode = st.checkbox(
                "⚖️ 比較モード",
                value=False,
                help="RAG検索とLLM検索の結果を並べて比較表示"
            )

    # Show system status based on search mode
    if search_mode == "RAG検索":
        if 'rag_system' in st.session_state and st.session_state.rag_system is not None:
            st.success("🚀 RAG検索システムが有効です")
        else:
            st.warning("⚠️ RAGシステムが利用できません。LLM検索を使用してください。")
    elif search_mode == "LLM検索":
        st.info("🤖 LLM検索モードを使用中")
    else:  # 自動選択
        if 'rag_system' in st.session_state and st.session_state.rag_system is not None:
            st.success("🚀 RAG検索システムが利用可能（自動選択モード）")
        else:
            st.info("🤖 LLM検索システムを使用中（自動選択モード）")

    # Advanced settings for RAG search (expandable)
    if search_mode == "RAG検索":
        with st.expander("⚙️ RAG検索の詳細設定"):
            col1, col2, col3 = st.columns(3)

            with col1:
                top_k = st.slider(
                    "検索結果数",
                    min_value=5,
                    max_value=20,
                    value=10,
                    help="RAG検索で取得する関連文書の数"
                )

            with col2:
                semantic_weight = st.slider(
                    "セマンティック重み",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="ベクトル検索vs構造化検索のバランス（1.0=完全ベクトル検索）"
                )

            with col3:
                use_filters = st.checkbox(
                    "フィルタ適用",
                    value=False,
                    help="サイドバーのフィルタをRAG検索にも適用"
                )

    st.info("💡 質問例:\n" +
            "・「オイル漏れの件数は？」\n" +
            "・「エンジン関連の故障を教えて」\n" +
            "・「今年のトラブル上位3位は？」\n" +
            "・「油圧系統の問題があった機種は？」\n" +
            "・「最近のポンプ交換の傾向は？」\n" +
            "・「U-30シリーズの故障パターン」\n" +
            "・「高稼働時間での不具合傾向」\n" +
            "・「予防保全のポイントは？」\n" +
            "・「部品交換コストが高い修理は？」")

    # Check for OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        st.error("OpenAI APIキーが設定されていません。環境変数 OPENAI_API_KEY を設定してください。")
        return

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat history management
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.write(f"💬 チャット履歴: {len(st.session_state.chat_history)}件")

    with col2:
        if st.button("🗑️ 履歴クリア", help="チャット履歴をすべてクリアします"):
            st.session_state.chat_history = []
            st.rerun()

    with col3:
        max_history = st.selectbox(
            "最大履歴数",
            options=[10, 20, 50, 100],
            index=1,  # Default to 20 (index 1)
            help="保持するチャット履歴の最大数"
        )

    # Limit chat history to prevent memory issues
    if len(st.session_state.chat_history) > max_history:
        st.session_state.chat_history = st.session_state.chat_history[-max_history:]

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
            if "dataframe" in chat:
                # Limit dataframe display size for performance
                df_display = chat["dataframe"]
                if len(df_display) > 100:
                    st.write(f"📊 データ件数: {len(df_display)}件 (上位100件を表示)")
                    df_display = df_display.head(100)
                st.dataframe(df_display, use_container_width=True, height=300)

    # Chat input
    if prompt := st.chat_input("質問を入力してください"):
        # Ensure chat history list exists and is writable
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Limit prompt length to prevent API issues
        if len(prompt) > 1000:
            st.warning("質問が長すぎます。1000文字以内で入力してください。")
        else:
            # Add user message to chat history
            try:
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.write(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("分析中..."):
                        try:
                            # Debug: Check OPENAI_API_KEY availability
                            api_key = os.environ.get('OPENAI_API_KEY')
                            if not api_key:
                                st.error("OPENAI_API_KEYが設定されていません。環境変数を確認してください。")
                                return

                            # Check if comparison mode is enabled
                            if compare_mode:
                                # Compare both search methods side by side
                                st.info("⚖️ 比較モード: 両方の検索方法を実行中...")

                                col_rag, col_llm = st.columns(2)

                                with col_rag:
                                    st.subheader("🚀 RAG検索結果")
                                    try:
                                        rag_params = {}
                                        if 'top_k' in locals():
                                            rag_params['top_k'] = top_k
                                        if 'semantic_weight' in locals():
                                            rag_params['semantic_weight'] = semantic_weight

                                        rag_response, rag_df = rag_enhanced_query(
                                            prompt, df_cases, df_parts, **rag_params
                                        )
                                        st.write(rag_response)
                                        if not rag_df.empty:
                                            st.dataframe(rag_df, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"RAG検索エラー: {str(e)}")

                                with col_llm:
                                    st.subheader("🤖 LLM検索結果")
                                    try:
                                        llm_response, llm_df = llm_enhanced_query(prompt, df_cases, df_parts)
                                        st.write(llm_response)
                                        if not llm_df.empty:
                                            st.dataframe(llm_df, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"LLM検索エラー: {str(e)}")

                                # Use RAG response for chat history (prioritize RAG)
                                response_text = rag_response if 'rag_response' in locals() else llm_response if 'llm_response' in locals() else "比較モードでエラーが発生しました。"
                                result_df = rag_df if 'rag_df' in locals() and not rag_df.empty else llm_df if 'llm_df' in locals() else pd.DataFrame()

                            else:
                                # Select search method based on user choice
                                if search_mode == "RAG検索":
                                    # Force RAG search with custom parameters
                                    st.info("🚀 RAG検索を実行中...")

                                    # Get RAG parameters from UI (if available)
                                    rag_params = {}
                                    if 'top_k' in locals():
                                        rag_params['top_k'] = top_k
                                    if 'semantic_weight' in locals():
                                        rag_params['semantic_weight'] = semantic_weight

                                    response_text, result_df = rag_enhanced_query(
                                        prompt, df_cases, df_parts, **rag_params
                                    )
                                elif search_mode == "LLM検索":
                                    # Force LLM search
                                    st.info("🤖 LLM検索を実行中...")
                                    response_text, result_df = llm_enhanced_query(prompt, df_cases, df_parts)
                                else:  # 自動選択
                                    # Auto-select: Try RAG first, fallback to LLM
                                    try:
                                        st.info("🔍 自動選択: RAG検索を試行中...")
                                        response_text, result_df = rag_enhanced_query(prompt, df_cases, df_parts)
                                        st.success("✅ RAG検索で回答を生成しました")
                                    except Exception as rag_error:
                                        st.warning(f"RAG検索でエラーが発生: {str(rag_error)}")
                                        st.info("🔄 LLM検索にフォールバックします...")
                                        response_text, result_df = llm_enhanced_query(prompt, df_cases, df_parts)
                                        st.success("✅ LLM検索で回答を生成しました")

                            # Display response (only if not in compare mode)
                            if not compare_mode:
                                st.write(response_text)

                                if not result_df.empty:
                                    st.success("✅ 検索完了")

                                    # Show search method used
                                    if search_mode == "RAG検索":
                                        st.info(f"🚀 RAG検索結果 (top_k={top_k if 'top_k' in locals() else 15}, semantic_weight={semantic_weight if 'semantic_weight' in locals() else 0.6})")
                                    elif search_mode == "LLM検索":
                                        st.info("🤖 LLM検索結果")
                                    else:
                                        st.info("🔍 自動選択検索結果")

                                    st.dataframe(result_df, use_container_width=True)

                                # Add to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response_text,
                                    "dataframe": result_df
                                })
                            else:
                                # More specific error message with debugging info
                                error_msg = "該当するデータが見つかりませんでした。\n\n"

                                error_msg += "\n💡 以下をお試しください:\n"
                                error_msg += "• より広い条件で検索する\n"
                                error_msg += "• 期間を変更する\n"
                                error_msg += "• 異なる項目で集計する\n"
                                error_msg += "• より一般的なキーワードを使用する"

                                st.write(error_msg)
                                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

                        except Exception as e:
                            error_msg = f"エラーが発生しました: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            except Exception as e:
                # Handle outer exception for chat history operations
                st.error(f"チャット処理エラー: {str(e)}")
                if "chat_history" in st.session_state:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"処理中にエラーが発生しました: {str(e)}"
                    })

# ---------- Main Application ----------

def main():
    """Main application function"""

    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title(APP_TITLE)

    # Sidebar
    st.sidebar.header("⚙️ 設定")

    # File selection
    selected_files = st.sidebar.multiselect(
        "📁 読み込むファイル",
        DEFAULT_FILES,
        default=[DEFAULT_FILES[0]],  # デフォルトは1ファイルのみ
        help="分析対象のExcelファイルを選択（データサイズを考慮して、デフォルトは1ファイルのみ選択）"
    )

    if not selected_files:
        st.warning("ファイルを選択してください")
        return

    # Load data
    with st.spinner("データを読み込み中..."):
        df_raw = load_excel(selected_files)

        if df_raw.empty:
            st.error("データの読み込みに失敗しました")
            return

        df_normalized = normalize_df(df_raw)
        df_cases, df_parts = split_fact_tables(df_normalized)

    st.sidebar.success(f"データ読み込み完了: {len(df_raw):,}行")

    # Filters
    st.sidebar.subheader("🔍 フィルタ")

    filters = {}

    # Date range filter
    if '基準日' in df_normalized.columns:
        min_date = df_normalized['基準日'].min()
        max_date = df_normalized['基準日'].max()

        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "📅 期間",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
                help="分析対象期間を選択"
            )

            if len(date_range) == 2:
                filters['date_range'] = date_range

    # Categorical filters
    categorical_fields = {
        '機種名': '🚜',
        'シリーズ': '📋',
        '種別': '🏷️',
        '部門名': '🏢',
        '整備区分': '🔧'
    }

    for field, icon in categorical_fields.items():
        if field in df_normalized.columns:
            unique_values = df_normalized[field].dropna().unique()
            if len(unique_values) > 0:
                selected_values = st.sidebar.multiselect(
                    f"{icon} {field}",
                    unique_values,
                    help=f"{field}でフィルタ"
                )
                if selected_values:
                    filters[field] = selected_values

    # Hour meter filter
    if 'アワ帯' in df_normalized.columns:
        hour_bins = df_normalized['アワ帯'].unique()
        selected_hour_bins = st.sidebar.multiselect(
            "⏱️ アワ帯",
            hour_bins,
            help="アワメータ帯域でフィルタ"
        )
        if selected_hour_bins:
            filters['hour_bins'] = selected_hour_bins

    # Options
    st.sidebar.subheader("⚙️ オプション")

    # Data sampling option for performance
    use_sampling = st.sidebar.checkbox(
        "高速表示モード（サンプリング）",
        value=True,
        help="大量データの場合、サンプリングして高速表示します"
    )

    if use_sampling:
        sample_size = st.sidebar.slider(
            "サンプルサイズ",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="表示するデータ行数（多いほど正確、少ないほど高速）"
        )
    else:
        sample_size = None

    filters['use_sampling'] = use_sampling
    filters['sample_size'] = sample_size

    filters['exclude_inspections'] = st.sidebar.checkbox(
        "点検整備を除外",
        help="点検・検査項目を集計から除外"
    )

    filters['aggregation_unit'] = st.sidebar.radio(
        "集計単位",
        ["cases", "parts"],
        format_func=lambda x: "案件単位" if x == "cases" else "部品明細単位",
        help="データの集計粒度を選択"
    )

    # Apply filters
    df_cases_filtered = apply_filters(df_cases, filters)
    df_parts_filtered = apply_filters(df_parts, filters)

    st.sidebar.write(f"📊 フィルタ後件数: {len(df_cases_filtered):,}案件")

    # Show sampling info
    if filters.get('use_sampling', False) and filters.get('sample_size'):
        original_count = len(df_cases)
        if len(df_cases_filtered) < original_count:
            st.sidebar.info(f"⚡ 高速表示モード: {filters['sample_size']:,}行でサンプリング表示中")

    # Download button
    if not df_cases_filtered.empty:
        csv_data = df_cases_filtered.to_csv(index=False, encoding='utf-8-sig')
        st.sidebar.download_button(
            "📥 CSVダウンロード",
            csv_data,
            "kubota_filtered_data.csv",
            "text/csv",
            help="フィルタ後のデータをCSV形式でダウンロード"
        )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 ダッシュボード",
        "📋 詳細テーブル",
        "💬 QAチャット",
        "🔍 データ品質"
    ])

    with tab1:
        render_dashboard(df_cases_filtered, df_parts_filtered, filters)
        render_additional_kpis(df_cases_filtered, df_parts_filtered)

    with tab2:
        active_df = df_parts_filtered if filters.get('aggregation_unit') == 'parts' else df_cases_filtered
        render_table(active_df)

    with tab3:
        qa_chat(df_cases_filtered, df_parts_filtered)

    with tab4:
        render_quality_report(df_normalized)

if __name__ == "__main__":
    main()