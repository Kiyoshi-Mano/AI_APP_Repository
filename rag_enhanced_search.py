# rag_enhanced_search.py
# RAG-Enhanced Search System for KUBOTA Maintenance Data

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# RAG and Vector Search Dependencies
RAG_DEPENDENCIES_AVAILABLE = True
MISSING_DEPENDENCIES = []

try:
    import openai
except ImportError:
    MISSING_DEPENDENCIES.append("openai")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    MISSING_DEPENDENCIES.append("sentence-transformers")
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    MISSING_DEPENDENCIES.append("faiss-cpu")
    faiss = None

try:
    import duckdb
except ImportError:
    MISSING_DEPENDENCIES.append("duckdb")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    MISSING_DEPENDENCIES.append("langchain")
    RecursiveCharacterTextSplitter = None
    Document = None

if MISSING_DEPENDENCIES:
    RAG_DEPENDENCIES_AVAILABLE = False
    print(f"Missing RAG dependencies: {', '.join(MISSING_DEPENDENCIES)}")
    print("Install with: uv add " + " ".join(MISSING_DEPENDENCIES))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: str

class RAGEnhancedSearchSystem:
    """
    RAG-Enhanced Search System using DuckDB for structured queries and vector search for semantic similarity
    """

    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 duckdb_path: str = ":memory:",
                 vector_index_path: str = "./vector_index.faiss"):
        """
        Initialize RAG-Enhanced Search System

        Args:
            embedding_model: SentenceTransformer model for embeddings
            duckdb_path: Path to DuckDB database
            vector_index_path: Path to FAISS vector index
        """
        # Check if dependencies are available
        if not RAG_DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Missing required dependencies: {', '.join(MISSING_DEPENDENCIES)}")

        self.embedding_model_name = embedding_model
        self.duckdb_path = duckdb_path
        self.vector_index_path = vector_index_path

        # Initialize components
        self._init_embedding_model()
        self._init_duckdb()
        self._init_vector_index()

        # Document storage
        self.documents: List[Document] = []
        self.document_embeddings = None

        # OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _init_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _init_duckdb(self):
        """Initialize DuckDB connection and setup tables"""
        try:
            self.duckdb_conn = duckdb.connect(self.duckdb_path)

            # Create vector search extension if available
            try:
                self.duckdb_conn.execute("INSTALL vss;")
                self.duckdb_conn.execute("LOAD vss;")
                logger.info("DuckDB vector search extension loaded")
            except Exception as e:
                logger.warning(f"Vector search extension not available: {e}")

            # Create tables for structured data
            self._create_tables()
            logger.info("DuckDB connection initialized")

        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")
            raise

    def _init_vector_index(self):
        """Initialize FAISS vector index"""
        try:
            # Try to load existing index
            if os.path.exists(self.vector_index_path):
                self.vector_index = faiss.read_index(self.vector_index_path)
                logger.info(f"Loaded existing vector index: {self.vector_index_path}")
            else:
                # Create new index
                self.vector_index = faiss.IndexFlatIP(self.embedding_dimension)
                logger.info("Created new vector index")
        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            raise

    def _create_tables(self):
        """Create DuckDB tables for structured data"""

        # Main maintenance records table
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_records (
                id INTEGER PRIMARY KEY,
                case_no VARCHAR,
                machine_model VARCHAR,
                series VARCHAR,
                category VARCHAR,
                department VARCHAR,
                work_date DATE,
                work_content TEXT,
                malfunction_content TEXT,
                parts_name VARCHAR,
                parts_code VARCHAR,
                work_hours FLOAT,
                hour_meter FLOAT,
                bulletin_no VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Document embeddings metadata table
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                doc_id VARCHAR PRIMARY KEY,
                source_table VARCHAR,
                source_id INTEGER,
                content_type VARCHAR,
                chunk_index INTEGER,
                embedding_model VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        logger.info("Created DuckDB tables")

    def load_maintenance_data(self, df: pd.DataFrame) -> None:
        """
        Load maintenance data into DuckDB and create embeddings

        Args:
            df: DataFrame with maintenance data
        """
        try:
            # Clean and prepare data
            df_clean = df.copy()

            # Map columns to standardized schema
            column_mapping = {
                '整備案件NO': 'case_no',
                '機種名': 'machine_model',
                'シリーズ': 'series',
                'カテゴリ': 'category',
                '部門名': 'department',
                '作業終了日': 'work_date',
                '作業内容': 'work_content',
                '不具合内容': 'malfunction_content',
                '部品品名': 'parts_name',
                '部品品番': 'parts_code',
                '作業時間': 'work_hours',
                'アワメータ_HR': 'hour_meter',
                'ブルチンNO': 'bulletin_no'
            }

            # Rename columns if they exist
            df_clean = df_clean.rename(columns={k: v for k, v in column_mapping.items() if k in df_clean.columns})

            # Select relevant columns
            required_columns = ['case_no', 'machine_model', 'series', 'category', 'department',
                              'work_date', 'work_content', 'malfunction_content', 'parts_name',
                              'parts_code', 'work_hours', 'hour_meter', 'bulletin_no']

            # Add missing columns with None values
            for col in required_columns:
                if col not in df_clean.columns:
                    df_clean[col] = None

            df_final = df_clean[required_columns].copy()
            df_final.reset_index(drop=True, inplace=True)
            df_final['id'] = range(1, len(df_final) + 1)

            # Insert into DuckDB
            self.duckdb_conn.execute("DELETE FROM maintenance_records")
            self.duckdb_conn.register('temp_df', df_final)
            self.duckdb_conn.execute("""
                INSERT INTO maintenance_records
                SELECT * FROM temp_df
            """)

            # Create documents for vector search
            self._create_documents_from_data(df_final)

            logger.info(f"Loaded {len(df_final)} maintenance records")

        except Exception as e:
            logger.error(f"Failed to load maintenance data: {e}")
            raise

    def _create_documents_from_data(self, df: pd.DataFrame) -> None:
        """
        Create documents from maintenance data for vector search

        Args:
            df: DataFrame with maintenance data
        """
        documents = []

        for idx, row in df.iterrows():
            # Create comprehensive text content for each record
            content_parts = []

            # Add structured information
            if pd.notna(row['machine_model']):
                content_parts.append(f"機種: {row['machine_model']}")
            if pd.notna(row['series']):
                content_parts.append(f"シリーズ: {row['series']}")
            if pd.notna(row['category']):
                content_parts.append(f"カテゴリ: {row['category']}")
            if pd.notna(row['department']):
                content_parts.append(f"部門: {row['department']}")

            # Add work content and malfunction details
            if pd.notna(row['work_content']):
                content_parts.append(f"作業内容: {row['work_content']}")
            if pd.notna(row['malfunction_content']):
                content_parts.append(f"不具合内容: {row['malfunction_content']}")

            # Add parts information
            if pd.notna(row['parts_name']):
                content_parts.append(f"交換部品: {row['parts_name']}")
            if pd.notna(row['parts_code']):
                content_parts.append(f"部品番号: {row['parts_code']}")

            # Add technical details
            if pd.notna(row['hour_meter']):
                content_parts.append(f"稼働時間: {row['hour_meter']}時間")
            if pd.notna(row['work_hours']):
                content_parts.append(f"作業時間: {row['work_hours']}時間")
            if pd.notna(row['bulletin_no']):
                content_parts.append(f"ブルチン番号: {row['bulletin_no']}")

            content = " | ".join(content_parts)

            if content:  # Only add non-empty documents
                doc = Document(
                    page_content=content,
                    metadata={
                        'source_id': row['id'],
                        'case_no': row['case_no'],
                        'machine_model': row['machine_model'],
                        'category': row['category'],
                        'work_date': str(row['work_date']) if pd.notna(row['work_date']) else None,
                        'document_type': 'maintenance_record'
                    }
                )
                documents.append(doc)

        # Split documents if they're too long
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[" | ", "。", "、", " "]
        )

        self.documents = text_splitter.split_documents(documents)

        # Create embeddings
        self._create_embeddings()

        logger.info(f"Created {len(self.documents)} document chunks")

    def _create_embeddings(self) -> None:
        """Create embeddings for all documents and build vector index"""
        if not self.documents:
            logger.warning("No documents to create embeddings for")
            return

        try:
            # Extract text content
            texts = [doc.page_content for doc in self.documents]

            # Create embeddings in batches
            batch_size = 32
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts, normalize_embeddings=True)
                embeddings.extend(batch_embeddings)

            self.document_embeddings = np.array(embeddings).astype('float32')

            # Add to FAISS index
            self.vector_index.reset()  # Clear existing index
            self.vector_index.add(self.document_embeddings)

            # Save index
            faiss.write_index(self.vector_index, self.vector_index_path)

            logger.info(f"Created embeddings for {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

    def hybrid_search(self,
                     query: str,
                     top_k: int = 10,
                     structured_filters: Optional[Dict[str, Any]] = None,
                     semantic_weight: float = 0.7) -> List[SearchResult]:
        """
        Perform hybrid search combining structured query and semantic similarity

        Args:
            query: Search query
            top_k: Number of results to return
            structured_filters: DuckDB filters (e.g., {'machine_model': 'U-30', 'category': 'エンジン'})
            semantic_weight: Weight for semantic similarity (0-1, higher means more semantic)

        Returns:
            List of SearchResult objects
        """
        try:
            # Get structured search results
            structured_results = self._structured_search(query, structured_filters, top_k * 2)

            # Get semantic search results
            semantic_results = self._semantic_search(query, top_k * 2)

            # Combine and rank results
            combined_results = self._combine_results(
                structured_results,
                semantic_results,
                semantic_weight,
                top_k
            )

            return combined_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _structured_search(self,
                          query: str,
                          filters: Optional[Dict[str, Any]] = None,
                          limit: int = 20) -> List[SearchResult]:
        """Perform structured search using DuckDB"""
        try:
            # Build base query
            sql_parts = [
                "SELECT *, 1.0 as score FROM maintenance_records WHERE 1=1"
            ]
            params = {}

            # Add text search conditions
            if query:
                sql_parts.append("""
                    AND (
                        work_content ILIKE '%' || $query || '%' OR
                        malfunction_content ILIKE '%' || $query || '%' OR
                        parts_name ILIKE '%' || $query || '%' OR
                        machine_model ILIKE '%' || $query || '%' OR
                        category ILIKE '%' || $query || '%'
                    )
                """)
                params['query'] = query

            # Add structured filters
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        sql_parts.append(f"AND {key} = ${key}")
                        params[key] = value

            sql_parts.append(f"ORDER BY id DESC LIMIT {limit}")

            full_query = " ".join(sql_parts)

            # Execute query
            result = self.duckdb_conn.execute(full_query, params).fetchall()
            columns = [desc[0] for desc in self.duckdb_conn.description]

            # Convert to SearchResult objects
            search_results = []
            for row in result:
                row_dict = dict(zip(columns, row))

                # Create content string
                content_parts = []
                if row_dict.get('work_content'):
                    content_parts.append(f"作業内容: {row_dict['work_content']}")
                if row_dict.get('malfunction_content'):
                    content_parts.append(f"不具合: {row_dict['malfunction_content']}")
                if row_dict.get('parts_name'):
                    content_parts.append(f"部品: {row_dict['parts_name']}")

                content = " | ".join(content_parts)

                search_result = SearchResult(
                    content=content,
                    score=row_dict.get('score', 0.5),
                    metadata=row_dict,
                    document_id=f"db_{row_dict['id']}"
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"Structured search failed: {e}")
            return []

    def _semantic_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Perform semantic search using vector similarity"""
        try:
            if not self.documents or self.document_embeddings is None:
                logger.warning("No documents or embeddings available for semantic search")
                return []

            # Create query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)

            # Search in vector index
            scores, indices = self.vector_index.search(query_embedding, min(top_k, len(self.documents)))

            # Convert to SearchResult objects
            search_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    doc = self.documents[idx]

                    search_result = SearchResult(
                        content=doc.page_content,
                        score=float(score),
                        metadata=doc.metadata,
                        document_id=f"vec_{idx}"
                    )
                    search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _combine_results(self,
                        structured_results: List[SearchResult],
                        semantic_results: List[SearchResult],
                        semantic_weight: float,
                        top_k: int) -> List[SearchResult]:
        """Combine and rank structured and semantic search results"""
        try:
            # Create combined results dictionary
            combined = {}

            # Add structured results with adjusted scores
            for result in structured_results:
                key = result.document_id
                combined[key] = SearchResult(
                    content=result.content,
                    score=result.score * (1 - semantic_weight),
                    metadata=result.metadata,
                    document_id=result.document_id
                )

            # Add semantic results with adjusted scores
            for result in semantic_results:
                key = result.document_id
                if key in combined:
                    # Combine scores if document appears in both results
                    combined[key].score += result.score * semantic_weight
                else:
                    combined[key] = SearchResult(
                        content=result.content,
                        score=result.score * semantic_weight,
                        metadata=result.metadata,
                        document_id=result.document_id
                    )

            # Sort by combined score and return top_k
            sorted_results = sorted(combined.values(), key=lambda x: x.score, reverse=True)
            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"Failed to combine results: {e}")
            return []

    def generate_rag_response(self,
                            query: str,
                            search_results: List[SearchResult],
                            max_context_length: int = 4000) -> str:
        """
        Generate RAG response using retrieved context

        Args:
            query: User query
            search_results: Retrieved search results
            max_context_length: Maximum context length for LLM

        Returns:
            Generated response string
        """
        try:
            if not search_results:
                return "関連する情報が見つかりませんでした。"

            # Prepare context from search results
            context_parts = []
            current_length = 0

            for result in search_results:
                result_text = f"【関連情報 - スコア: {result.score:.2f}】\n{result.content}\n"

                if current_length + len(result_text) > max_context_length:
                    break

                context_parts.append(result_text)
                current_length += len(result_text)

            context = "\n".join(context_parts)

            # Create prompt for LLM
            system_prompt = """あなたは建機の修理データ分析専門のAIアシスタントです。
            提供された関連情報を基に、ユーザーの質問に正確で実用的な回答を提供してください。

            回答のガイドライン:
            - 提供された情報に基づいて回答する
            - 数値や具体例を含める
            - 実用的な洞察や推奨事項を提供する
            - 情報が不足している場合は正直に伝える
            - 簡潔で分かりやすい日本語で回答する
            """

            user_prompt = f"""質問: {query}

            関連情報:
            {context}

            上記の関連情報を基に、質問に回答してください。"""

            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            return f"回答生成中にエラーが発生しました: {str(e)}"

    def search_and_generate(self,
                          query: str,
                          filters: Optional[Dict[str, Any]] = None,
                          top_k: int = 10,
                          semantic_weight: float = 0.7) -> Tuple[str, List[SearchResult]]:
        """
        Complete RAG pipeline: search and generate response

        Args:
            query: User query
            filters: Structured filters
            top_k: Number of results to retrieve
            semantic_weight: Weight for semantic similarity

        Returns:
            Tuple of (generated_response, search_results)
        """
        # Perform hybrid search
        search_results = self.hybrid_search(query, top_k, filters, semantic_weight)

        # Generate response
        response = self.generate_rag_response(query, search_results)

        return response, search_results

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'duckdb_conn'):
            self.duckdb_conn.close()


# Example usage and integration functions
def create_enhanced_search_system(maintenance_data: pd.DataFrame) -> Optional[RAGEnhancedSearchSystem]:
    """
    Create and initialize RAG-enhanced search system with maintenance data

    Args:
        maintenance_data: DataFrame with maintenance records

    Returns:
        Initialized RAGEnhancedSearchSystem or None if dependencies missing
    """
    if not RAG_DEPENDENCIES_AVAILABLE:
        logger.error(f"Cannot create RAG system. Missing dependencies: {', '.join(MISSING_DEPENDENCIES)}")
        return None

    try:
        search_system = RAGEnhancedSearchSystem(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            duckdb_path="./maintenance_data.duckdb",
            vector_index_path="./maintenance_vector_index.faiss"
        )

        search_system.load_maintenance_data(maintenance_data)
        return search_system
    except Exception as e:
        logger.error(f"Failed to create RAG search system: {e}")
        return None


def demo_search_queries():
    """Demonstrate various search query examples"""
    return [
        "オイル漏れの修理事例を教えて",
        "エンジン関連の不具合で多いものは？",
        "U-30シリーズの油圧トラブル",
        "ポンプ交換の作業時間はどのくらい？",
        "最近多い故障パターンは？",
        "高稼働時間での故障傾向",
        "部品交換コストが高い修理",
        "予防保全のポイント"
    ]