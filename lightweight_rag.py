# lightweight_rag.py
# Lightweight RAG implementation without heavy dependencies

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import re

# Simple text similarity using TF-IDF without heavy dependencies
from collections import Counter
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: str

class LightweightRAGSystem:
    """
    Lightweight RAG system using simple TF-IDF similarity and SQLite
    """

    def __init__(self, db_path: str = "./lightweight_rag.db"):
        """
        Initialize lightweight RAG system

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.documents = []
        self.tfidf_vectors = []
        self.vocabulary = {}

        # Initialize SQLite database
        self._init_database()

        # OpenAI client
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = OpenAI(api_key=api_key) if api_key else None
        except ImportError:
            self.openai_client = None
            logger.warning("OpenAI not available")

    def _init_database(self):
        """Initialize SQLite database and tables"""
        self.conn = sqlite3.connect(self.db_path)

        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_records (
                id INTEGER PRIMARY KEY,
                case_no TEXT,
                machine_model TEXT,
                work_content TEXT,
                malfunction_content TEXT,
                parts_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def load_maintenance_data(self, df: pd.DataFrame) -> None:
        """
        Load maintenance data and create searchable documents

        Args:
            df: DataFrame with maintenance data
        """
        try:
            # Clear existing data
            self.conn.execute("DELETE FROM documents")
            self.conn.execute("DELETE FROM maintenance_records")

            documents = []

            for idx, row in df.iterrows():
                # Create comprehensive text content for each record
                content_parts = []

                # Add structured information
                if pd.notna(row.get('機種名')):
                    content_parts.append(f"機種: {row['機種名']}")
                if pd.notna(row.get('作業内容')):
                    content_parts.append(f"作業内容: {row['作業内容']}")
                if pd.notna(row.get('不具合内容')):
                    content_parts.append(f"不具合内容: {row['不具合内容']}")
                if pd.notna(row.get('部品品名')):
                    content_parts.append(f"交換部品: {row['部品品名']}")

                content = " | ".join(content_parts)

                if content:
                    # Create metadata
                    metadata = {
                        'case_no': str(row.get('整備案件NO', '')),
                        'machine_model': str(row.get('機種名', '')),
                        'category': str(row.get('カテゴリ', '')),
                        'row_index': idx
                    }

                    documents.append({
                        'content': content,
                        'metadata': json.dumps(metadata)
                    })

                    # Insert into maintenance_records table
                    self.conn.execute("""
                        INSERT INTO maintenance_records
                        (case_no, machine_model, work_content, malfunction_content, parts_name)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        str(row.get('整備案件NO', '')),
                        str(row.get('機種名', '')),
                        str(row.get('作業内容', '')),
                        str(row.get('不具合内容', '')),
                        str(row.get('部品品名', ''))
                    ))

            # Insert documents
            for doc in documents:
                self.conn.execute("""
                    INSERT INTO documents (content, metadata) VALUES (?, ?)
                """, (doc['content'], doc['metadata']))

            self.conn.commit()

            # Build TF-IDF index
            self.documents = [doc['content'] for doc in documents]
            self._build_tfidf_index()

            logger.info(f"Loaded {len(documents)} maintenance records")

        except Exception as e:
            logger.error(f"Failed to load maintenance data: {e}")
            raise

    def _build_tfidf_index(self):
        """Build simple TF-IDF index"""
        if not self.documents:
            return

        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]

        # Build vocabulary
        all_words = set()
        for doc in tokenized_docs:
            all_words.update(doc)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}

        # Calculate TF-IDF vectors
        self.tfidf_vectors = []
        for doc in tokenized_docs:
            vector = self._calculate_tfidf(doc, tokenized_docs)
            self.tfidf_vectors.append(vector)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split by non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def _calculate_tfidf(self, doc_tokens: List[str], all_docs: List[List[str]]) -> Dict[int, float]:
        """Calculate TF-IDF vector for a document"""
        vector = {}
        doc_length = len(doc_tokens)

        # Calculate term frequencies
        tf = Counter(doc_tokens)

        for word, count in tf.items():
            if word in self.vocabulary:
                word_idx = self.vocabulary[word]

                # Term frequency
                tf_score = count / doc_length

                # Document frequency
                df = sum(1 for doc in all_docs if word in doc)

                # Inverse document frequency
                idf = math.log(len(all_docs) / (df + 1))

                # TF-IDF score
                vector[word_idx] = tf_score * idf

        return vector

    def _cosine_similarity(self, vec1: Dict[int, float], vec2: Dict[int, float]) -> float:
        """Calculate cosine similarity between two sparse vectors"""
        # Get common keys
        common_keys = set(vec1.keys()) & set(vec2.keys())

        if not common_keys:
            return 0.0

        # Calculate dot product
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)

        # Calculate magnitudes
        mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v**2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform semantic search using TF-IDF similarity"""
        if not self.documents or not self.tfidf_vectors:
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)

            # Calculate query TF-IDF vector
            query_vector = {}
            tf = Counter(query_tokens)

            for word, count in tf.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    tf_score = count / len(query_tokens)

                    # Use average IDF from training corpus
                    idf = 1.0  # Simplified
                    query_vector[word_idx] = tf_score * idf

            # Calculate similarities
            similarities = []
            for i, doc_vector in enumerate(self.tfidf_vectors):
                similarity = self._cosine_similarity(query_vector, doc_vector)
                similarities.append((i, similarity))

            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Create search results
            results = []
            for i, score in similarities[:top_k]:
                if score > 0:  # Only include non-zero similarities
                    # Get metadata from database
                    cursor = self.conn.execute(
                        "SELECT metadata FROM documents WHERE rowid = ?", (i + 1,)
                    )
                    metadata_row = cursor.fetchone()
                    metadata = json.loads(metadata_row[0]) if metadata_row else {}

                    result = SearchResult(
                        content=self.documents[i],
                        score=score,
                        metadata=metadata,
                        document_id=f"doc_{i}"
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def structured_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform structured search using SQLite LIKE queries"""
        try:
            # Simple keyword search across maintenance records
            cursor = self.conn.execute("""
                SELECT rowid, case_no, machine_model, work_content, malfunction_content, parts_name
                FROM maintenance_records
                WHERE work_content LIKE ? OR malfunction_content LIKE ? OR parts_name LIKE ?
                OR machine_model LIKE ?
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', top_k))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                rowid, case_no, machine_model, work_content, malfunction_content, parts_name = row

                # Create content string
                content_parts = []
                if machine_model:
                    content_parts.append(f"機種: {machine_model}")
                if work_content:
                    content_parts.append(f"作業内容: {work_content}")
                if malfunction_content:
                    content_parts.append(f"不具合: {malfunction_content}")
                if parts_name:
                    content_parts.append(f"部品: {parts_name}")

                content = " | ".join(content_parts)

                metadata = {
                    'case_no': case_no,
                    'machine_model': machine_model,
                    'row_id': rowid
                }

                result = SearchResult(
                    content=content,
                    score=0.8,  # Fixed score for structured search
                    metadata=metadata,
                    document_id=f"struct_{rowid}"
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Structured search failed: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Combine semantic and structured search"""
        # Get results from both methods
        semantic_results = self.semantic_search(query, top_k // 2)
        structured_results = self.structured_search(query, top_k // 2)

        # Combine and deduplicate
        all_results = semantic_results + structured_results

        # Sort by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate response using search results"""
        if not search_results:
            return "関連する情報が見つかりませんでした。"

        if self.openai_client:
            try:
                # Prepare context
                context_parts = []
                for result in search_results[:5]:  # Use top 5 results
                    context_parts.append(f"関連情報: {result.content}")

                context = "\n".join(context_parts)

                # Generate response
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Use cheaper model
                    messages=[
                        {"role": "system", "content": "あなたは建機の修理データ分析専門のAIアシスタントです。提供された情報を基に、簡潔で実用的な回答を提供してください。"},
                        {"role": "user", "content": f"質問: {query}\n\n関連情報:\n{context}\n\n上記の情報を基に回答してください。"}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.error(f"OpenAI response generation failed: {e}")

        # Fallback: Simple template response
        return f"'{query}'に関連する情報を{len(search_results)}件見つけました。詳細は検索結果をご確認ください。"

    def search_and_generate(self, query: str, top_k: int = 10) -> Tuple[str, List[SearchResult]]:
        """Complete search and response generation pipeline"""
        search_results = self.hybrid_search(query, top_k)
        response = self.generate_response(query, search_results)
        return response, search_results

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'conn'):
            self.conn.close()


def create_lightweight_rag_system(maintenance_data: pd.DataFrame) -> Optional[LightweightRAGSystem]:
    """
    Create and initialize lightweight RAG system

    Args:
        maintenance_data: DataFrame with maintenance records

    Returns:
        Initialized LightweightRAGSystem or None if failed
    """
    try:
        rag_system = LightweightRAGSystem()
        rag_system.load_maintenance_data(maintenance_data)
        return rag_system
    except Exception as e:
        logger.error(f"Failed to create lightweight RAG system: {e}")
        return None