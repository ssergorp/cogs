#!/usr/bin/env python3
"""
Tool to regenerate embeddings for COGS database
"""

import sqlite3
import json
import numpy as np
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingClient:
    """Client for generating embeddings using OpenAI, Mistral, or local models"""

    def __init__(self, model_name: str = "mistral-embed", api_key: Optional[str] = None, provider: str = "mistral"):
        self.model_name = model_name
        self.api_key = api_key
        self.provider = provider.lower()

        if self.provider == "mistral":
            self.base_url = "https://api.mistral.ai/v1/embeddings"
            if not model_name.startswith("mistral"):
                self.model_name = "mistral-embed"  # Default Mistral embedding model
        elif self.provider == "openai":
            self.base_url = "https://api.openai.com/v1/embeddings"
            if not model_name.startswith("text-embedding"):
                self.model_name = "text-embedding-3-small"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def generate_embedding(self, text: str, session: aiohttp.ClientSession) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not self.api_key:
            # Fallback to dummy embeddings for testing
            logger.warning("No API key provided, generating dummy embeddings")
            return self._generate_dummy_embedding(text)

        try:
            if self.provider == "mistral":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": self.model_name,
                    "input": [text[:8000]],  # Mistral expects a list, truncate to avoid token limits
                    "encoding_format": "float"
                }

            elif self.provider == "openai":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": self.model_name,
                    "input": text[:8000],  # OpenAI expects string or list
                    "encoding_format": "float"
                }

            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["data"][0]["embedding"]
                else:
                    logger.error(f"API error {response.status}: {await response.text()}")
                    return None

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """Generate deterministic dummy embedding for testing"""
        # Use hash of text to create consistent dummy embedding
        hash_val = hash(text)
        np.random.seed(abs(hash_val) % (2**32))

        # Different dimension based on provider/model
        if self.provider == "mistral":
            dims = 1024  # Mistral embed dimension
        else:
            dims = 1536  # OpenAI text-embedding-3-small dimension

        return np.random.normal(0, 1, dims).tolist()

async def regenerate_embeddings(db_path: str,
                              embedding_client: EmbeddingClient,
                              batch_size: int = 10,
                              max_docs: Optional[int] = None) -> None:
    """Regenerate embeddings for documents in the database"""

    schema = get_database_schema(db_path)

    if 'embedding' not in schema:
        print("❌ No embedding column found!")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    id_col = schema['id']
    content_col = schema['content']
    embedding_col = schema['embedding']
    source_col = schema.get('source', id_col)

    print(f"🔍 Using schema: ID={id_col}, Content={content_col}, Embedding={embedding_col}")

    # Get documents without embeddings or with null embeddings
    query = f"""
        SELECT {id_col}, {source_col}, {content_col}
        FROM documents
        WHERE {embedding_col} IS NULL OR {embedding_col} = ''
    """

    if max_docs:
        query += f" LIMIT {max_docs}"

    cur.execute(query)
    documents = cur.fetchall()

    if not documents:
        print("✅ All documents already have embeddings!")
        conn.close()
        return

    print(f"🔄 Regenerating embeddings for {len(documents)} documents...")

    # Process in batches
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
            batch = documents[i:i + batch_size]

            # Generate embeddings for batch
            tasks = []
            for doc_id, source_id, content in batch:
                task = embedding_client.generate_embedding(content, session)
                tasks.append((doc_id, task))

            # Wait for all embeddings in batch
            batch_results = []
            for doc_id, task in tasks:
                embedding = await task
                if embedding:
                    batch_results.append((json.dumps(embedding), doc_id))

            # Update database with batch results
            if batch_results:
                cur.executemany(
                    f"UPDATE documents SET {embedding_col} = ? WHERE {id_col} = ?",
                    batch_results
                )
                conn.commit()

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)

    conn.close()
    print(f"✅ Completed embedding regeneration!")

def get_database_schema(db_path: str) -> Dict[str, str]:
    """Get the actual database schema"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get column info
    cur.execute("PRAGMA table_info(documents)")
    columns_info = cur.fetchall()
    columns = [col[1] for col in columns_info]

    # Find the right column names
    schema = {}

    # ID column
    for col_name in ['id', 'doc_id', 'document_id', 'rowid']:
        if col_name in columns:
            schema['id'] = col_name
            break
    if 'id' not in schema:
        schema['id'] = 'rowid'

    # Source column
    for col_name in ['source_id', 'source', 'section_id', 'doc_id']:
        if col_name in columns:
            schema['source'] = col_name
            break

    # Content column
    for col_name in ['content', 'text', 'chunk_text', 'document_text']:
        if col_name in columns:
            schema['content'] = col_name
            break

    # Embedding column
    for col_name in ['embedding', 'embeddings', 'vector', 'embedding_vector']:
        if col_name in columns:
            schema['embedding'] = col_name
            break

    conn.close()
    return schema

def check_existing_embeddings(db_path: str) -> Dict[str, int]:
    """Check the current state of embeddings in the database"""
    schema = get_database_schema(db_path)

    if 'embedding' not in schema:
        return {"error": "No embedding column found"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    embedding_col = schema['embedding']

    # Count different embedding states
    cur.execute(f"SELECT COUNT(*) FROM documents WHERE {embedding_col} IS NOT NULL AND {embedding_col} != ''")
    has_embeddings = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(*) FROM documents WHERE {embedding_col} IS NULL OR {embedding_col} = ''")
    no_embeddings = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM documents")
    total = cur.fetchone()[0]

    conn.close()

    return {
        "total": total,
        "has_embeddings": has_embeddings,
        "no_embeddings": no_embeddings,
        "embedding_column": embedding_col
    }

def verify_embedding_format(db_path: str, sample_size: int = 10) -> bool:
    """Verify that embeddings are in the correct format"""
    schema = get_database_schema(db_path)

    if 'embedding' not in schema:
        print("⚠️  No embedding column found")
        return False

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    embedding_col = schema['embedding']

    cur.execute(f"""
        SELECT {embedding_col}
        FROM documents
        WHERE {embedding_col} IS NOT NULL AND {embedding_col} != ''
        LIMIT ?
    """, (sample_size,))

    embeddings = cur.fetchall()
    conn.close()

    if not embeddings:
        print("⚠️  No embeddings found to verify")
        return False

    valid_count = 0
    for (embedding_data,) in embeddings:
        try:
            # Try to parse as JSON
            parsed = json.loads(embedding_data)
            if isinstance(parsed, list) and len(parsed) > 0:
                # Check if all elements are numbers
                if all(isinstance(x, (int, float)) for x in parsed[:10]):
                    valid_count += 1
        except:
            continue

    success_rate = valid_count / len(embeddings)
    print(f"📊 Embedding verification: {valid_count}/{len(embeddings)} valid ({success_rate:.1%})")

    return success_rate > 0.8

async def main():
    parser = argparse.ArgumentParser(description="Regenerate embeddings for COGS database")
    parser.add_argument("--db", default="data/cogs_memory.db", help="Database path")
    parser.add_argument("--api-key", help="API key (Mistral or OpenAI)")
    parser.add_argument("--provider", default="mistral", choices=["mistral", "openai"], help="API provider")
    parser.add_argument("--model", help="Embedding model name (auto-selected based on provider if not specified)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing (lower for Mistral)")
    parser.add_argument("--max-docs", type=int, help="Maximum documents to process (for testing)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing embeddings")

    args = parser.parse_args()

    # Auto-select model based on provider if not specified
    if not args.model:
        if args.provider == "mistral":
            model = "mistral-embed"
        else:
            model = "text-embedding-3-small"
    else:
        model = args.model

    if not Path(args.db).exists():
        print(f"❌ Database {args.db} not found!")
        sys.exit(1)

    print(f"🔍 Checking embeddings in: {args.db}")

    # Check current state
    embedding_stats = check_existing_embeddings(args.db)

    if "error" in embedding_stats:
        print(f"❌ {embedding_stats['error']}")
        return

    print(f"📊 Embedding Status:")
    print(f"  Total documents: {embedding_stats['total']}")
    print(f"  With embeddings: {embedding_stats['has_embeddings']}")
    print(f"  Without embeddings: {embedding_stats['no_embeddings']}")
    print(f"  Using column: {embedding_stats['embedding_column']}")

    # Verify existing embeddings
    if embedding_stats['has_embeddings'] > 0:
        verify_embedding_format(args.db)

    if args.verify_only:
        return

    if embedding_stats['no_embeddings'] == 0:
        print("✅ All documents already have embeddings!")
        return

    # Initialize embedding client
    embedding_client = EmbeddingClient(
        model_name=model,
        api_key=args.api_key,
        provider=args.provider
    )

    if not args.api_key:
        print(f"⚠️  No API key provided - will generate dummy embeddings for testing")
        print(f"    Provider: {args.provider}, Model: {model}")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"🚀 Using {args.provider} API with model: {model}")

    # Regenerate embeddings
    await regenerate_embeddings(
        db_path=args.db,
        embedding_client=embedding_client,
        batch_size=args.batch_size,
        max_docs=args.max_docs
    )

    # Final verification
    print("\n🔍 Final verification...")
    final_stats = check_existing_embeddings(args.db)
    print(f"✅ Embeddings regenerated: {final_stats['has_embeddings']}/{final_stats['total']} documents")

if __name__ == "__main__":
    asyncio.run(main())
