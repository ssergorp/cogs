import os
import argparse
import logging
import asyncio
import numpy as np
import json
import sqlite3
import json
from typing import List, Dict, Any, Optional
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import UserMessage

load_dotenv()
LOG = logging.getLogger(__name__)

def save_chunk_to_memory(memory: "MemorySystem", doc_id: str, chunk: dict, embedding: list):
    """Save a chunk with embedding to the memory system, enforcing JSON format."""
    try:
        # Ensure embedding is a clean list of floats
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        embedding = [float(x) for x in embedding]
        
        # Always JSON-encode text (avoid mixing raw/plain strings and JSON)
        text_json = json.dumps(chunk.get("text", ""))
        
        memory.save_document(
            doc_id=doc_id,
            section_id=chunk.get("section_id"),
            title=chunk.get("title", ""),
            text=text_json,
            refs=[],  # Add empty refs list
            embedding=embedding,
        )
    except Exception as e:
        LOG.error(f"Error saving chunk {doc_id}-{chunk.get('section_id')}: {e}")
        raise

# ---------------------------
# Config & Data Classes
# ---------------------------

@dataclass
class SearchResult:
    score: float
    doc_id: str
    section_id: str
    title: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class RAGConfig:
    mistral_api_key: str
    embed_model: str = "mistral-embed"
    chat_model: str = "mistral-large-2411"
    max_context_chars: int = 6000
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    keyword_boost: float = 1.2
    similarity_threshold: float = 0.1

# Load config
config = RAGConfig(
    mistral_api_key=os.environ.get("MISTRAL_API_KEY")
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if not config.mistral_api_key:
    raise RuntimeError("MISTRAL_API_KEY not set in environment")

class MemorySystem:
    def __init__(self, db_path: str = "data/cogs_memory.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._init_tables()

    def _init_tables(self):
        c = self._conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT,
            section_id TEXT,
            title TEXT,
            text TEXT,
            refs TEXT,
            embedding TEXT,
            UNIQUE(doc_id, section_id)
        )
        """)
        self._conn.commit()

    def clear_document(self, doc_id: str):
        """Remove all chunks for a specific document"""
        c = self._conn.cursor()
        c.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        print(f"Cleared {c.rowcount} existing chunks for doc_id: {doc_id}")

    def save_document(
        self,
        doc_id: str,
        section_id: str,
        title: str,
        text: str,
        refs: List[str],
        embedding: List[float],
    ):
        c = self._conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO documents (doc_id, section_id, title, text, refs, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, section_id, title, text, json.dumps(refs), json.dumps(embedding)))
        self._conn.commit()

    def get_all_documents(self):
        cur = self._conn.cursor()
        cur.execute("SELECT doc_id, section_id, title, text, embedding FROM documents")
        rows = cur.fetchall()
        results = []
        for r in rows:
            try:
                text_val = json.loads(r[3])  # works if it's valid JSON
            except Exception:
                text_val = r[3]  # fallback: treat as plain string
            try:
                emb_val = json.loads(r[4]) if r[4] else None
            except Exception:
                emb_val = r[4]
            results.append({
                "doc_id": r[0],
                "section_id": r[1],
                "title": r[2],
                "text": text_val,
                "embedding": emb_val
            })
        return results

# ---------------------------
# Enhanced Embedding & Similarity
# ---------------------------

class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    def __init__(self):
        self._cache = {}

    def get(self, text: str) -> Optional[List[float]]:
        return self._cache.get(hash(text))

    def set(self, text: str, embedding: List[float]):
        self._cache[hash(text)] = embedding

embedding_cache = EmbeddingCache()

async def mistral_embed(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Async embeddings with caching and batching."""
    import httpx
    
    # Check cache first
    results = []
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        cached = embedding_cache.get(text)
        if cached:
            results.append((i, cached))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # Batch process uncached texts
    headers = {
        "Authorization": f"Bearer {config.mistral_api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start:batch_start + batch_size]
            batch_indices = uncached_indices[batch_start:batch_start + batch_size]
            
            payload = {"model": config.embed_model, "input": batch}
            
            try:
                resp = await client.post(
                    "https://api.mistral.ai/v1/embeddings",
                    headers=headers,
                    json=payload
                )
                resp.raise_for_status()
                embeddings = [d["embedding"] for d in resp.json()["data"]]
                
                # Cache and collect results
                for idx, embedding, original_idx in zip(range(len(batch)), embeddings, batch_indices):
                    embedding_cache.set(batch[idx], embedding)
                    results.append((original_idx, embedding))
            except Exception as e:
                logging.error(f"Embedding batch failed: {e}")
                # Return zero vectors for failed embeddings
                for original_idx in batch_indices:
                    results.append((original_idx, [0.0] * 1024))  # Assuming 1024-dim embeddings
    
    # Sort by original order
    results.sort(key=lambda x: x[0])
    return [emb for _, emb in results]

def enhanced_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Cosine similarity with error handling."""
    try:
        v1, v2 = np.array(vec1, dtype=float), np.array(vec2, dtype=float)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        return np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)
    except Exception as e:
        logging.warning(f"Similarity calculation failed: {e}")
        return 0.0

def extract_keywords(text: str) -> List[str]:
    """Simple keyword extraction (can be enhanced with NLP libraries)."""
    import re
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if len(w) > 2 and w not in stop_words]

def calculate_keyword_score(query_terms: List[str], text: str) -> float:
    """Enhanced keyword matching with TF-IDF-like scoring."""
    text_lower = text.lower()
    text_terms = extract_keywords(text)
    
    if not text_terms:
        return 0.0
    
    # Calculate term frequency and coverage
    matches = 0
    total_tf = 0
    for term in query_terms:
        term_count = text_lower.count(term)
        if term_count > 0:
            matches += 1
            # Diminishing returns for repeated terms
            total_tf += np.log(1 + term_count)
    
    if matches == 0:
        return 0.0
    
    # Combine coverage ratio with term frequency
    coverage = matches / len(query_terms)
    tf_score = total_tf / len(text_terms)
    
    return coverage * 0.7 + tf_score * 0.3

# ---------------------------
# Enhanced Search
# ---------------------------

async def enhanced_search_chunks(
    memory: MemorySystem,
    query: str,
    k: int = 5,
    include_metadata: bool = True
) -> List[SearchResult]:
    """Enhanced hybrid retrieval with better scoring."""
    # Get embeddings
    q_embed = (await mistral_embed([query]))[0]
    all_chunks = memory.get_all_documents()
    
    if not all_chunks:
        return []
    
    # Prepare embeddings
    chunk_embeddings = []
    valid_chunks = []
    
    for chunk in all_chunks:
        emb = chunk.get("embedding")
        if emb is None:
            continue
            
        # Handle different embedding formats
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except json.JSONDecodeError:
                continue
                
        if isinstance(emb, list):
            emb = np.array(emb, dtype=float)
            
        if len(emb) != len(q_embed):
            continue
            
        chunk_embeddings.append(emb)
        valid_chunks.append(chunk)
    
    if not valid_chunks:
        logging.warning("No valid chunks with embeddings found")
        return []
    
    # Vector similarity scoring
    vector_scores = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        sim = enhanced_cosine_similarity(q_embed, chunk_emb)
        if sim >= config.similarity_threshold:  # Filter low-similarity chunks
            vector_scores.append((i, sim))
    
    # Keyword scoring
    query_terms = extract_keywords(query)
    keyword_scores = {}
    
    for i, chunk in enumerate(valid_chunks):
        text = chunk.get("text", "")
        title = chunk.get("title", "")
        combined_text = f"{title} {text}"
        keyword_score = calculate_keyword_score(query_terms, combined_text)
        if keyword_score > 0:
            keyword_scores[i] = keyword_score
    
    # Combine scores with configurable weights
    final_scores = []
    
    # Process vector scores
    for i, v_score in vector_scores:
        k_score = keyword_scores.get(i, 0)
        # Apply keyword boost if there's a keyword match
        if k_score > 0:
            k_score *= config.keyword_boost
        
        combined_score = (
            config.vector_weight * v_score +
            config.keyword_weight * k_score
        )
        final_scores.append((combined_score, i))
    
    # Add keyword-only matches that weren't captured by vector search
    for i, k_score in keyword_scores.items():
        if not any(idx == i for _, idx in vector_scores):
            boosted_score = config.keyword_weight * k_score * config.keyword_boost
            final_scores.append((boosted_score, i))
    
    # Sort and deduplicate
    final_scores.sort(reverse=True, key=lambda x: x[0])
    seen = set()
    results = []
    
    for score, idx in final_scores:
        if idx >= len(valid_chunks):
            continue
            
        chunk = valid_chunks[idx]
        chunk_id = (chunk["doc_id"], chunk["section_id"])
        
        if chunk_id not in seen:
            result = SearchResult(
                score=score,
                doc_id=chunk["doc_id"],
                section_id=chunk["section_id"],
                title=chunk.get("title", ""),
                text=chunk.get("text", ""),
                metadata=chunk.get("metadata", {}) if include_metadata else {}
            )
            results.append(result)
            seen.add(chunk_id)
        
        if len(results) >= k:
            break
    
    return results

def smart_trim_context(results: List[SearchResult], max_chars: int) -> str:
    """Intelligent context trimming with priority preservation."""
    if not results:
        return ""
    
    # Prioritize higher-scored chunks
    context_parts = []
    total_len = 0
    
    for result in results:
        # Format with score and metadata if available
        snippet = f"[Score: {result.score:.3f}] {result.title}:\n{result.text}\n"
        if result.metadata:
            snippet += f"Metadata: {json.dumps(result.metadata, indent=2)}\n"
        snippet += "\n"
        
        if total_len + len(snippet) > max_chars:
            remaining = max_chars - total_len
            if remaining > 100:  # Only add if we have meaningful space left
                # Try to cut at sentence boundary
                truncated = snippet[:remaining]
                last_period = truncated.rfind('.')
                if last_period > remaining * 0.8:  # If we can preserve most of the text
                    truncated = truncated[:last_period + 1]
                context_parts.append(truncated + "\n[TRUNCATED]")
            break
        
        context_parts.append(snippet)
        total_len += len(snippet)
    
    return "".join(context_parts)

async def generate_enhanced_answer(query: str, context: str, stream: bool = True) -> str:
    """Enhanced answer generation with better prompting."""
    system_prompt = """You are an expert research assistant. Answer the question based strictly on the provided context.

Guidelines:
- Be precise and factual
- If the context doesn't contain enough information, say so
- Include relevant details and examples from the context
- Structure your answer clearly
- If there are conflicting information in the context, acknowledge it"""

    user_prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""

    client = Mistral(api_key=config.mistral_api_key)
    messages = [
        UserMessage(content=f"{system_prompt}\n\n{user_prompt}")
    ]

    if stream:
        response_stream = await client.chat.stream_async(
            model=config.chat_model,
            messages=messages,
            temperature=0.1,  # Lower temperature for more focused answers
            max_tokens=2000,
        )
        
        answer = ""
        async for chunk in response_stream:
            delta = chunk.data.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                answer += delta
        print("\n")
        return answer.strip()
    else:
        response = await client.chat.complete_async(
            model=config.chat_model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()

# ---------------------------
# Main Function
# ---------------------------

async def main():
    parser = argparse.ArgumentParser(description="Enhanced Async RAG QA System")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top chunks to retrieve")
    parser.add_argument("--db", default="data/cogs_memory.db", help="Database path")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--verbose", action="store_true", help="Show detailed scoring info")
    args = parser.parse_args()

    try:
        memory = MemorySystem(db_path=args.db)
        print(f"🔍 Searching for: '{args.query}'\n")

        results = await enhanced_search_chunks(memory, args.query, k=args.top_k)

        if not results:
            print("❌ No relevant documents found.")
            return

        # Build context
        context_text = smart_trim_context(results, max_chars=config.max_context_chars)

        print("📝 --- Answer ---\n")
        answer = await generate_enhanced_answer(
            args.query,
            context_text,
            stream=not args.no_stream
        )

        print(f"\n📚 --- Sources ({len(results)} found) ---\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. [Score: {result.score:.4f}] {result.doc_id} - {result.section_id}")
            if args.verbose and result.metadata:
                print(f"   Metadata: {json.dumps(result.metadata, indent=2)}")
            preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
            print(f"   Preview: {preview}\n")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
