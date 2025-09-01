#!/usr/bin/env python3
"""
Comprehensive database diagnostic tool for COGS RAG system - FIXED VERSION
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional
import argparse
import sys
from pathlib import Path

def check_table_schema(db_path: str, table_name: str) -> Dict[str, Any]:
    """Check the schema of a specific table"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Get table info
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = cur.fetchall()

        # Get sample data
        cur.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample_data = cur.fetchall()

        # Get total count
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_count = cur.fetchone()[0]

        conn.close()

        return {
            "exists": True,
            "columns": columns,
            "sample_data": sample_data,
            "total_count": total_count
        }
    except sqlite3.OperationalError as e:
        return {
            "exists": False,
            "error": str(e)
        }

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

    # Title column
    if 'title' in columns:
        schema['title'] = 'title'

    conn.close()
    return schema

def analyze_embeddings(db_path: str) -> Dict[str, Any]:
    """Analyze embedding storage and quality"""
    schema = get_database_schema(db_path)
    columns = list(schema.keys())

    print(f"📋 Detected schema: {schema}")

    if 'embedding' not in schema:
        return {"error": "No embedding column found", "available_columns": columns}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    id_col = schema['id']
    embedding_col = schema['embedding']

    print(f"🧮 Using columns - ID: {id_col}, Embedding: {embedding_col}")

    # Check embedding formats - get a sample
    cur.execute(f"SELECT {id_col}, {embedding_col} FROM documents WHERE {embedding_col} IS NOT NULL AND {embedding_col} != '' LIMIT 10")
    embedding_samples = cur.fetchall()

    embedding_analysis = {
        "embedding_column": embedding_col,
        "id_column": id_col,
        "total_with_embeddings": 0,
        "blob_embeddings": 0,
        "json_embeddings": 0,
        "null_embeddings": 0,
        "empty_embeddings": 0,
        "embedding_dimensions": set(),
        "sample_formats": [],
        "sample_values": []
    }

    # Count all embeddings
    cur.execute(f"SELECT COUNT(*) FROM documents WHERE {embedding_col} IS NOT NULL AND {embedding_col} != ''")
    embedding_analysis["total_with_embeddings"] = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(*) FROM documents WHERE {embedding_col} IS NULL")
    embedding_analysis["null_embeddings"] = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(*) FROM documents WHERE {embedding_col} = ''")
    embedding_analysis["empty_embeddings"] = cur.fetchone()[0]

    # Analyze sample embeddings
    print(f"🔍 Analyzing {len(embedding_samples)} sample embeddings...")

    for doc_id, embedding_data in embedding_samples:
        if embedding_data is None or embedding_data == '':
            continue

        # Store first few samples for inspection
        if len(embedding_analysis["sample_values"]) < 3:
            sample_preview = str(embedding_data)[:100] + ("..." if len(str(embedding_data)) > 100 else "")
            embedding_analysis["sample_values"].append(f"ID {doc_id}: {sample_preview}")

        # Try to determine format
        if isinstance(embedding_data, bytes):
            embedding_analysis["blob_embeddings"] += 1
            try:
                # Try to decode as numpy array
                arr = np.frombuffer(embedding_data, dtype=np.float32)
                embedding_analysis["embedding_dimensions"].add(len(arr))
                embedding_analysis["sample_formats"].append(f"BLOB - dim: {len(arr)}")
            except:
                embedding_analysis["sample_formats"].append("BLOB - unknown format")
        else:
            # Try as JSON string
            try:
                parsed = json.loads(embedding_data)
                if isinstance(parsed, list):
                    embedding_analysis["json_embeddings"] += 1
                    embedding_analysis["embedding_dimensions"].add(len(parsed))
                    embedding_analysis["sample_formats"].append(f"JSON - dim: {len(parsed)}")
                else:
                    embedding_analysis["sample_formats"].append("JSON - not a list")
            except json.JSONDecodeError:
                # Maybe it's a string representation or something else
                embedding_analysis["sample_formats"].append(f"String format (len: {len(str(embedding_data))})")

    conn.close()
    return embedding_analysis

def search_content_patterns(db_path: str, patterns: List[str]) -> Dict[str, List[Dict]]:
    """Search for specific content patterns in the database"""
    schema = get_database_schema(db_path)

    if 'content' not in schema:
        return {"error": "No content column found"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    id_col = schema['id']
    content_col = schema['content']
    source_col = schema.get('source')
    title_col = schema.get('title')

    print(f"🔍 Using columns for search - ID: {id_col}, Content: {content_col}, Source: {source_col}, Title: {title_col}")

    results = {}

    # Build SELECT clause
    select_parts = [id_col]
    if source_col:
        select_parts.append(source_col)
    if title_col:
        select_parts.append(title_col)
    select_parts.append(f"SUBSTR({content_col}, 1, 200) as preview")

    select_clause = ", ".join(select_parts)

    for pattern in patterns:
        matches = []

        # Case-insensitive search in content
        query = f"""
            SELECT {select_clause}
            FROM documents
            WHERE LOWER({content_col}) LIKE LOWER(?)
            LIMIT 5
        """

        try:
            cur.execute(query, (f'%{pattern}%',))

            for row in cur.fetchall():
                match_data = {id_col: row[0]}

                idx = 1
                if source_col:
                    match_data["source"] = row[idx]
                    idx += 1

                if title_col:
                    match_data["title"] = row[idx]
                    idx += 1

                match_data["preview"] = row[idx]
                matches.append(match_data)

        except Exception as e:
            print(f"❌ Error searching for pattern '{pattern}': {e}")
            continue

        results[pattern] = matches

    conn.close()
    return results

def check_database_integrity(db_path: str) -> Dict[str, Any]:
    """Check overall database integrity and structure"""
    if not Path(db_path).exists():
        return {"error": f"Database file {db_path} does not exist"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]

    # Check database size
    cur.execute("PRAGMA page_count")
    page_count = cur.fetchone()[0]
    cur.execute("PRAGMA page_size")
    page_size = cur.fetchone()[0]
    db_size_mb = (page_count * page_size) / (1024 * 1024)

    conn.close()

    return {
        "exists": True,
        "tables": tables,
        "size_mb": round(db_size_mb, 2),
        "page_count": page_count
    }

def main():
    parser = argparse.ArgumentParser(description="Diagnose COGS database issues")
    parser.add_argument("--db", default="data/cogs_memory.db", help="Database path")
    parser.add_argument("--patterns", nargs="+",
                       default=["restaurant", "patio", "outdoor dining", "CS district"],
                       help="Content patterns to search for")

    args = parser.parse_args()

    print(f"🔍 Diagnosing database: {args.db}")
    print("=" * 60)

    # 1. Check database integrity
    print("\n📊 Database Integrity Check")
    integrity = check_database_integrity(args.db)
    if "error" in integrity:
        print(f"❌ {integrity['error']}")
        return

    print(f"✅ Database exists: {integrity['size_mb']} MB")
    print(f"📋 Tables found: {', '.join(integrity['tables'])}")

    # 2. Check documents table schema
    print("\n📋 Documents Table Analysis")
    if "documents" not in integrity['tables']:
        print("❌ No 'documents' table found!")
        print("Available tables:", integrity['tables'])
        return

    schema_info = check_table_schema(args.db, "documents")
    print(f"✅ Documents table has {schema_info['total_count']} records")
    print("Columns:", [f"{col[1]} ({col[2]})" for col in schema_info['columns']])

    # 3. Analyze embeddings
    print("\n🧮 Embedding Analysis")
    embedding_info = analyze_embeddings(args.db)
    if "error" in embedding_info:
        print(f"❌ {embedding_info['error']}")
        if "available_columns" in embedding_info:
            print(f"Available columns: {embedding_info['available_columns']}")
    else:
        print(f"📊 Total with embeddings: {embedding_info['total_with_embeddings']}")
        print(f"📊 Null embeddings: {embedding_info['null_embeddings']}")
        print(f"📊 Empty embeddings: {embedding_info.get('empty_embeddings', 0)}")
        print(f"📊 BLOB format: {embedding_info['blob_embeddings']}")
        print(f"📊 JSON format: {embedding_info['json_embeddings']}")
        print(f"📏 Dimensions found: {sorted(embedding_info['embedding_dimensions'])}")

        if embedding_info['sample_formats']:
            print("🔍 Sample formats:")
            for fmt in set(embedding_info['sample_formats'][:5]):
                print(f"  - {fmt}")

        if embedding_info['sample_values']:
            print("📄 Sample embedding values:")
            for sample in embedding_info['sample_values']:
                print(f"  - {sample}")

    # 4. Search for content patterns
    print(f"\n🔍 Content Pattern Search")
    pattern_results = search_content_patterns(args.db, args.patterns)

    if "error" in pattern_results:
        print(f"❌ {pattern_results['error']}")
    else:
        for pattern, matches in pattern_results.items():
            print(f"\n🎯 Pattern: '{pattern}' - {len(matches)} matches")
            for match in matches[:3]:  # Show top 3
                # Build display string based on available fields
                display_parts = []

                # Handle different ID column names
                for id_field in ['doc_id', 'id', 'rowid']:
                    if id_field in match:
                        display_parts.append(f"📄 {id_field}: {match[id_field]}")
                        break

                if 'source' in match:
                    display_parts.append(f"Source: {match['source']}")

                if 'title' in match and match['title']:
                    display_parts.append(f"Title: {match['title'][:30]}...")

                print(f"  {' | '.join(display_parts)}")
                print(f"     {match['preview'][:150]}...")

    # 5. Recommendations
    print(f"\n💡 Recommendations")

    if embedding_info.get('total_with_embeddings', 0) == 0:
        print("❗ No embeddings found - regenerate embeddings")
    elif embedding_info.get('json_embeddings', 0) == 0 and embedding_info.get('blob_embeddings', 0) == 0:
        print("❗ Embeddings exist but are in unrecognized format - check embedding format")
    elif len(embedding_info.get('embedding_dimensions', set())) == 0:
        print("❗ Could not parse embedding dimensions - embeddings may be corrupted")

    if not any(len(matches) > 0 for matches in pattern_results.values() if isinstance(pattern_results, dict) and "error" not in pattern_results):
        print("❗ No content matches found - check if data contains expected content")

    # Check if embeddings are actually usable
    total_embeddings = embedding_info.get('total_with_embeddings', 0)
    valid_embeddings = embedding_info.get('json_embeddings', 0) + embedding_info.get('blob_embeddings', 0)

    if total_embeddings > 0 and valid_embeddings == 0:
        print("⚠️  Embeddings exist but appear to be in unusable format")
        print("   Try: python embedding_fixer.py --verify-only")

if __name__ == "__main__":
    main()
