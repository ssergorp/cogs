#!/usr/bin/env python3
"""
Comprehensive RAG database diagnostics to find why CS district info isn't surfacing.
"""

import sqlite3
import json
import numpy as np
from collections import defaultdict, Counter

def analyze_database_structure(db_path="data/cogs_memory.db"):
    """Analyze the basic structure of the database."""
    print("=== DATABASE STRUCTURE ANALYSIS ===")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get table schema
    cur.execute("PRAGMA table_info(documents)")
    columns = cur.fetchall()
    print("Table columns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Get total counts
    cur.execute("SELECT COUNT(*) FROM documents")
    total_docs = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
    embedded_docs = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT doc_id) FROM documents")
    unique_docs = cur.fetchone()[0]
    
    print(f"\nDocument counts:")
    print(f"  Total documents: {total_docs}")
    print(f"  With embeddings: {embedded_docs}")
    print(f"  Unique doc_ids: {unique_docs}")
    
    # Check for CS mentions
    cur.execute("SELECT COUNT(*) FROM documents WHERE LOWER(text) LIKE '%cs%'")
    cs_mentions = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM documents WHERE LOWER(text) LIKE '%cs district%'")
    cs_district_mentions = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM documents WHERE LOWER(text) LIKE '%commercial service%'")
    commercial_service_mentions = cur.fetchone()[0]
    
    print(f"\nCS-related mentions:")
    print(f"  Contains 'cs': {cs_mentions}")
    print(f"  Contains 'cs district': {cs_district_mentions}")
    print(f"  Contains 'commercial service': {commercial_service_mentions}")
    
    conn.close()

def analyze_document_distribution(db_path="data/cogs_memory.db"):
    """Analyze how documents are distributed across doc_ids and sections."""
    print("\n=== DOCUMENT DISTRIBUTION ANALYSIS ===")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Document ID distribution
    cur.execute("SELECT doc_id, COUNT(*) FROM documents GROUP BY doc_id ORDER BY COUNT(*) DESC LIMIT 10")
    doc_counts = cur.fetchall()
    
    print("Top document IDs by section count:")
    for doc_id, count in doc_counts:
        print(f"  {doc_id}: {count} sections")
    
    # Section ID patterns
    cur.execute("SELECT section_id FROM documents WHERE section_id IS NOT NULL")
    section_ids = [row[0] for row in cur.fetchall()]
    
    if section_ids:
        section_prefixes = defaultdict(int)
        for sid in section_ids:
            if isinstance(sid, str) and len(sid) > 3:
                prefix = sid[:3]
                section_prefixes[prefix] += 1
        
        print(f"\nSection ID patterns (top 10):")
        for prefix, count in Counter(section_prefixes).most_common(10):
            print(f"  {prefix}*: {count} sections")
    
    conn.close()

def find_cs_documents_detailed(db_path="data/cogs_memory.db"):
    """Find and analyze documents containing CS mentions in detail."""
    print("\n=== DETAILED CS DOCUMENT ANALYSIS ===")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all CS documents
    cur.execute("""
        SELECT doc_id, section_id, title, text, 
               CASE WHEN embedding IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embedding
        FROM documents 
        WHERE LOWER(text) LIKE '%cs%'
        ORDER BY doc_id, section_id
        LIMIT 20
    """)
    
    cs_docs = cur.fetchall()
    
    print(f"Found {len(cs_docs)} documents with CS mentions (showing first 20):")
    
    zoning_keywords = ['district', 'zone', 'zoning', 'permitted', 'allowed', 'use', 'commercial']
    signage_keywords = ['sign', 'signage', 'display', 'advertisement']
    
    for i, (doc_id, section_id, title, text, has_embedding) in enumerate(cs_docs):
        text_str = str(text).lower()
        
        # Count different types of content
        zoning_score = sum(1 for kw in zoning_keywords if kw in text_str)
        signage_score = sum(1 for kw in signage_keywords if kw in text_str)
        
        content_type = "ZONING" if zoning_score > signage_score else "SIGNAGE" if signage_score > 0 else "OTHER"
        
        print(f"\n{i+1}. {doc_id}-{section_id} [{content_type}] [Embedded: {has_embedding}]")
        print(f"    Title: {title}")
        print(f"    Zoning keywords: {zoning_score}, Signage keywords: {signage_score}")
        
        # Show CS context
        cs_pos = text_str.find(' cs ')
        if cs_pos >= 0:
            start = max(0, cs_pos - 50)
            end = min(len(text_str), cs_pos + 100)
            context = text_str[start:end].replace('\n', ' ')
            print(f"    CS context: ...{context}...")
        
        # Show first 150 chars of full text
        preview = text_str[:150].replace('\n', ' ')
        print(f"    Text preview: {preview}...")
    
    conn.close()

def test_embedding_quality(db_path="data/cogs_memory.db"):
    """Test if embeddings are actually working for CS-related content."""
    print("\n=== EMBEDDING QUALITY TEST ===")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get some CS documents with embeddings
    cur.execute("""
        SELECT doc_id, section_id, text, embedding
        FROM documents 
        WHERE LOWER(text) LIKE '%cs%' 
        AND embedding IS NOT NULL
        LIMIT 5
    """)
    
    cs_docs_with_embeddings = cur.fetchall()
    
    if not cs_docs_with_embeddings:
        print("❌ No CS documents have embeddings!")
        return
    
    print(f"Found {len(cs_docs_with_embeddings)} CS documents with embeddings")
    
    for doc_id, section_id, text, embedding_raw in cs_docs_with_embeddings:
        # Parse embedding
        try:
            if isinstance(embedding_raw, (bytes, memoryview)):
                embedding = np.frombuffer(embedding_raw, dtype=np.float32)
            else:
                embedding = json.loads(embedding_raw) if isinstance(embedding_raw, str) else embedding_raw
                embedding = np.array(embedding)
            
            print(f"\n{doc_id}-{section_id}:")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
            print(f"  Text preview: {str(text)[:100]}...")
            
            # Check if embedding is all zeros or has other issues
            if np.all(embedding == 0):
                print("  ⚠️  WARNING: Embedding is all zeros!")
            elif np.std(embedding) < 0.001:
                print("  ⚠️  WARNING: Embedding has very low variance!")
            else:
                print("  ✅ Embedding looks normal")
                
        except Exception as e:
            print(f"  ❌ Error parsing embedding: {e}")
    
    conn.close()

def analyze_signage_dominance(db_path="data/cogs_memory.db"):
    """Understand why signage documents might be dominating results."""
    print("\n=== SIGNAGE DOMINANCE ANALYSIS ===")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Count documents by content type
    cur.execute("SELECT COUNT(*) FROM documents WHERE LOWER(text) LIKE '%sign%'")
    signage_docs = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM documents WHERE LOWER(text) LIKE '%zoning%' OR LOWER(text) LIKE '%district%'")
    zoning_docs = cur.fetchone()[0]
    
    print(f"Content distribution:")
    print(f"  Signage-related: {signage_docs}")
    print(f"  Zoning-related: {zoning_docs}")
    
    # Check if signage docs have better embeddings
    cur.execute("""
        SELECT COUNT(*) FROM documents 
        WHERE LOWER(text) LIKE '%sign%' AND embedding IS NOT NULL
    """)
    signage_embedded = cur.fetchone()[0]
    
    cur.execute("""
        SELECT COUNT(*) FROM documents 
        WHERE (LOWER(text) LIKE '%zoning%' OR LOWER(text) LIKE '%district%') 
        AND embedding IS NOT NULL
    """)
    zoning_embedded = cur.fetchone()[0]
    
    print(f"Embedding coverage:")
    print(f"  Signage docs with embeddings: {signage_embedded}/{signage_docs} ({100*signage_embedded/max(signage_docs,1):.1f}%)")
    print(f"  Zoning docs with embeddings: {zoning_embedded}/{zoning_docs} ({100*zoning_embedded/max(zoning_docs,1):.1f}%)")
    
    conn.close()

def main():
    db_path = "data/cogs_memory.db"
    
    print("🔍 Starting RAG Database Diagnostics...")
    print(f"Database: {db_path}")
    print("=" * 60)
    
    try:
        analyze_database_structure(db_path)
        analyze_document_distribution(db_path)
        find_cs_documents_detailed(db_path)
        test_embedding_quality(db_path)
        analyze_signage_dominance(db_path)
        
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATIONS:")
        print("1. Check if CS zoning documents are properly chunked/embedded")
        print("2. Verify embedding quality for zoning vs signage content")
        print("3. Consider boosting keyword matching for exact term matches")
        print("4. Check if document parsing is handling zoning sections correctly")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()
