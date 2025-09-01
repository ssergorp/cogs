#!/usr/bin/env python3
"""
Emergency Embedding Regeneration Fix
====================================

This script fixes the catastrophic duplicate embedding issue where 99.98% of 
documents have identical embeddings, making semantic search impossible.

Usage:
    python fix_embeddings.py --db data/cogs_memory.db --batch-size 50 --test-mode
    python fix_embeddings.py --db data/cogs_memory.db --regenerate-all
"""

import sqlite3
import json
import time
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
import requests
import os
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingJob:
    doc_id: str
    section_id: str
    content: str
    title: str
    current_hash: str

class EmbeddingRegenerator:
    """Fix duplicate embeddings by regenerating them properly"""
    
    def __init__(self, db_path: str, api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable or api_key parameter required")
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # API configuration
        self.api_url = "https://api.mistral.ai/v1/embeddings"
        self.model = "mistral-embed"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Rate limiting
        self.requests_per_minute = 100  # Conservative limit
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = time.time()
        
    def analyze_duplicate_problem(self) -> Dict:
        """Analyze the extent of the duplicate embedding problem"""
        logger.info("🔍 Analyzing duplicate embedding problem...")
        
        cursor = self.conn.execute("""
            SELECT doc_id, section_id, title, text, embedding
            FROM documents 
            WHERE embedding IS NOT NULL
            ORDER BY doc_id, section_id
        """)
        
        embedding_hashes = defaultdict(list)
        total_docs = 0
        
        for row in cursor.fetchall():
            total_docs += 1
            embedding_data = row['embedding']
            
            # Create hash of embedding
            emb_hash = hashlib.md5(embedding_data.encode()).hexdigest()
            embedding_hashes[emb_hash].append({
                'doc_id': row['doc_id'],
                'section_id': row['section_id'],
                'title': row['title'],
                'content_preview': row['text'][:100] if row['text'] else 'No content'
            })
        
        analysis = {
            'total_documents': total_docs,
            'unique_embeddings': len(embedding_hashes),
            'duplicate_percentage': ((total_docs - len(embedding_hashes)) / total_docs * 100) if total_docs > 0 else 0,
            'largest_duplicate_group': max(len(docs) for docs in embedding_hashes.values()) if embedding_hashes else 0,
            'duplicate_groups': [
                {
                    'hash': emb_hash[:12] + '...',
                    'count': len(docs),
                    'sample_docs': docs[:3]
                }
                for emb_hash, docs in embedding_hashes.items() 
                if len(docs) > 1
            ][:5]  # Top 5 duplicate groups
        }
        
        return analysis
    
    def get_embedding_jobs(self, test_mode: bool = False, limit: int = None) -> List[EmbeddingJob]:
        """Get list of documents that need embedding regeneration"""
        logger.info("📋 Identifying documents needing embedding regeneration...")
        
        # In test mode, just get a few diverse documents
        if test_mode:
            query = """
                SELECT doc_id, section_id, title, text, embedding
                FROM documents 
                WHERE text IS NOT NULL AND text != ''
                ORDER BY length(text) DESC
                LIMIT 10
            """
        else:
            query = """
                SELECT doc_id, section_id, title, text, embedding
                FROM documents 
                WHERE text IS NOT NULL AND text != ''
                ORDER BY doc_id, section_id
            """
            if limit:
                query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query)
        jobs = []
        
        for row in cursor.fetchall():
            # Create content for embedding
            content = self._prepare_content_for_embedding(row['title'], row['text'])
            
            # Get current embedding hash
            current_hash = hashlib.md5(row['embedding'].encode()).hexdigest() if row['embedding'] else 'none'
            
            jobs.append(EmbeddingJob(
                doc_id=row['doc_id'],
                section_id=row['section_id'],
                content=content,
                title=row['title'] or 'No title',
                current_hash=current_hash
            ))
        
        logger.info(f"📊 Found {len(jobs)} documents to process")
        return jobs
    
    def _prepare_content_for_embedding(self, title: str, text: str) -> str:
        """Prepare content for embedding - this is crucial for quality"""
        # Clean and prepare content
        title = title or ""
        text = text or ""
        
        # Combine title and text meaningfully
        if title and text:
            content = f"{title}\n\n{text}"
        else:
            content = title or text
        
        # Clean up common issues
        content = content.replace('\x00', '')  # Remove null bytes
        content = ' '.join(content.split())    # Normalize whitespace
        
        # Truncate if too long (Mistral has limits)
        if len(content) > 8000:  # Conservative limit
            content = content[:8000] + "..."
            logger.debug(f"Truncated content for embedding: {len(content)} chars")
        
        return content
    
    def _rate_limit_check(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.minute_start > 60:
            self.minute_start = current_time
            self.request_count = 0
        
        # Check if we need to wait
        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.info(f"⏳ Rate limiting: waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.minute_start = time.time()
                self.request_count = 0
        
        # Small delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < 0.6:  # ~1 request per second max
            time.sleep(0.6 - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def generate_single_embedding(self, content: str) -> Optional[List[float]]:
        """Generate embedding for a single piece of content"""
        self._rate_limit_check()
        
        payload = {
            "model": self.model,
            "input": [content]
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data['data'][0]['embedding']
                logger.debug(f"✅ Generated embedding: {len(embedding)} dimensions")
                return embedding
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            return None
    
    def test_embedding_generation(self) -> bool:
        """Test embedding generation with sample content"""
        logger.info("🧪 Testing embedding generation...")
        
        test_contents = [
            "SEC. 28-121. RESTAURANT LICENSING REQUIREMENTS. All restaurants must comply with health codes.",
            "SEC. 51A-4.200. COMMERCIAL SERVICE DISTRICT REGULATIONS. CS district permits retail and service uses.",
            "Table showing zoning requirements for different district types and permitted uses."
        ]
        
        embeddings = []
        for i, content in enumerate(test_contents):
            logger.info(f"🧪 Testing content {i+1}/3...")
            embedding = self.generate_single_embedding(content)
            
            if embedding:
                embeddings.append(embedding)
                logger.info(f"✅ Test {i+1}: Generated {len(embedding)}-dim embedding")
            else:
                logger.error(f"❌ Test {i+1}: Failed to generate embedding")
                return False
        
        # Check if embeddings are different
        if len(set(str(emb) for emb in embeddings)) == len(embeddings):
            logger.info("✅ All test embeddings are unique - API is working correctly!")
            return True
        else:
            logger.error("❌ Test embeddings are identical - API issue persists!")
            return False
    
    def regenerate_embeddings(self, jobs: List[EmbeddingJob], batch_size: int = 50) -> Dict:
        """Regenerate embeddings for all jobs"""
        logger.info(f"🔄 Starting embedding regeneration for {len(jobs)} documents...")
        
        results = {
            'success_count': 0,
            'failure_count': 0,
            'updated_count': 0,
            'failed_docs': []
        }
        
        for i, job in enumerate(jobs):
            logger.info(f"🔄 Processing {i+1}/{len(jobs)}: {job.doc_id}-{job.section_id}")
            
            # Generate new embedding
            new_embedding = self.generate_single_embedding(job.content)
            
            if new_embedding:
                # Check if it's actually different
                new_hash = hashlib.md5(json.dumps(new_embedding).encode()).hexdigest()
                
                if new_hash != job.current_hash:
                    # Update database
                    try:
                        self.conn.execute("""
                            UPDATE documents 
                            SET embedding = ?
                            WHERE doc_id = ? AND section_id = ?
                        """, (json.dumps(new_embedding), job.doc_id, job.section_id))
                        
                        results['updated_count'] += 1
                        logger.debug(f"✅ Updated embedding for {job.doc_id}-{job.section_id}")
                    except Exception as e:
                        logger.error(f"❌ Database update failed: {e}")
                        results['failure_count'] += 1
                        results['failed_docs'].append(f"{job.doc_id}-{job.section_id}")
                else:
                    logger.warning(f"⚠️  New embedding identical to old one: {job.doc_id}-{job.section_id}")
                
                results['success_count'] += 1
            else:
                results['failure_count'] += 1
                results['failed_docs'].append(f"{job.doc_id}-{job.section_id}")
            
            # Commit periodically
            if (i + 1) % batch_size == 0:
                self.conn.commit()
                logger.info(f"💾 Committed batch {(i+1)//batch_size}")
        
        # Final commit
        self.conn.commit()
        logger.info(f"✅ Regeneration complete: {results['updated_count']} updated, {results['failure_count']} failed")
        
        return results
    
    def verify_fix(self) -> Dict:
        """Verify that the embedding fix worked"""
        logger.info("✅ Verifying embedding fix...")
        
        analysis = self.analyze_duplicate_problem()
        
        logger.info(f"📊 Post-fix analysis:")
        logger.info(f"   Total documents: {analysis['total_documents']}")
        logger.info(f"   Unique embeddings: {analysis['unique_embeddings']}")
        logger.info(f"   Duplicate percentage: {analysis['duplicate_percentage']:.2f}%")
        logger.info(f"   Largest duplicate group: {analysis['largest_duplicate_group']}")
        
        return analysis
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix duplicate embedding issues")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--test-mode", action="store_true", help="Test with small sample first")
    parser.add_argument("--regenerate-all", action="store_true", help="Regenerate all embeddings")
    parser.add_argument("--verify-only", action="store_true", help="Only verify current state")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"❌ Database not found: {args.db}")
        return 1
    
    try:
        regenerator = EmbeddingRegenerator(args.db)
        
        if args.verify_only:
            # Just analyze current state
            analysis = regenerator.analyze_duplicate_problem()
            print(f"\n📊 CURRENT STATE:")
            print(f"   Total documents: {analysis['total_documents']}")
            print(f"   Unique embeddings: {analysis['unique_embeddings']}")
            print(f"   Duplicate percentage: {analysis['duplicate_percentage']:.2f}%")
            return 0
        
        # Analyze current problem
        analysis = regenerator.analyze_duplicate_problem()
        print(f"\n🔍 INITIAL ANALYSIS:")
        print(f"   Duplicate percentage: {analysis['duplicate_percentage']:.2f}%")
        print(f"   Largest duplicate group: {analysis['largest_duplicate_group']}")
        
        if analysis['duplicate_percentage'] < 10:
            print("✅ Embedding quality looks good, no fix needed!")
            return 0
        
        # Test embedding generation
        if not regenerator.test_embedding_generation():
            print("❌ Embedding generation test failed - check API configuration")
            return 1
        
        # Get jobs
        jobs = regenerator.get_embedding_jobs(
            test_mode=args.test_mode,
            limit=args.limit
        )
        
        if not jobs:
            print("❌ No documents found to process")
            return 1
        
        if args.test_mode:
            print(f"🧪 TEST MODE: Processing {len(jobs)} sample documents")
        elif not args.regenerate_all and len(jobs) > 100:
            print(f"⚠️  Found {len(jobs)} documents to process.")
            print("   Use --regenerate-all to confirm processing all documents")
            print("   Use --limit N to process only N documents")
            return 0
        
        # Regenerate embeddings
        results = regenerator.regenerate_embeddings(jobs, args.batch_size)
        
        print(f"\n📊 REGENERATION RESULTS:")
        print(f"   Successfully processed: {results['success_count']}")
        print(f"   Updated embeddings: {results['updated_count']}")
        print(f"   Failed: {results['failure_count']}")
        
        if results['failed_docs']:
            print(f"   Failed documents: {results['failed_docs'][:5]}...")
        
        # Verify fix
        if results['updated_count'] > 0:
            final_analysis = regenerator.verify_fix()
            print(f"\n✅ FINAL STATE:")
            print(f"   Duplicate percentage: {final_analysis['duplicate_percentage']:.2f}%")
            
            if final_analysis['duplicate_percentage'] < 10:
                print("🎉 SUCCESS: Embedding duplication fixed!")
            else:
                print("⚠️  Still have duplicate issues - may need further investigation")
    
    except Exception as e:
        logger.error(f"Error during embedding regeneration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
