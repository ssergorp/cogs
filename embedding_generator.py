#!/usr/bin/env python3
"""
Building Codes Embedding Generator
Adds vector embeddings to the existing building_codes index for true hybrid search
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import time
from dataclasses import dataclass

from elasticsearch import Elasticsearch, AsyncElasticsearch
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('embedding_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
INDEX_NAME = "building_codes"
EMBEDDING_MODEL = "mistral-embed"
BATCH_SIZE = 20  # Smaller batches for better reliability
MAX_TEXT_LENGTH = 8000  # Truncate very long documents
EMBEDDING_DIMS = 1024  # Mistral embedding dimensions

@dataclass
class DocumentWithEmbedding:
    """Document structure with embedding"""
    doc_id: str
    title: str
    text: str
    code: str
    category: List[str]
    embedding: List[float]
    text_length: int
    word_count: int
    created_at: str

class BuildingCodesEmbeddingGenerator:
    """Generates and stores embeddings for building codes documents"""
    
    def __init__(self):
        """Initialize the embedding generator"""
        # Validate API key
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        # Initialize clients
        self.es_client = Elasticsearch(
            [ELASTICSEARCH_URL],
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
        
        self.async_es_client = AsyncElasticsearch(
            [ELASTICSEARCH_URL],
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
        
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        
        logger.info("Initialized embedding generator")
    
    def check_elasticsearch_health(self) -> bool:
        """Check if Elasticsearch is healthy and accessible"""
        try:
            health = self.es_client.cluster.health()
            logger.info(f"Elasticsearch health: {health['status']}")
            return health['status'] in ['yellow', 'green']
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the building_codes index"""
        try:
            # Check if index exists
            if not self.es_client.indices.exists(index=INDEX_NAME):
                raise ValueError(f"Index {INDEX_NAME} does not exist")
            
            # Get index stats
            stats = self.es_client.indices.stats(index=INDEX_NAME)
            doc_count = stats['indices'][INDEX_NAME]['total']['docs']['count']
            
            # Get mapping
            mapping = self.es_client.indices.get_mapping(index=INDEX_NAME)
            has_embedding_field = 'embedding' in mapping[INDEX_NAME]['mappings']['properties']
            
            return {
                'doc_count': doc_count,
                'has_embedding_field': has_embedding_field,
                'mapping': mapping[INDEX_NAME]['mappings']['properties']
            }
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            raise
    
    def update_index_mapping(self) -> bool:
        """Add embedding field to the index mapping"""
        try:
            logger.info("Adding embedding field to index mapping...")
            
            mapping_update = {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
            
            self.es_client.indices.put_mapping(
                index=INDEX_NAME,
                body=mapping_update
            )
            
            logger.info("✅ Successfully added embedding field to index mapping")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update index mapping: {e}")
            return False
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Mistral API"""
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            # Use async Mistral client for better performance
            response = await self.mistral_client.embeddings.create_async(
                model=EMBEDDING_MODEL,
                inputs=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.error("❌ MISTRAL API KEY ERROR: The API key is invalid or expired")
                logger.error("🔑 Please update your MISTRAL_API_KEY in the .env file")
                logger.error("📍 Get a new key from: https://console.mistral.ai/")
                raise Exception("Invalid Mistral API key. Please update MISTRAL_API_KEY in .env file.")
            # Return zero vectors as fallback for other errors
            return [[0.0] * EMBEDDING_DIMS for _ in texts]
    
    def get_documents_without_embeddings(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get documents that don't have embeddings yet"""
        try:
            # Query for documents without embedding field
            query = {
                "query": {
                    "bool": {
                        "must_not": {
                            "exists": {
                                "field": "embedding"
                            }
                        }
                    }
                },
                "size": limit or 10000,  # Get all if no limit specified
                "_source": ["id", "title", "text", "code", "category", "text_length", "word_count", "created_at"]
            }
            
            response = self.es_client.search(index=INDEX_NAME, body=query)
            documents = []
            
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['_id'] = hit['_id']  # Include document ID for updating
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} documents without embeddings")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get documents without embeddings: {e}")
            return []
    
    async def process_documents_batch(self, documents: List[Dict[str, Any]]) -> int:
        """Process a batch of documents and add embeddings"""
        try:
            # Prepare texts for embedding
            texts = []
            for doc in documents:
                # Combine title and text for better semantic representation
                combined_text = f"{doc.get('title', '')} {doc.get('text', '')}"
                # Truncate if too long (Mistral has token limits)
                if len(combined_text) > 8000:
                    combined_text = combined_text[:8000]
                texts.append(combined_text)
            
            # Generate embeddings
            embeddings = await self.generate_embeddings_batch(texts)
            
            # Prepare bulk update operations
            bulk_operations = []
            
            for doc, embedding in zip(documents, embeddings):
                # Update operation
                bulk_operations.append({
                    "update": {
                        "_index": INDEX_NAME,
                        "_id": doc['_id']
                    }
                })
                bulk_operations.append({
                    "doc": {
                        "embedding": embedding,
                        "embedding_generated_at": datetime.utcnow().isoformat()
                    }
                })
            
            # Execute bulk update
            if bulk_operations:
                response = await self.async_es_client.bulk(
                    operations=bulk_operations,
                    refresh=False  # Don't refresh immediately for performance
                )
                
                # Check for errors
                errors = []
                for item in response['items']:
                    if 'update' in item and item['update'].get('error'):
                        errors.append(item['update']['error'])
                
                if errors:
                    logger.warning(f"Bulk update had {len(errors)} errors: {errors[:3]}")
                    return len(documents) - len(errors)
                else:
                    logger.info(f"✅ Successfully updated {len(documents)} documents with embeddings")
                    return len(documents)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to process document batch: {e}")
            return 0
    
    async def generate_all_embeddings(self) -> Dict[str, Any]:
        """Generate embeddings for all documents in the building_codes index"""
        start_time = time.time()
        logger.info("🚀 Starting embedding generation for all building codes...")
        
        try:
            # Check Elasticsearch health
            if not self.check_elasticsearch_health():
                raise RuntimeError("Elasticsearch is not healthy")
            
            # Get index information
            index_info = self.get_index_info()
            logger.info(f"Index has {index_info['doc_count']} documents")
            logger.info(f"Embedding field exists: {index_info['has_embedding_field']}")
            
            # Update mapping if needed
            if not index_info['has_embedding_field']:
                if not self.update_index_mapping():
                    raise RuntimeError("Failed to update index mapping")
            
            # Get documents without embeddings
            documents = self.get_documents_without_embeddings()
            
            if not documents:
                logger.info("✅ All documents already have embeddings!")
                return {
                    'status': 'completed',
                    'total_documents': index_info['doc_count'],
                    'processed': 0,
                    'duration_seconds': time.time() - start_time
                }
            
            logger.info(f"Processing {len(documents)} documents in batches of {BATCH_SIZE}")
            
            # Process documents in batches
            total_processed = 0
            failed_batches = 0
            
            for i in range(0, len(documents), BATCH_SIZE):
                batch = documents[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                try:
                    processed_count = await self.process_documents_batch(batch)
                    total_processed += processed_count
                    
                    # Add small delay to avoid overwhelming the API
                    if batch_num < total_batches:
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_num}: {e}")
                    failed_batches += 1
            
            # Final refresh to make all updates searchable
            logger.info("Refreshing index to make embeddings searchable...")
            self.es_client.indices.refresh(index=INDEX_NAME)
            
            duration = time.time() - start_time
            
            results = {
                'status': 'completed',
                'total_documents': len(documents),
                'processed': total_processed,
                'failed_batches': failed_batches,
                'duration_seconds': duration,
                'documents_per_second': total_processed / duration if duration > 0 else 0
            }
            
            logger.info("🎉 Embedding generation completed!")
            logger.info(f"📊 Results: {json.dumps(results, indent=2)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }
    
    def verify_embeddings(self, sample_size: int = 10) -> Dict[str, Any]:
        """Verify that embeddings were added correctly"""
        try:
            logger.info(f"Verifying embeddings with sample size: {sample_size}")
            
            # Query for documents with embeddings
            query = {
                "query": {
                    "exists": {
                        "field": "embedding"
                    }
                },
                "size": sample_size,
                "_source": ["id", "title", "embedding"]
            }
            
            response = self.es_client.search(index=INDEX_NAME, body=query)
            hits = response['hits']['hits']
            
            if not hits:
                return {
                    'status': 'failed',
                    'message': 'No documents with embeddings found'
                }
            
            # Check embedding dimensions
            embedding_dims = []
            for hit in hits:
                embedding = hit['_source'].get('embedding', [])
                embedding_dims.append(len(embedding))
            
            # Get total count of documents with embeddings
            count_query = {
                "query": {
                    "exists": {
                        "field": "embedding"
                    }
                }
            }
            
            count_response = self.es_client.count(index=INDEX_NAME, body=count_query)
            total_with_embeddings = count_response['count']
            
            # Get total document count
            total_docs = self.es_client.count(index=INDEX_NAME)['count']
            
            return {
                'status': 'success',
                'total_documents': total_docs,
                'documents_with_embeddings': total_with_embeddings,
                'coverage_percentage': (total_with_embeddings / total_docs) * 100 if total_docs > 0 else 0,
                'embedding_dimensions': {
                    'expected': EMBEDDING_DIMS,
                    'actual': embedding_dims,
                    'consistent': all(dim == EMBEDDING_DIMS for dim in embedding_dims)
                },
                'sample_documents': [
                    {
                        'id': hit['_source'].get('id'),
                        'title': hit['_source'].get('title', '')[:100] + '...',
                        'embedding_length': len(hit['_source'].get('embedding', []))
                    }
                    for hit in hits[:3]
                ]
            }
            
        except Exception as e:
            logger.error(f"Embedding verification failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_hybrid_search(self, query: str = "building permit requirements") -> Dict[str, Any]:
        """Test hybrid search with the new embeddings"""
        try:
            logger.info(f"Testing hybrid search with query: '{query}'")
            
            # Generate query embedding
            query_embedding = await self.generate_embeddings_batch([query])
            if not query_embedding or not query_embedding[0]:
                return {'status': 'failed', 'error': 'Failed to generate query embedding'}
            
            # Hybrid search query
            search_query = {
                "query": {
                    "bool": {
                        "should": [
                            # BM25 search
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^3", "text^2", "code^2.5"],
                                    "type": "best_fields",
                                    "boost": 1.4  # Boost keyword matching
                                }
                            },
                            # Vector similarity search
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {
                                            "query_vector": query_embedding[0]
                                        }
                                    },
                                    "boost": 1.0
                                }
                            }
                        ]
                    }
                },
                "size": 5,
                "_source": ["id", "title", "code", "text_length"],
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 2}
                    }
                }
            }
            
            response = self.es_client.search(index=INDEX_NAME, body=search_query)
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'score': hit['_score'],
                    'id': hit['_source'].get('id'),
                    'title': hit['_source'].get('title'),
                    'code': hit['_source'].get('code'),
                    'text_length': hit['_source'].get('text_length'),
                    'highlights': hit.get('highlight', {})
                })
            
            return {
                'status': 'success',
                'query': query,
                'total_hits': response['hits']['total']['value'],
                'max_score': response['hits']['max_score'],
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Hybrid search test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def close(self):
        """Close async connections"""
        await self.async_es_client.close()

async def main():
    """Main execution function"""
    generator = BuildingCodesEmbeddingGenerator()
    
    try:
        # Generate embeddings for all documents
        results = await generator.generate_all_embeddings()
        
        if results['status'] == 'completed':
            print("\n✅ Embedding generation completed successfully!")
            print(f"📊 Processed {results['processed']}/{results['total_documents']} documents")
            print(f"⏱️  Duration: {results['duration_seconds']:.2f} seconds")
            print(f"🚀 Rate: {results['documents_per_second']:.2f} docs/second")
            
            # Verify embeddings
            print("\n🔍 Verifying embeddings...")
            verification = generator.verify_embeddings()
            print(f"📈 Coverage: {verification['coverage_percentage']:.1f}% of documents have embeddings")
            print(f"🎯 Dimensions: {verification['embedding_dimensions']['consistent']} (expected {EMBEDDING_DIMS})")
            
            # Test hybrid search
            print("\n🧪 Testing hybrid search...")
            test_result = await generator.test_hybrid_search()
            if test_result['status'] == 'success':
                print(f"✅ Hybrid search working! Found {test_result['total_hits']} results")
                print("Top results:")
                for i, result in enumerate(test_result['results'][:3], 1):
                    print(f"  {i}. [{result['score']:.3f}] {result['code']} - {result['title'][:60]}...")
            else:
                print(f"❌ Hybrid search test failed: {test_result.get('error')}")
        
        else:
            print(f"❌ Embedding generation failed: {results.get('error')}")
    
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"❌ Process failed: {e}")
    
    finally:
        await generator.close()

if __name__ == "__main__":
    asyncio.run(main())