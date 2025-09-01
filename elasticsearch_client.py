#!/usr/bin/env python3
"""
COGS Elasticsearch Integration Client
Connects to the building_codes index for advanced search capabilities
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio
from elasticsearch import AsyncElasticsearch, Elasticsearch
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ElasticsearchResult:
    """Standardized result from Elasticsearch"""
    id: str
    title: str
    text: str
    code: str
    category: List[str]
    text_length: int
    word_count: int
    score: float
    highlight: Optional[Dict[str, List[str]]] = None

@dataclass
class SearchStrategy:
    """Defines search strategy metadata"""
    name: str
    confidence_multiplier: float
    description: str

class COGSElasticsearchClient:
    """
    Elasticsearch client optimized for COGS building code searches
    Supports multiple search strategies with confidence scoring
    """
    
    def __init__(self, 
                 hosts: List[str] = None, 
                 index_name: str = "building_codes",
                 timeout: int = 30):
        """Initialize Elasticsearch client"""
        
        self.hosts = hosts or [os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")]
        self.index_name = index_name
        self.timeout = timeout
        
        # Initialize sync and async clients with proper API compatibility
        client_config = {
            "request_timeout": self.timeout,
            "retry_on_timeout": True,
            "max_retries": 3,
            "headers": {"Accept": "application/json", "Content-Type": "application/json"}
        }
        
        self.client = Elasticsearch(self.hosts, **client_config)
        self.async_client = AsyncElasticsearch(self.hosts, **client_config)
        
        # Initialize Mistral client for embeddings
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY")) if os.getenv("MISTRAL_API_KEY") else None
        
        # Search strategies with confidence multipliers
        self.strategies = {
            "exact_code": SearchStrategy(
                name="exact_code",
                confidence_multiplier=1.0,
                description="Exact section code match"
            ),
            "exact_title": SearchStrategy(
                name="exact_title", 
                confidence_multiplier=0.95,
                description="Exact title phrase match"
            ),
            "semantic_search": SearchStrategy(
                name="semantic_search",
                confidence_multiplier=0.8,
                description="Multi-field semantic search"
            ),
            "fuzzy_search": SearchStrategy(
                name="fuzzy_search",
                confidence_multiplier=0.7,
                description="Fuzzy matching for typos"
            ),
            "category_filtered": SearchStrategy(
                name="category_filtered",
                confidence_multiplier=0.85,
                description="Category-specific search"
            )
        }
        
        logger.info(f"Initialized COGS Elasticsearch client for index: {self.index_name}")
    
    def health_check(self) -> bool:
        """Check if Elasticsearch is healthy"""
        try:
            health = self.client.cluster.health()
            return health['status'] in ['yellow', 'green']
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False
    
    def exact_code_search(self, code: str, size: int = 10) -> List[ElasticsearchResult]:
        """
        Search for exact section codes (e.g., "48C-46", "Section 12.5")
        Highest confidence strategy
        """
        try:
            query = {
                "query": {
                    "bool": {
                        "should": [
                            # Exact code match (highest priority)
                            {
                                "term": {
                                    "code.keyword": {
                                        "value": code,
                                        "boost": 3.0
                                    }
                                }
                            },
                            # Code as phrase in text
                            {
                                "match_phrase": {
                                    "text": {
                                        "query": code,
                                        "boost": 2.0
                                    }
                                }
                            },
                            # Code in title
                            {
                                "match_phrase": {
                                    "title": {
                                        "query": code,
                                        "boost": 2.5
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 3},
                        "code": {}
                    }
                },
                "size": size
            }
            
            response = self.client.search(index=self.index_name, body=query)
            return self._parse_results(response, "exact_code")
            
        except Exception as e:
            logger.error(f"Exact code search failed for '{code}': {e}")
            return []
    
    def exact_title_search(self, title: str, size: int = 10) -> List[ElasticsearchResult]:
        """
        Search for exact title phrases
        High confidence strategy
        """
        try:
            query = {
                "query": {
                    "match_phrase": {
                        "title": {
                            "query": title,
                            "boost": 2.0
                        }
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 2}
                    }
                },
                "size": size
            }
            
            response = self.client.search(index=self.index_name, body=query)
            return self._parse_results(response, "exact_title")
            
        except Exception as e:
            logger.error(f"Exact title search failed for '{title}': {e}")
            return []
    
    def semantic_search(self, query_text: str, size: int = 20) -> List[ElasticsearchResult]:
        """
        Multi-field semantic search for natural language queries
        Medium-high confidence strategy
        """
        try:
            query = {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "title^3",      # Title gets highest weight
                            "text^2",       # Text content
                            "code^2.5",     # Code sections
                            "category^1.5"  # Categories
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "minimum_should_match": "75%"
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 3},
                        "code": {}
                    }
                },
                "size": size
            }
            
            response = self.client.search(index=self.index_name, body=query)
            return self._parse_results(response, "semantic_search")
            
        except Exception as e:
            logger.error(f"Semantic search failed for '{query_text}': {e}")
            return []
    
    def category_filtered_search(self, query_text: str, category: str, size: int = 15) -> List[ElasticsearchResult]:
        """
        Search within specific categories (e.g., "building", "fire_safety")
        Medium-high confidence when category is known
        """
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["title^2", "text", "code^1.5"]
                                }
                            }
                        ],
                        "filter": [
                            {
                                "term": {
                                    "category": category
                                }
                            }
                        ]
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 3}
                    }
                },
                "size": size
            }
            
            response = self.client.search(index=self.index_name, body=query)
            return self._parse_results(response, "category_filtered")
            
        except Exception as e:
            logger.error(f"Category filtered search failed for '{query_text}' in '{category}': {e}")
            return []
    
    def fuzzy_search(self, query_text: str, fuzziness: str = "AUTO", size: int = 15) -> List[ElasticsearchResult]:
        """
        Fuzzy search for handling typos and variations
        Lower confidence strategy
        """
        try:
            query = {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["title^2", "text", "code^1.5"],
                        "fuzziness": fuzziness,
                        "prefix_length": 2,
                        "max_expansions": 50
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 2}
                    }
                },
                "size": size
            }
            
            response = self.client.search(index=self.index_name, body=query)
            return self._parse_results(response, "fuzzy_search")
            
        except Exception as e:
            logger.error(f"Fuzzy search failed for '{query_text}': {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[ElasticsearchResult]:
        """Get specific document by ID"""
        try:
            response = self.client.get(index=self.index_name, id=doc_id)
            source = response['_source']
            
            return ElasticsearchResult(
                id=source.get('id', doc_id),
                title=source.get('title', ''),
                text=source.get('text', ''),
                code=source.get('code', ''),
                category=source.get('category', []),
                text_length=source.get('text_length', 0),
                word_count=source.get('word_count', 0),
                score=1.0  # Direct retrieval gets max score
            )
            
        except Exception as e:
            logger.error(f"Document retrieval failed for ID '{doc_id}': {e}")
            return None
    
    def get_available_categories(self) -> List[str]:
        """Get all available categories for filtering"""
        try:
            query = {
                "aggs": {
                    "categories": {
                        "terms": {
                            "field": "category",
                            "size": 100
                        }
                    }
                },
                "size": 0
            }
            
            response = self.client.search(index=self.index_name, body=query)
            buckets = response['aggregations']['categories']['buckets']
            return [bucket['key'] for bucket in buckets]
            
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def _parse_results(self, response: Dict[str, Any], strategy: str) -> List[ElasticsearchResult]:
        """Parse Elasticsearch response into standardized results"""
        results = []
        strategy_conf = self.strategies.get(strategy, self.strategies["semantic_search"])
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Calculate confidence score
            es_score = hit['_score']
            max_score = response['hits']['max_score'] or 1.0
            normalized_score = (es_score / max_score) * strategy_conf.confidence_multiplier
            
            result = ElasticsearchResult(
                id=source.get('id', ''),
                title=source.get('title', ''),
                text=source.get('text', ''),
                code=source.get('code', ''),
                category=source.get('category', []),
                text_length=source.get('text_length', 0),
                word_count=source.get('word_count', 0),
                score=min(normalized_score, 1.0),  # Cap at 1.0
                highlight=hit.get('highlight', {})
            )
            
            results.append(result)
        
        return results
    
    async def async_search(self, query_text: str, strategy: str = "semantic_search", **kwargs) -> List[ElasticsearchResult]:
        """Async version of search methods"""
        try:
            if strategy == "exact_code":
                return await self._async_exact_code_search(query_text, **kwargs)
            elif strategy == "exact_title":
                return await self._async_exact_title_search(query_text, **kwargs)
            elif strategy == "semantic_search":
                return await self._async_semantic_search(query_text, **kwargs)
            elif strategy == "fuzzy_search":
                return await self._async_fuzzy_search(query_text, **kwargs)
            else:
                return await self._async_semantic_search(query_text, **kwargs)
                
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return []
    
    async def _async_semantic_search(self, query_text: str, size: int = 20) -> List[ElasticsearchResult]:
        """Async semantic search"""
        query = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^3", "text^2", "code^2.5", "category^1.5"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "text": {"fragment_size": 150, "number_of_fragments": 3}
                }
            },
            "size": size
        }
        
        response = await self.async_client.search(index=self.index_name, body=query)
        return self._parse_results(response, "semantic_search")
    
    async def hybrid_vector_search(self, query: str, size: int = 10) -> List[ElasticsearchResult]:
        """
        Hybrid search combining BM25 keyword matching with vector similarity
        Uses the approach from your simplified hybrid_rag_search.py
        """
        if not self.mistral_client:
            logger.warning("Mistral client not available, falling back to semantic search")
            return self.semantic_search(query, size)
        
        try:
            # Generate query embedding
            embed_response = await self.mistral_client.embeddings.create_async(
                model="mistral-embed", 
                inputs=[query]
            )
            query_vector = embed_response.data[0].embedding
            
            # Note: Since building_codes index doesn't have embeddings yet,
            # we'll combine BM25 search with semantic matching for now
            # TODO: Add embeddings to building_codes index for true hybrid search
            
            query_body = {
                "query": {
                    "bool": {
                        "should": [
                            # BM25 keyword matching (higher weight)
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^3", "text^2", "code^2.5"],
                                    "type": "best_fields",
                                    "boost": 2.0
                                }
                            },
                            # Fuzzy matching for semantic similarity approximation
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^2", "text"],
                                    "fuzziness": "AUTO",
                                    "boost": 1.0
                                }
                            }
                        ]
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 3},
                        "code": {}
                    }
                },
                "size": size
            }
            
            response = await self.async_client.search(
                index=self.index_name, 
                body=query_body
            )
            
            # Parse results with hybrid confidence scoring
            results = []
            max_score = response['hits']['max_score'] or 1.0
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                
                # Hybrid confidence calculation
                bm25_score = hit['_score'] / max_score
                # For now, use BM25 score as base confidence
                # When we add vectors, this will combine BM25 + cosine similarity
                confidence = bm25_score * 0.9  # Hybrid strategy multiplier
                
                result = ElasticsearchResult(
                    id=source.get('id', ''),
                    title=source.get('title', ''),
                    text=source.get('text', ''),
                    code=source.get('code', ''),
                    category=source.get('category', []),
                    text_length=source.get('text_length', 0),
                    word_count=source.get('word_count', 0),
                    score=min(confidence, 1.0),
                    highlight=hit.get('highlight', {})
                )
                
                results.append(result)
            
            logger.info(f"Hybrid search returned {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid vector search failed for '{query}': {e}")
            # Fallback to regular semantic search
            return self.semantic_search(query, size)
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Mistral"""
        if not self.mistral_client:
            raise ValueError("Mistral client not initialized")
        
        try:
            response = await self.mistral_client.embeddings.create_async(
                model="mistral-embed",
                inputs=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def close(self):
        """Close async client connection"""
        await self.async_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Sync client doesn't need explicit closing
        pass

# Test the client
if __name__ == "__main__":
    import asyncio
    
    async def test_client():
        client = COGSElasticsearchClient()
        
        # Test health check
        print(f"Elasticsearch healthy: {client.health_check()}")
        
        # Test different search strategies
        test_queries = [
            ("48C-46", "exact_code"),
            ("building permit", "semantic_search"), 
            ("fire safety requirements", "semantic_search"),
            ("noise ordinance", "fuzzy_search")
        ]
        
        for query, strategy in test_queries:
            print(f"\n=== Testing {strategy} with query: '{query}' ===")
            
            if strategy == "exact_code":
                results = client.exact_code_search(query, size=3)
            elif strategy == "semantic_search":
                results = client.semantic_search(query, size=3)
            elif strategy == "fuzzy_search":
                results = client.fuzzy_search(query, size=3)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result.score:.3f}] {result.code} - {result.title[:60]}...")
                if result.highlight:
                    for field, highlights in result.highlight.items():
                        print(f"   {field}: {highlights[0][:100]}...")
        
        # Test categories
        print(f"\nAvailable categories: {client.get_available_categories()[:10]}")
        
        await client.close()
    
    asyncio.run(test_client())