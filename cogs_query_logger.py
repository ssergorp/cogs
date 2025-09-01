#!/usr/bin/env python3
"""
COGS Query Logger - Elasticsearch Integration
Logs COGS query performance data to Elasticsearch for real-time monitoring
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from elasticsearch import Elasticsearch

log = logging.getLogger(__name__)

@dataclass
class QueryLogEntry:
    """Structure for COGS query log entries"""
    query_id: str
    timestamp: str
    query_text: str
    strategy_used: str
    confidence_score: float
    response_time_ms: int
    results_count: int
    fallback_used: bool
    user_ip: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    # Query analysis details
    query_type: Optional[str] = None
    detected_codes: Optional[list] = None
    categories: Optional[list] = None
    word_count: Optional[int] = None
    
    # Performance metrics
    elasticsearch_time_ms: Optional[int] = None
    vector_search_time_ms: Optional[int] = None
    embedding_generation_time_ms: Optional[int] = None

class COGSQueryLogger:
    """Elasticsearch-based query logger for COGS monitoring"""
    
    def __init__(self, 
                 es_hosts: list = None,
                 index_name: str = "cogs_queries",
                 enabled: bool = True):
        self.enabled = enabled
        self.index_name = index_name
        
        if not self.enabled:
            log.info("COGS query logging disabled")
            return
            
        # Initialize Elasticsearch client
        self.es_hosts = es_hosts or ["http://localhost:9200"]
        
        try:
            self.client = Elasticsearch(
                self.es_hosts,
                request_timeout=10,
                retry_on_timeout=True,
                max_retries=3,
                headers={"Accept": "application/json", "Content-Type": "application/json"}
            )
            
            # Create index if it doesn't exist
            self._ensure_index_exists()
            log.info(f"COGS query logger initialized with index: {self.index_name}")
            
        except Exception as e:
            log.error(f"Failed to initialize COGS query logger: {e}")
            self.enabled = False
    
    def _ensure_index_exists(self):
        """Create the COGS queries index with appropriate mapping"""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "query_id": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "query_text": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "strategy_used": {"type": "keyword"},
                        "confidence_score": {"type": "float"},
                        "response_time_ms": {"type": "integer"},
                        "results_count": {"type": "integer"},
                        "fallback_used": {"type": "boolean"},
                        "success": {"type": "boolean"},
                        "user_ip": {"type": "ip"},
                        "query_type": {"type": "keyword"},
                        "detected_codes": {"type": "keyword"},
                        "categories": {"type": "keyword"},
                        "word_count": {"type": "integer"},
                        "elasticsearch_time_ms": {"type": "integer"},
                        "vector_search_time_ms": {"type": "integer"},
                        "embedding_generation_time_ms": {"type": "integer"},
                        "error_message": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "5s"
                }
            }
            
            self.client.indices.create(index=self.index_name, body=mapping)
            log.info(f"Created COGS queries index: {self.index_name}")
    
    def log_query_start(self, query: str, user_ip: str = None) -> str:
        """Log the start of a query and return query ID"""
        if not self.enabled:
            return str(uuid.uuid4())
        
        query_id = str(uuid.uuid4())
        
        try:
            # Just store basic info for now, will update with results later
            initial_entry = {
                "query_id": query_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_text": query[:500],  # Truncate very long queries
                "user_ip": user_ip,
                "status": "started"
            }
            
            # Store in a temporary dict for completion later
            if not hasattr(self, '_pending_queries'):
                self._pending_queries = {}
            
            self._pending_queries[query_id] = initial_entry
            
            log.debug(f"Started tracking query {query_id}")
            return query_id
            
        except Exception as e:
            log.error(f"Failed to log query start: {e}")
            return query_id
    
    def log_query_completion(self, 
                           query_id: str,
                           strategy_used: str,
                           confidence_score: float,
                           response_time_ms: int,
                           results_count: int,
                           fallback_used: bool,
                           query_analysis: Dict[str, Any] = None,
                           performance_metrics: Dict[str, Any] = None,
                           error_message: str = None):
        """Log the completion of a query with full details"""
        if not self.enabled:
            return
        
        try:
            # Get the initial entry
            initial_entry = getattr(self, '_pending_queries', {}).get(query_id, {})
            
            # Create complete log entry
            log_entry = QueryLogEntry(
                query_id=query_id,
                timestamp=initial_entry.get('timestamp', datetime.now(timezone.utc).isoformat()),
                query_text=initial_entry.get('query_text', ''),
                strategy_used=strategy_used,
                confidence_score=confidence_score,
                response_time_ms=response_time_ms,
                results_count=results_count,
                fallback_used=fallback_used,
                user_ip=initial_entry.get('user_ip'),
                success=error_message is None,
                error_message=error_message
            )
            
            # Add query analysis details if provided
            if query_analysis:
                log_entry.query_type = query_analysis.get('query_type')
                log_entry.detected_codes = query_analysis.get('detected_codes')
                log_entry.categories = query_analysis.get('categories')
                log_entry.word_count = query_analysis.get('word_count')
            
            # Add performance metrics if provided
            if performance_metrics:
                log_entry.elasticsearch_time_ms = performance_metrics.get('elasticsearch_time_ms')
                log_entry.vector_search_time_ms = performance_metrics.get('vector_search_time_ms')
                log_entry.embedding_generation_time_ms = performance_metrics.get('embedding_generation_time_ms')
            
            # Index to Elasticsearch
            self.client.index(
                index=self.index_name,
                id=query_id,
                body=asdict(log_entry)
            )
            
            # Clean up pending queries
            if hasattr(self, '_pending_queries') and query_id in self._pending_queries:
                del self._pending_queries[query_id]
            
            log.debug(f"Logged query completion for {query_id}")
            
        except Exception as e:
            log.error(f"Failed to log query completion: {e}")
    
    def get_query_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get query statistics for the last N hours"""
        if not self.enabled:
            return {}
        
        try:
            # Query for recent stats
            query = {
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": f"now-{hours}h"
                        }
                    }
                },
                "aggs": {
                    "strategy_usage": {
                        "terms": {"field": "strategy_used"}
                    },
                    "avg_confidence": {
                        "avg": {"field": "confidence_score"}
                    },
                    "avg_response_time": {
                        "avg": {"field": "response_time_ms"}
                    },
                    "success_rate": {
                        "terms": {"field": "success"}
                    },
                    "fallback_rate": {
                        "terms": {"field": "fallback_used"}
                    }
                },
                "size": 0
            }
            
            response = self.client.search(index=self.index_name, body=query)
            
            stats = {
                "total_queries": response['hits']['total']['value'],
                "avg_confidence": response['aggregations']['avg_confidence']['value'],
                "avg_response_time_ms": response['aggregations']['avg_response_time']['value'],
                "strategy_usage": {
                    bucket['key']: bucket['doc_count'] 
                    for bucket in response['aggregations']['strategy_usage']['buckets']
                },
                "success_rate": None,
                "fallback_rate": None
            }
            
            # Calculate success rate
            success_buckets = response['aggregations']['success_rate']['buckets']
            total = sum(bucket['doc_count'] for bucket in success_buckets)
            if total > 0:
                success_count = next(
                    (bucket['doc_count'] for bucket in success_buckets if bucket['key']), 0
                )
                stats['success_rate'] = success_count / total
            
            # Calculate fallback rate  
            fallback_buckets = response['aggregations']['fallback_rate']['buckets']
            total = sum(bucket['doc_count'] for bucket in fallback_buckets)
            if total > 0:
                fallback_count = next(
                    (bucket['doc_count'] for bucket in fallback_buckets if bucket['key']), 0
                )
                stats['fallback_rate'] = fallback_count / total
            
            return stats
            
        except Exception as e:
            log.error(f"Failed to get query stats: {e}")
            return {}
    
    def create_kibana_visualizations(self):
        """Create suggested Kibana visualizations for COGS monitoring"""
        if not self.enabled:
            return
        
        suggestions = [
            {
                "name": "COGS Query Volume Over Time",
                "type": "Line chart",
                "index": self.index_name,
                "x_axis": "timestamp (date histogram)",
                "y_axis": "count of queries",
                "description": "Track query volume trends"
            },
            {
                "name": "Strategy Usage Distribution", 
                "type": "Pie chart",
                "index": self.index_name,
                "aggregation": "terms on strategy_used",
                "description": "See which search strategies are used most"
            },
            {
                "name": "Confidence Score Distribution",
                "type": "Histogram",
                "index": self.index_name,
                "field": "confidence_score",
                "description": "Analyze confidence score patterns"
            },
            {
                "name": "Response Time Performance",
                "type": "Line chart",
                "index": self.index_name,
                "x_axis": "timestamp",
                "y_axis": "average response_time_ms",
                "description": "Monitor system performance over time"
            },
            {
                "name": "Failed Queries Analysis",
                "type": "Data table",
                "index": self.index_name,
                "filter": "success:false",
                "columns": "query_text, error_message, timestamp",
                "description": "Debug failed queries"
            },
            {
                "name": "Popular Query Terms",
                "type": "Word cloud",
                "index": self.index_name,
                "field": "query_text",
                "description": "Identify common search patterns"
            }
        ]
        
        log.info(f"Suggested Kibana visualizations for {self.index_name} index:")
        for suggestion in suggestions:
            print(f"\n📊 {suggestion['name']}")
            print(f"   Type: {suggestion['type']}")
            print(f"   Config: {suggestion.get('field', suggestion.get('aggregation', suggestion.get('x_axis', 'N/A')))}")
            print(f"   Purpose: {suggestion['description']}")

# Global logger instance
query_logger = COGSQueryLogger()

def get_query_logger() -> COGSQueryLogger:
    """Get the global query logger instance"""
    return query_logger

# Test the logger
if __name__ == "__main__":
    logger = COGSQueryLogger()
    
    if logger.enabled:
        # Test query logging
        query_id = logger.log_query_start("test query", "127.0.0.1")
        
        logger.log_query_completion(
            query_id=query_id,
            strategy_used="elasticsearch_hybrid",
            confidence_score=0.85,
            response_time_ms=150,
            results_count=5,
            fallback_used=False,
            query_analysis={
                "query_type": "natural_language",
                "word_count": 4,
                "categories": ["permits"]
            }
        )
        
        print("✅ Test query logged successfully")
        
        # Show suggested visualizations
        logger.create_kibana_visualizations()
        
        # Get stats
        stats = logger.get_query_stats()
        print(f"\n📈 Current stats: {stats}")
    else:
        print("❌ Query logger not enabled")